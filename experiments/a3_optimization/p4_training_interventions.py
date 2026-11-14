"""Training-time reparameterizations for A3 P4(b).

The attention value/output gauge V -> cV, O -> O/c preserves the initialized
network function exactly. Training from gauge-related initializations changes
the optimizer's coordinate system without applying a manual final-checkpoint
intervention, testing whether AdamW preserves or breaks that symmetry.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from modeling import AutoregressiveLM


class GaugeInitializedLM(AutoregressiveLM):
    """Standard course LM initialized in an exact attention V/O gauge."""

    def __init__(self, config, *, gauge_factor: float, **kwargs):
        if gauge_factor <= 0:
            raise ValueError("gauge_factor must be positive")
        self.gauge_factor = float(gauge_factor)
        super().__init__(config, **kwargs)

    def post_initialize(self):
        with torch.no_grad():
            for layer in self.model.layers:
                layer.self_attn.v_proj.weight.mul_(self.gauge_factor)
                layer.self_attn.o_proj.weight.div_(self.gauge_factor)


class _ScaledLinear(torch.nn.Linear):
    """Linear layer whose stored coordinates have a fixed forward scale."""

    gauge_scale: float

    def forward(self, input):
        return F.linear(input, self.weight * self.gauge_scale, self.bias)


class PersistentGaugeLM(AutoregressiveLM):
    """Function-preserving V/O gauge retained throughout training.

    The stored parameters and state-dict names are unchanged.  Only their
    forward coordinate map differs, so a fresh model at the same seed computes
    exactly the standard function while its parameter-space Hessian changes.
    """

    def __init__(self, config, *, gauge_factor: float, **kwargs):
        if gauge_factor <= 0:
            raise ValueError("gauge_factor must be positive")
        self.gauge_factor = float(gauge_factor)
        super().__init__(config, **kwargs)
        for layer in self.model.layers:
            value = layer.self_attn.v_proj
            output = layer.self_attn.o_proj
            value.__class__ = _ScaledLinear
            output.__class__ = _ScaledLinear
            value.gauge_scale = self.gauge_factor
            output.gauge_scale = 1.0 / self.gauge_factor


class _ScaledRMSNorm(torch.nn.Module):
    """RMSNorm whose stored gains have a fixed extra forward scale."""

    def __init__(self, module, gauge_scale: float):
        super().__init__()
        self.weight = module.weight
        self.variance_epsilon = module.variance_epsilon
        self.gauge_scale = float(gauge_scale)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, gauge_scale={self.gauge_scale}"

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * self.gauge_scale) * hidden_states.to(input_dtype)


class PersistentHeadGaugeLM(AutoregressiveLM):
    """Function-preserving final-norm/head gauge retained throughout training.

    Late-training sharpness lives in the head family, not attention.  Because
    logits = W_head (g ⊙ x̂), the map g -> c g and W_head -> W_head / c leaves
    the initialized function unchanged while changing the head-block Hessian.
    """

    def __init__(self, config, *, gauge_factor: float, **kwargs):
        if gauge_factor <= 0:
            raise ValueError("gauge_factor must be positive")
        self.gauge_factor = float(gauge_factor)
        super().__init__(config, **kwargs)
        self.model.norm = _ScaledRMSNorm(self.model.norm, self.gauge_factor)
        self.lm_head.__class__ = _ScaledLinear
        self.lm_head.gauge_scale = 1.0 / self.gauge_factor


class FrozenNormPersistentGaugeLM(PersistentGaugeLM):
    """Persistent V/O gauge with frozen RMSNorm gains.

    The persistent coordinate map was largely washed out by training.  Freezing
    RMSNorm gains at 1 removes a cheap scale-absorbing degree of freedom, so
    the same function-preserving gauge should remain visible in the Hessian.
    """

    def __init__(self, config, *, gauge_factor: float, **kwargs):
        super().__init__(config, gauge_factor=gauge_factor, **kwargs)
        for module in self.modules():
            if isinstance(module, torch.nn.Parameter):
                continue
            weight = getattr(module, "weight", None)
            if (
                weight is not None
                and getattr(module, "variance_epsilon", None) is not None
            ):
                weight.requires_grad_(False)
