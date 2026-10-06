"""Width and depth prescriptions from the Assignment 2 handout.

Problem 4.1 applies these rules inside the five-step stress model.
Problem 4.2 applies the same width and depth multipliers on the course model.
"""

from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn

from modeling import AutoregressiveLM, LlamaDecoderLayer, initialize_model
from optimizers import ADAMW_EPSILON, should_apply_weight_decay


STRESS_REFERENCE_WIDTH = 512
STRESS_REFERENCE_DEPTH = 2
LONG_REFERENCE_WIDTH = 512
LONG_REFERENCE_DEPTH = 8
HIDDEN_LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
BLOCK_NORMS = {"input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"}


def depth_multipliers(policy, depth, reference_depth):
    """Residual scale, block LR multiplier, and block Adam-epsilon multiplier."""
    if policy in {"kaiming", "mup", "baseline"}:
        return 1.0, 1.0, 1.0
    relative = depth / reference_depth
    if policy == "depth_mup":
        scale = relative ** -0.5
        return scale, scale, scale
    if policy == "completep":
        return relative ** -1.0, 1.0, relative ** -1.0
    raise ValueError(f"unknown depth policy {policy!r}")


def stress_spec(policy, width, depth, reference_width, reference_depth):
    """Numeric five-step prescription, independent of parameter names."""
    if policy not in {"kaiming", "mup", "depth_mup", "completep"}:
        raise ValueError(f"unknown stress policy {policy!r}")
    width_ratio = width / reference_width
    if policy == "kaiming":
        residual, block_lr, block_eps = 1.0, 1.0, 1.0
        output_multiplier = 1.0
        hidden_lr_multiplier = 1.0
        readout_variance = 1.0 / width
    else:
        residual, block_lr, block_eps = depth_multipliers(policy, depth, reference_depth)
        output_multiplier = 1.0 / width_ratio
        hidden_lr_multiplier = block_lr / width_ratio
        readout_variance = 1.0 / reference_width
    return {
        "policy": policy,
        "width_ratio": width_ratio,
        "relative_depth": depth / reference_depth,
        "output_multiplier": output_multiplier,
        "residual_multiplier": residual,
        "block_lr_multiplier": block_lr,
        "block_eps_multiplier": block_eps,
        "hidden_lr_multiplier": hidden_lr_multiplier,
        "block_norm_lr_multiplier": block_lr,
        "other_lr_multiplier": 1.0,
        "readout_variance": readout_variance,
        "embedding_variance": 1.0,
        "reference_width": reference_width,
        "reference_depth": reference_depth,
    }


def stress_identity(policy, width, depth, base_lr, reference_width, reference_depth):
    """Training identity with the policy name removed, so identical rules collapse."""
    spec = stress_spec(policy, width, depth, reference_width, reference_depth)
    return {
        "output_multiplier": spec["output_multiplier"],
        "residual_multiplier": spec["residual_multiplier"],
        "hidden_lr": base_lr * spec["hidden_lr_multiplier"],
        "block_norm_lr": base_lr * spec["block_norm_lr_multiplier"],
        "other_lr": base_lr * spec["other_lr_multiplier"],
        "block_eps": ADAMW_EPSILON * spec["block_eps_multiplier"],
        "other_eps": ADAMW_EPSILON,
        "readout_variance": spec["readout_variance"],
        "embedding_variance": spec["embedding_variance"],
        "hidden_variance": "1/fan_in",
    }


def initialize_stress(model, policy, reference_width, reference_depth):
    """Gaussian initialization and forward multipliers for one five-step policy."""
    spec = stress_spec(
        policy,
        model.config.width,
        model.config.depth,
        reference_width,
        reference_depth,
    )
    model.output_multiplier = spec["output_multiplier"]
    for block in model.blocks:
        block.residual_multiplier = spec["residual_multiplier"]
    with torch.no_grad():
        nn.init.normal_(model.embed.weight, mean=0.0, std=1.0)
        for block in model.blocks:
            for name in ("q", "k", "v", "o", "gate", "up", "down"):
                weight = getattr(block, name).weight
                nn.init.normal_(weight, mean=0.0, std=(1.0 / weight.shape[1]) ** 0.5)
            for name in ("norm1", "norm2", "qnorm", "knorm"):
                getattr(block, name).weight.fill_(1.0)
        nn.init.normal_(
            model.head.weight,
            mean=0.0,
            std=spec["readout_variance"] ** 0.5,
        )
        model.norm.weight.fill_(1.0)
    return spec


def stress_groups(model, base_lr, policy, reference_width, reference_depth):
    """Adam groups for the stress runner. Keys are exactly params, lr, and eps."""
    spec = stress_spec(
        policy,
        model.config.width,
        model.config.depth,
        reference_width,
        reference_depth,
    )
    hidden_lr = base_lr * spec["hidden_lr_multiplier"]
    block_norm_lr = base_lr * spec["block_norm_lr_multiplier"]
    other_lr = base_lr * spec["other_lr_multiplier"]
    block_eps = ADAMW_EPSILON * spec["block_eps_multiplier"]
    other_eps = ADAMW_EPSILON
    entries = []
    for block in model.blocks:
        for name in ("q", "k", "v", "o", "gate", "up", "down"):
            entries.append((getattr(block, name).weight, hidden_lr, block_eps))
        for name in ("norm1", "norm2", "qnorm", "knorm"):
            entries.append((getattr(block, name).weight, block_norm_lr, block_eps))
    entries.append((model.embed.weight, other_lr, other_eps))
    entries.append((model.head.weight, other_lr, other_eps))
    entries.append((model.norm.weight, other_lr, other_eps))
    expected = {id(parameter) for parameter in model.parameters()}
    if {id(parameter) for parameter, _, _ in entries} != expected or len(entries) != len(expected):
        raise RuntimeError("stress parameter groups do not match the model")
    buckets = {}
    for parameter, lr, eps in entries:
        buckets.setdefault((lr, eps), []).append(parameter)
    return [
        {"params": parameters, "lr": lr, "eps": eps}
        for (lr, eps), parameters in buckets.items()
    ]


def long_scaling(parameterization, width, depth, reference_width=LONG_REFERENCE_WIDTH, reference_depth=LONG_REFERENCE_DEPTH):
    """Forward and optimizer multipliers for a longer-training run.

    ``baseline`` is the course initialization: embedding G/width and readout
    G/sqrt(width), with no depth multiplier. The muP family shares one
    truncated Gaussian and then rescales it to the reference width.
    """
    if parameterization == "baseline":
        residual, block_lr, block_eps = 1.0, 1.0, 1.0
        return {
            "parameterization": parameterization,
            "reference_width": reference_width,
            "reference_depth": reference_depth,
            "output_multiplier": 1.0,
            "residual_multiplier": residual,
            "block_lr_multiplier": block_lr,
            "block_eps_multiplier": block_eps,
            "hidden_lr_multiplier": 1.0,
            "norm_lr_multiplier": 1.0,
            "width_ratio": width / reference_width,
            "relative_depth": depth / reference_depth,
        }
    if parameterization not in {"mup", "depth_mup", "completep"}:
        raise ValueError(f"unknown parameterization {parameterization!r}")
    width_ratio = width / reference_width
    residual, block_lr, block_eps = depth_multipliers(
        parameterization, depth, reference_depth
    )
    return {
        "parameterization": parameterization,
        "reference_width": reference_width,
        "reference_depth": reference_depth,
        "output_multiplier": 1.0 / width_ratio,
        "residual_multiplier": residual,
        "block_lr_multiplier": block_lr,
        "block_eps_multiplier": block_eps,
        "hidden_lr_multiplier": block_lr / width_ratio,
        "norm_lr_multiplier": block_lr,
        "width_ratio": width_ratio,
        "relative_depth": depth / reference_depth,
    }


def _scaled_layer_forward(layer, hidden_states, position_embeddings, attention_mask=None):
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = layer.self_attn(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    hidden_states = nn.functional.dropout(
        hidden_states, p=layer.dropout, training=layer.training
    )
    hidden_states = residual + layer.residual_multiplier * hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=layer.dropout, training=layer.training
    )
    return residual + layer.residual_multiplier * hidden_states


class ScaledAutoregressiveLM(AutoregressiveLM):
    """Course architecture plus the handout's readout and residual multipliers."""

    def __init__(
        self,
        config,
        dtype=torch.float32,
        qk_norm=True,
        tie_word_embeddings=False,
        dropout=0.0,
        **scaling,
    ):
        if tie_word_embeddings:
            raise ValueError("muP keeps the embedding and readout untied")
        super().__init__(
            config,
            dtype=dtype,
            qk_norm=qk_norm,
            tie_word_embeddings=False,
            dropout=dropout,
        )
        required = (
            "parameterization",
            "reference_width",
            "reference_depth",
            "output_multiplier",
            "residual_multiplier",
            "block_lr_multiplier",
            "block_eps_multiplier",
            "hidden_lr_multiplier",
            "norm_lr_multiplier",
        )
        missing = [key for key in required if key not in scaling]
        if missing:
            raise ValueError(f"scaled model is missing {missing}")
        self.output_multiplier = float(scaling["output_multiplier"])
        self.a2_scaling = {
            "parameterization": str(scaling["parameterization"]),
            "reference_width": int(scaling["reference_width"]),
            "reference_depth": int(scaling["reference_depth"]),
            "output_multiplier": float(scaling["output_multiplier"]),
            "residual_multiplier": float(scaling["residual_multiplier"]),
            "block_lr_multiplier": float(scaling["block_lr_multiplier"]),
            "block_eps_multiplier": float(scaling["block_eps_multiplier"]),
            "hidden_lr_multiplier": float(scaling["hidden_lr_multiplier"]),
            "norm_lr_multiplier": float(scaling["norm_lr_multiplier"]),
        }
        for layer in self.model.layers:
            if not isinstance(layer, LlamaDecoderLayer):
                raise TypeError(f"expected LlamaDecoderLayer, got {type(layer).__name__}")
            layer.residual_multiplier = float(scaling["residual_multiplier"])
            layer.forward = MethodType(_scaled_layer_forward, layer)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.lm_head(hidden_states * self.output_multiplier)

    def initialize_parameters(self):
        initialize_model(self)
        width = self.config.hidden_size
        reference_width = self.a2_scaling["reference_width"]
        embedding = self.get_input_embeddings()
        readout = self.get_output_embeddings()
        with torch.no_grad():
            base = embedding.weight.detach() * width
            embedding.weight.copy_(base / reference_width)
            readout.weight.copy_(base / math.sqrt(reference_width))


def build_scaled_model(
    config,
    dtype=torch.float32,
    qk_norm=True,
    tie_word_embeddings=False,
    dropout=0.0,
    **scaling,
):
    return ScaledAutoregressiveLM(
        config,
        dtype=dtype,
        qk_norm=qk_norm,
        tie_word_embeddings=tie_word_embeddings,
        dropout=dropout,
        **scaling,
    )


def _parameter_roles(model):
    roles = {"hidden": [], "block_norm": [], "readout": [], "undecayed": []}
    seen = set()
    for module_name, module in model.named_modules():
        leaf = module_name.rsplit(".", 1)[-1]
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if leaf in HIDDEN_LINEARS and isinstance(module, nn.Linear):
                roles["hidden"].append(parameter)
            elif leaf in BLOCK_NORMS:
                roles["block_norm"].append(parameter)
            elif leaf == "lm_head" and parameter_name == "weight":
                roles["readout"].append(parameter)
            elif should_apply_weight_decay(module, parameter_name):
                roles["readout"].append(parameter)
            else:
                roles["undecayed"].append(parameter)
    if seen != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError("scaled optimizer did not classify every parameter")
    return roles


def build_scaled_optimizer(
    model,
    optimizer_name,
    learning_rate,
    weight_decay,
    beta1,
    beta2,
    **_unused,
):
    """AdamW with per-role base LRs. The scheduler scales every group together."""
    if optimizer_name != "adamw":
        raise ValueError(f"scaled prescriptions use adamw, got {optimizer_name!r}")
    spec = model.a2_scaling
    roles = _parameter_roles(model)
    block_eps = ADAMW_EPSILON * spec["block_eps_multiplier"]
    assignments = (
        (roles["hidden"], learning_rate * spec["hidden_lr_multiplier"], block_eps, weight_decay),
        (roles["block_norm"], learning_rate * spec["norm_lr_multiplier"], block_eps, 0.0),
        (roles["readout"], learning_rate, ADAMW_EPSILON, weight_decay),
        (roles["undecayed"], learning_rate, ADAMW_EPSILON, 0.0),
    )
    groups = []
    for parameters, lr, eps, decay in assignments:
        if parameters:
            groups.append(
                {"params": parameters, "lr": float(lr), "eps": float(eps), "weight_decay": float(decay)}
            )
    uniform = len({(group["lr"], group["eps"]) for group in groups}) == 1
    cuda = any(parameter.is_cuda for group in groups for parameter in group["params"])
    kwargs = {"betas": (beta1, beta2), "eps": ADAMW_EPSILON, "weight_decay": 0.0}
    if uniform and cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)
