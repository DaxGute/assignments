"""Problem 4: edit the temporal update, keeping the modern RNN's depth stack.

    uv run python -m experiments.a4_stability.p4_recurrent_updates

z is the fraction of NEW information, following the handout. Reset is applied
BEFORE the candidate's recurrent projection; torch.nn.GRU uses a different form.
"""
import torch
from torch import nn
from .p2_rnn_depth import TanhCell, ModernRNN


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return self.input(x)

    def forward(self, projected, h):
        return (projected + self.hidden(h)).sigmoid()


class GRUCell(TanhCell):
    def __init__(self, config):
        super().__init__(config)
        self.reset = self.make_reset(config)
        self.update = self.make_update(config)

    def make_reset(self, config):
        return Gate(config)

    def make_update(self, config):
        return Gate(config)

    def project(self, x):
        return self.input(x), self.reset.project(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(r*h))
        return self.combine(h, candidate, z)

    def combine(self, h, candidate, z):
        return (1-z)*h + z*candidate


class GRU(ModernRNN):
    cell_type = GRUCell


class NoResetGRUCell(GRUCell):
    """Drop the reset gate: the candidate always sees the full previous state."""
    def make_reset(self, config):
        return None

    def project(self, x):
        return self.input(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(h))
        return self.combine(h, candidate, z)


class NoUpdateGRUCell(GRUCell):
    """Force z = 1, so the carried state is replaced by the reset-gated candidate."""
    def make_update(self, config):
        return None

    def project(self, x):
        return self.input(x), self.reset.project(x)

    def step(self, projected, h):
        candidate_input, reset_input = projected
        r = self.reset(reset_input, h)
        return torch.tanh(candidate_input + self.hidden(r * h))


class RetainGRU(GRU):
    """Same GRU, but the update gate starts near 0.1 so early steps retain h."""
    def initialize_parameters(self):
        super().initialize_parameters()
        # sigmoid(-2.197224577) = 0.1. Input bias only; recurrent bias stays 0.
        for module in self.modules():
            if isinstance(module, GRUCell):
                nn.init.constant_(module.update.input.bias, -2.197224577)
                nn.init.zeros_(module.update.hidden.bias)


class TimeResidualCell(TanhCell):
    """h <- h + a * tanh(Wx + Uh), with scalar a = sigmoid(w x) in (0, 1).

    The temporal Jacobian contains I. Forgetting requires cancellation, unlike
    the GRU convex combination.
    """
    def __init__(self, config):
        super().__init__(config)
        self.gate = nn.Linear(config.hidden_size, 1)

    def project(self, x):
        return self.input(x), self.gate(x)

    def step(self, projected, h):
        candidate_input, gate_input = projected
        update = torch.tanh(candidate_input + self.hidden(h))
        return h + gate_input.sigmoid() * update


class NoResetGRU(GRU):
    cell_type = NoResetGRUCell


class NoUpdateGRU(GRU):
    cell_type = NoUpdateGRUCell


class TimeResidualRNN(ModernRNN):
    cell_type = TimeResidualCell


MODELS = dict(gru=GRU, tanh=ModernRNN, no_reset=NoResetGRU, no_update=NoUpdateGRU,
              retain=RetainGRU, time_residual=TimeResidualRNN)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('gru')  # Edit the variant and LR here.
