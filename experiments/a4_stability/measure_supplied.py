"""Probe supplied P1/P2 checkpoints on Modal. Does not train.

Writes one JSON per run under results/a4_stability/probes/.
"""
import json
from pathlib import Path

RUNS = [
    # Best final checkpoint of each supplied architecture.
    "a4-d8-lr0.003",
    "a4-d8-gated-attention-lr0.006",
    "a4-d8-gated-norm-lr0.00225",
    "a4-d8-gated-residual-lr0.003",
    "a4-d8-no-final-norm-lr0.003",
    "a4-d8-no-prenorm-lr0.001",
    "a4-d8-no-qknorm-lr0.003",
    "a4-d8-no-residual-lr0.00075",
    "a4-d8-no-rope-lr0.003",
    "a4-d8-postnorm-lr0.001",
    "a4-rnn-lr0.002",
    "a4-rnn-final-norm-lr0.001",
    "a4-rnn-gated-norm-lr0.001",
    "a4-rnn-gated-output-lr0.001",
    "a4-rnn-gated-residual-lr0.001",
    "a4-rnn-no-prenorm-lr0.002",
    "a4-rnn-no-residual-lr0.001",
    "a4-rnn-postnorm-lr0.001",
    "a4-rnn-rope-lr0.002",
    "a4-rnn1-lr0.003",
    "a4-rnn4-lr0.001",
    # Same runs at step 100, where the manifest has an early/ checkpoint.
    "a4-d8-lr0.003/early",
    "a4-d8-no-residual-lr0.00075/early",
    "a4-d8-no-prenorm-lr0.001/early",
    "a4-d8-no-qknorm-lr0.003/early",
    "a4-d8-postnorm-lr0.001/early",
    "a4-rnn-lr0.002/early",
    "a4-rnn4-lr0.001/early",
    "a4-rnn1-lr0.003/early",
    "a4-rnn-no-residual-lr0.001/early",
    "a4-rnn-no-prenorm-lr0.002/early",
]


def measure_runs(runs=RUNS):
    import traceback
    import torch
    from data import dclm_val_dataset, load_token_dataset
    from experiments.a4_stability.diagnostics import profile, context_loss
    from experiments.a4_stability.measure_default import REFERENCE_ROOT, measure
    from model_io import load_model

    validation = load_token_dataset(dclm_val_dataset(), role="val", data_seed=42)
    tokens = torch.tensor(validation[:1]["input_ids"].astype("int64"), device="cuda")
    # Keep the official single-run helper honest on the default final checkpoint.
    del measure, load_model, profile, context_loss
    out = {}
    root = REFERENCE_ROOT
    from experiments.a4_stability.diagnostics import profile as profile_fn, context_loss as context_fn
    from model_io import load_model as load_fn
    for run in runs:
        try:
            model = load_fn(root / run, device="cuda", precision="fp32")
            stats = profile_fn(model, tokens)
            full = context_fn(model, tokens, keep=tokens.shape[1] - 1)
            stats["context_loss_increase"] = {
                f"{keep}_first_{first}": (context_fn(model, tokens, keep, first) - full).tolist()
                for keep in (16, 64, 256) for first in (False, True)}
            stats["n_parameters"] = sum(p.numel() for p in model.parameters())
            out[run] = stats
            print(f"OK {run} full_loss={stats['full']['loss']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:
            out[run] = {"error": traceback.format_exc()}
            print(f"FAIL {run}", flush=True)
    return out


if __name__ == "__main__":
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image
    from experiments.a4_stability.diagnostics import save

    probe = app.function(
        image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=4 * 60 * 60,
    )(measure_runs)
    output = Path("results/a4_stability/probes")
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        measured = probe.remote()
    for run, stats in measured.items():
        save(output / f"{run.replace('/', '__')}.json", stats)
    print(f"wrote {len(measured)} probes to {output}")
