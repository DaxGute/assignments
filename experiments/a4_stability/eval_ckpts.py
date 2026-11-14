"""Evaluate finished student checkpoints that lack a W&B val_loss.

    uv run python -m experiments.a4_stability.eval_ckpts
"""
import json
from pathlib import Path

RUNS = [
    "a4-gru-retain-lr0.001",
    "a4-gru-retain-lr0.003",
    "a4-d8-attention-lr0.003",
    "a4-gru-time-residual-lr0.003",
]


def eval_runs(runs=RUNS):
    import traceback
    import torch
    from model_io import load_model
    from train import (
        evaluate, checked_train_config, load_train_and_val_datasets, create_batches,
    )
    from experiments.a4_stability.sweeps import config as make_cfg
    from dataclasses import replace

    out = {}
    for run in runs:
        try:
            rest = run[len("a4-"):]
            i = rest.rfind("-lr")
            variant, lr = rest[:i], float(rest[i + 3:])
            tc = replace(make_cfg(variant, lr), model_dir="/root/data/ckpts")
            tc = checked_train_config(tc)
            model = load_model(f"/root/data/ckpts/{run}", device="cuda")
            _, val_sets = load_train_and_val_datasets(tc)
            batches = create_batches(val_sets["val"], tc.batch_size)
            loss = evaluate(model, batches, tc, torch.device("cuda"))
            out[run] = {
                "val_loss": float(loss),
                "lr": lr,
                "variant": variant,
                "n_parameters": sum(p.numel() for p in model.parameters()),
            }
            print(f"OK {run} val_loss={out[run]['val_loss']:.6f}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:
            out[run] = {"error": traceback.format_exc()}
            print(f"FAIL {run}", flush=True)
    return out


if __name__ == "__main__":
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image

    fn = app.function(
        image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=60 * 60,
    )(eval_runs)
    output = Path("experiments/a4_stability/deliverables/recovered_vals.json")
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        measured = fn.remote()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(measured, indent=2))
    print(json.dumps(measured, indent=2))
    print(f"wrote {output}")
