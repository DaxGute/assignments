"""Eager-mode GRU gate / routing re-measure. Writes results/a4_stability/gaps/gates.json."""
import json
import traceback
from pathlib import Path

RUNS = [
    "a4-gru-lr0.006",
    "a4-gru-lr0.003",
    "a4-gru-no-reset-lr0.003",
    "a4-gru-no-update-lr0.003",
    "a4-gru-time-residual-lr0.003",
    "a4-d8-attention-lr0.003",
    "a4-d8-identity_mix-lr0.003",
    "a4-d8-block4-lr0.003",
]


def measure():
    import math
    import torch
    from data import dclm_val_dataset, load_token_dataset
    from experiments.a4_stability.diagnostics import profile
    from experiments.a4_stability.measure_gaps import _gate_stats, _routing_summary
    from model_io import load_model

    validation = load_token_dataset(dclm_val_dataset(), role="val", data_seed=42)
    tokens = torch.tensor(validation[:8]["input_ids"].astype("int64"), device="cuda")
    ckpt_root = Path("/root/data/ckpts")
    shared = Path("/root/shared_data/a4_stability")
    out = {}

    for run in RUNS:
        try:
            model = None
            errors = []
            for p in (ckpt_root / run, shared / run):
                try:
                    model = load_model(p, device="cuda", precision="fp32")
                    break
                except Exception as e:
                    errors.append(f"{p}: {e}")
            if model is None:
                raise FileNotFoundError("; ".join(errors))
            # Prefer eager for hooks
            try:
                import torch._dynamo as dynamo
                dynamo.reset()
            except Exception:
                pass
            stats = profile(model, tokens[:1])
            row = {
                "loss0": stats["full"]["loss"],
                "routing_summary": _routing_summary(stats.get("routing") or {}),
                "gate_summary": _gate_stats(model, tokens[:1]),
            }
            # Multi-seq mean entropy over layers with >=2 sources
            ents, maxs = [], []
            for r in row["routing_summary"]:
                if r["n_sources"] >= 2:
                    ents.append(r["entropy"])
                    maxs.append(r["max"])
            if ents:
                row["routing_mean_entropy_ge2"] = sum(ents) / len(ents)
                row["routing_mean_max_ge2"] = sum(maxs) / len(maxs)
            out[run] = row
            print(f"OK {run} gates={row['gate_summary']} route_ent={row.get('routing_mean_entropy_ge2')}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:
            out[run] = {"error": traceback.format_exc()}
            print(f"FAIL {run}", flush=True)
    return out


def measure_and_persist():
    """Run measure() and also write JSON onto the Modal volume so a dropped client cannot lose it."""
    measured = measure()
    out = Path("/root/data/a4_stability_gaps")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gates.json"
    path.write_text(json.dumps(measured, indent=2))
    print(f"persisted {path}", flush=True)
    return measured


if __name__ == "__main__":
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image, user_volume
    from experiments.a4_stability.diagnostics import save

    fn = app.function(
        image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=60 * 60,
    )(measure_and_persist)
    out_dir = Path("results/a4_stability/gaps")
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        measured = fn.remote()
        try:
            user_volume.commit()
        except Exception as e:
            print("volume commit warn", e, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(out_dir / "gates.json", measured)
    print(json.dumps({k: (v if "error" not in v else {"error": "yes"}) for k, v in measured.items()}, indent=2)[:3000])
