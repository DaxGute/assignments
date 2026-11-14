"""Fill missing diagnostics for report v2: probes, gates, context truncation.

Does not train. Writes results/a4_stability/gaps/*.json
"""
import json
import traceback
from pathlib import Path

PROBE_RUNS = [
    # High-LR collapses
    "a4-d8-no-qknorm-lr0.006",
    "a4-d8-no-qknorm-lr0.012",
    "a4-d8-postnorm-lr0.004",
    # P3 routing
    "a4-d8-attention-lr0.003",
    "a4-d8-attention-lr0.0015",
    "a4-d8-uniform-lr0.003",
    "a4-d8-identity_mix-lr0.003",
    "a4-d8-block4-lr0.003",
    "a4-d8-ema09-lr0.003",
    "a4-d8-half-lr0.003",
    # P4 cells
    "a4-gru-lr0.006",
    "a4-gru-lr0.003",
    "a4-gru-no-reset-lr0.003",
    "a4-gru-no-update-lr0.003",
    "a4-gru-time-residual-lr0.003",
    # Init-ish early where present
    "a4-d8-lr0.003/early",
    "a4-d8-no-residual-lr0.00075/early",
    "a4-rnn-lr0.002/early",
]

CONTEXT_RUNS = [
    "a4-d8-lr0.003",
    "a4-d8-attention-lr0.003",
    "a4-gru-lr0.006",
    "a4-rnn-lr0.002",
]


def _gate_stats(model, tokens):
    """Bias summary + empirical gate means on a forward (GRU / time-residual).

    Avoids .item() inside hooks (breaks torch.compile). Collects CPU tensors after
    the forward; works under compiled or eager modules.
    """
    import torch
    from experiments.a4_stability.p4_recurrent_updates import (
        GRUCell, TimeResidualCell, Gate,
    )

    out = {"n_cells": 0}
    z_bias, r_bias, a_bias = [], [], []
    cells = []
    for m in model.modules():
        if isinstance(m, (GRUCell,)):
            cells.append(m)
            if getattr(m, "update", None) is not None:
                z_bias.append(m.update.input.bias.detach().float().mean().item())
            if getattr(m, "reset", None) is not None:
                r_bias.append(m.reset.input.bias.detach().float().mean().item())
        if isinstance(m, TimeResidualCell):
            cells.append(m)
            if m.gate.bias is not None:
                a_bias.append(m.gate.bias.detach().float().mean().item())
    out["n_cells"] = len(cells)
    if z_bias:
        out["update_input_bias_mean"] = sum(z_bias) / len(z_bias)
    if r_bias:
        out["reset_input_bias_mean"] = sum(r_bias) / len(r_bias)
    if a_bias:
        out["time_residual_bias_mean"] = sum(a_bias) / len(a_bias)

    # Collect tensors only — convert to floats after forward leaves the compile graph.
    captured = {"z": [], "r": [], "a": []}
    handles = []

    def make_hook(kind):
        def hook(module, args, output):
            # Gate.forward already returns sigmoid; Linear gate needs sigmoid.
            t = output.detach()
            if kind == "a":
                t = t.float().sigmoid()
            else:
                t = t.float()
            captured[kind].append(t.mean().detach().cpu())
        return hook

    for m in model.modules():
        if isinstance(m, GRUCell):
            if getattr(m, "update", None) is not None:
                handles.append(m.update.register_forward_hook(make_hook("z")))
            if getattr(m, "reset", None) is not None:
                handles.append(m.reset.register_forward_hook(make_hook("r")))
        if isinstance(m, TimeResidualCell):
            handles.append(m.gate.register_forward_hook(make_hook("a")))
        # Also hook Gate modules directly in case cell wrappers differ after load.
        if isinstance(m, Gate):
            # Skip if already hooked via cell.update/reset (same module object).
            pass

    try:
        was = model.training
        model.eval()
        try:
            import torch._dynamo as dynamo
            dynamo.reset()
            dynamo.config.cache_size_limit = max(getattr(dynamo.config, "cache_size_limit", 64), 512)
            dynamo.config.suppress_errors = True
        except Exception:
            pass
        with torch.no_grad():
            # torch.compiler.disable is the supported API on recent PyTorch.
            try:
                disable = torch.compiler.disable
            except Exception:
                disable = None
            if disable is not None:
                @disable
                def _fwd(m, x):
                    return m(x)
                _fwd(model, tokens[:, :-1])
            else:
                model(tokens[:, :-1])
        model.train(was)
    except Exception as e:
        out["forward_error"] = f"{type(e).__name__}: {e}"
    finally:
        for h in handles:
            h.remove()

    for k in ("z", "r", "a"):
        if captured[k]:
            vals = [float(t) for t in captured[k]]
            out[f"{k}_mean_over_forward"] = sum(vals) / len(vals)
            out[f"{k}_n_captures"] = len(vals)
            out[f"{k}_min"] = min(vals)
            out[f"{k}_max"] = max(vals)
    return out


def _routing_summary(routing: dict):
    """Compress Softmax router outputs captured by profile()."""
    rows = []
    for name, weights in routing.items():
        # weights: mean over batch/heads already in diagnostics → list over sources
        if not isinstance(weights, list):
            continue
        w = weights
        rows.append({
            "name": name,
            "n_sources": len(w),
            "entropy": -sum(p * __import__("math").log(max(p, 1e-12)) for p in w),
            "max": max(w) if w else None,
            "uniform": 1.0 / len(w) if w else None,
            "weights": w,
        })
    return rows


def measure():
    import math
    import torch
    from data import dclm_val_dataset, load_token_dataset
    from experiments.a4_stability.diagnostics import profile, context_loss
    from model_io import load_model

    validation = load_token_dataset(dclm_val_dataset(), role="val", data_seed=42)
    # Multi-sequence for context: first 32 val rows
    n_ctx = 32
    tokens_multi = torch.tensor(
        validation[:n_ctx]["input_ids"].astype("int64"), device="cuda"
    )
    tokens_one = tokens_multi[:1]

    ckpt_root = Path("/root/data/ckpts")
    shared = Path("/root/shared_data/a4_stability")
    out = {"probes": {}, "context": {}, "gates": {}}

    def resolve(run: str):
        # Student volume first, then course shared a4_stability tree.
        candidates = [ckpt_root / run, shared / run]
        # early checkpoints live as run/early or run with /early suffix path
        errors = []
        for p in candidates:
            try:
                return load_model(p, device="cuda", precision="fp32")
            except Exception as e:
                errors.append(f"{p}: {e}")
        raise FileNotFoundError("; ".join(errors))

    for run in PROBE_RUNS:
        try:
            model = resolve(run)
            # Multi-sequence: average full loss over first 8 val sequences; detailed profile on seq0.
            losses = []
            for i in range(min(8, tokens_multi.shape[0])):
                stats_i = profile(model, tokens_multi[i : i + 1])
                losses.append(stats_i["full"]["loss"])
                if i == 0:
                    stats = stats_i
            stats["n_parameters"] = sum(p.numel() for p in model.parameters())
            stats["full_loss_mean_8seq"] = sum(losses) / len(losses)
            stats["full_loss_std_8seq"] = (
                sum((x - stats["full_loss_mean_8seq"]) ** 2 for x in losses) / len(losses)
            ) ** 0.5
            stats["routing_summary"] = _routing_summary(stats.get("routing") or {})
            try:
                stats["gate_summary"] = _gate_stats(model, tokens_multi[:1])
            except Exception as e:
                stats["gate_summary"] = {"error": str(e)}
            out["probes"][run] = stats
            print(
                f"OK probe {run} loss0={stats['full']['loss']:.4f} mean8={stats['full_loss_mean_8seq']:.4f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        except Exception:
            out["probes"][run] = {"error": traceback.format_exc()}
            print(f"FAIL probe {run}", flush=True)

    for run in CONTEXT_RUNS:
        try:
            model = resolve(run)
            full = context_loss(model, tokens_multi, keep=tokens_multi.shape[1] - 1)
            row = {"n_sequences": int(tokens_multi.shape[0]), "full_mean": float(full.mean())}
            for keep in (16, 64, 256, 512):
                for first in (False, True):
                    inc = context_loss(model, tokens_multi, keep=keep, preserve_first=first) - full
                    row[f"keep{keep}_bos{int(first)}_mean_delta"] = float(inc.mean())
                    row[f"keep{keep}_bos{int(first)}_std_delta"] = float(inc.std(unbiased=False))
            out["context"][run] = row
            print(f"OK context {run} {row}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:
            out["context"][run] = {"error": traceback.format_exc()}
            print(f"FAIL context {run}", flush=True)

    return out


if __name__ == "__main__":
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image
    from experiments.a4_stability.diagnostics import save

    fn = app.function(
        image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=3 * 60 * 60,
    )(measure)
    out_dir = Path("results/a4_stability/gaps")
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        measured = fn.remote()
    out_dir.mkdir(parents=True, exist_ok=True)
    save(out_dir / "gap_fill.json", measured)
    # also split
    for key in ("probes", "context"):
        save(out_dir / f"{key}.json", measured.get(key, {}))
    print(f"wrote {out_dir}")
