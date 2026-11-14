"""Print class name and weight hashes for student checkpoints."""
import hashlib
import json
from pathlib import Path

RUNS = [
    "a4-d8-block2-lr0.003",
    "a4-d8-attention-lr0.003",
    "a4-d8-attention-lr0.0015",
    "a4-d8-identity_mix-lr0.0015",
    "a4-d8-identity_mix-lr0.003",
    "a4-d8-block2-lr0.009",
    "a4-d8-block4-lr0.003",
]


def probe(runs=RUNS):
    from model_io import load_model

    out = {}
    for run in runs:
        p = Path(f"/root/data/ckpts/{run}")
        if not p.exists():
            out[run] = {"missing": True}
            continue
        model = load_model(str(p), device="cpu", precision="fp32")
        params = list(model.parameters())
        out[run] = {
            "class": type(model).__name__,
            "nparam": sum(p.numel() for p in params),
            "w0": hashlib.sha256(params[0].detach().float().cpu().numpy().tobytes()[:8192]).hexdigest()[:16],
            "w5": hashlib.sha256(params[5].detach().float().cpu().numpy().tobytes()).hexdigest()[:16],
            "files": sorted(x.name for x in p.iterdir()),
        }
        if hasattr(model, "block_size"):
            out[run]["block_size"] = int(model.block_size)
        if hasattr(model, "routers"):
            out[run]["n_routers"] = len(model.routers)
        del model
    return out


if __name__ == "__main__":
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image

    fn = app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=30 * 60)(probe)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        measured = fn.remote()
    print(json.dumps(measured, indent=2))
    Path("experiments/a4_stability/deliverables/ckpt_identity.json").write_text(
        json.dumps(measured, indent=2)
    )
