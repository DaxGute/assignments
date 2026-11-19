"""One short H100 job that times the real d8 step and checks the release mount."""

from pathlib import Path
import json
import time
import numpy as np
import torch

from experiments.a6_generalization.common import VOCAB, manifest, reference
from experiments.a6_generalization.models import build
from experiments.a6_generalization.common import Config
from experiments.a6_generalization.training import nll, autocast


def smoke_remote(steps=16):
    started = time.perf_counter()
    root = Path('/root/shared_data/a6_generalization/v1')
    info = dict(reference_root=str(root), root_exists=root.is_dir())
    try:
        release = manifest()
        info['manifest_keys'] = sorted(release)[:40]
        info['manifest_version'] = {
            key: release.get(key)
            for key in ('version', 'published', 'name', 'schema_version')
            if key in release
        }
        train = np.load(reference('splits/target_30k.npy'), mmap_mode='r')
        validation = np.load(reference('splits/validation.npy'), mmap_mode='r')
        info['train_shape'] = list(train.shape)
        info['validation_shape'] = list(validation.shape)
        info['bos_token'] = VOCAB.index('<bos>')
        info['train_bos_fraction'] = float(np.mean(np.array(train[:, 0]) == info['bos_token']))
        info['split_overlap_rows'] = int(
            len(set(map(bytes, np.array(train[:200]))) & set(map(bytes, np.array(validation[:200]))))
        )
        device = 'cuda'
        config = Config(name='a6-smoke', probe_sharpness=False)
        torch.manual_seed(config.seed)
        model = build(config).to(device)
        from experiments.a6_generalization.optimizers import build as build_optimizer

        optimizer = build_optimizer(model, config)
        batch = np.array(train[: config.batch])
        # Warm the allocator, then time steady steps.
        def one_step(rows):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(rows), config.micro_batch):
                ids = torch.as_tensor(rows[offset : offset + config.micro_batch], dtype=torch.long, device=device)
                with autocast(device):
                    loss = nll(model(ids), ids) * len(ids) / len(rows)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return float(loss.detach())

        one_step(batch)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        losses = [one_step(batch) for _ in range(steps)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        info['timed_steps'] = steps
        info['seconds_per_step'] = elapsed / steps
        info['last_loss'] = losses[-1]
        info['parameters'] = sum(p.numel() for p in model.parameters())
        state = torch.load(
            reference('checkpoints/baseline_d8_e32_s42.pt'), map_location='cpu', weights_only=False
        )['model']
        missing, unexpected = model.cpu().load_state_dict(state, strict=False)
        info['baseline_missing'] = list(missing)
        info['baseline_unexpected'] = list(unexpected)
        for kind in ('moe', 'loop'):
            probe = build(Config(architecture=kind, name=f'smoke-{kind}', probe_sharpness=False)).to(device)
            ids = torch.as_tensor(batch[:2], dtype=torch.long, device=device)
            loss = nll(probe(ids), ids)
            loss.backward()
            info[f'{kind}_parameters'] = sum(p.numel() for p in probe.parameters())
            info[f'{kind}_loss_finite'] = bool(torch.isfinite(loss))
        for name in ('related', 'dclm', 'noise'):
            checkpoint = torch.load(
                reference(f'checkpoints/{name}.pt'), map_location='cpu', weights_only=False
            )['model']
            info[f'{name}_tensors'] = len(checkpoint)
        info['ok'] = True
    except Exception as exc:
        info['ok'] = False
        info['error'] = repr(exc)
    info['wall_seconds'] = time.perf_counter() - started
    from modal_utils import MODAL_MODEL_DIR, user_volume

    output = Path(MODAL_MODEL_DIR) / 'a6' / 'smoke'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result.json').write_text(json.dumps(info, indent=2) + '\n')
    user_volume.commit()
    print(json.dumps(info), flush=True)
    return info


def launch_smoke(steps=16, gpu='H100'):
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets

    remote = app.function(
        image=build_image(),
        volumes=VOLUME_MOUNTS,
        secrets=secrets(include_wandb=False),
        gpu=gpu,
        cpu=4,
        memory=32768,
        timeout=3600,
        max_containers=1,
    )(smoke_remote)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
        call = remote.spawn(steps)
        print('smoke call', call.object_id, flush=True)
        return call.object_id
