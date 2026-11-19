"""Small single-GPU A6 loop. Local and Modal runs call exactly this function."""

from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import numpy as np
import torch
from torch.nn import functional as F
from optimizers import build_masked_weight_decay_parameter_groups
from . import models, optimizers
from .common import data


def autocast(device):
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def nll(logits, ids):
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    )


@torch.no_grad()
def evaluate(model, rows, batch=8, device='cuda', predictions=None):
    model.eval()
    losses = []
    for start in range(0, len(rows), batch):
        ids = torch.as_tensor(
            np.array(rows[start : start + batch]), dtype=torch.long, device=device
        )
        with autocast(device):
            logits = model(ids).float()
        chunk = (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
                reduction='none',
            )
            .reshape(len(ids), -1)
            .mean(1)
        )
        if not torch.isfinite(chunk).all():
            raise FloatingPointError('Nonfinite validation loss')
        losses.extend(chunk.cpu().tolist())
        if predictions is not None:
            predictions[start : start + len(ids)] = logits[:, :-1].cpu().numpy()
    return float(np.mean(losses))


def _log_wandb(config, payload, finish=False):
    if not os.environ.get('WANDB_API_KEY'):
        return
    import wandb

    if wandb.run is None:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT') or 'assignments',
            entity=os.environ.get('WANDB_ENTITY') or None,
            name=config.name,
            config=asdict(config),
            reinit=True,
        )
    wandb.log(payload)
    if finish:
        wandb.finish()


def _parameter_report(model, config):
    stored = sum(p.numel() for p in model.parameters())
    report = dict(stored_parameters=int(stored), architecture=config.architecture)
    if config.architecture == 'moe':
        expert = sum(
            p.numel()
            for module in model.modules()
            if module.__class__.__name__ == 'Experts'
            for p in module.parameters()
        )
        report.update(
            expert_parameters=int(expert),
            active_parameters=int(stored - expert + expert // 4),
            experts=4,
            top_k=1,
        )
    elif config.architecture == 'loop':
        report.update(
            executed_depth=int(model.executed_depth),
            core_size=int(model.core_size),
            passes=int(model.passes),
            unique_blocks=int(2 + model.core_size),
        )
    return report


def _diagnostics(model, config, train_rows, validation, device):
    report = _parameter_report(model, config)
    if config.probe_sharpness:
        from .p3_optimization import directional_sharpness

        report['sharpness'] = directional_sharpness(
            model, np.array(train_rows[:4]), radius=0.05
        )
    if config.architecture == 'moe':
        was = model.training
        model.eval()
        for block in model.model.layers:
            block.mlp.record = True
        ids = torch.as_tensor(np.array(validation[:8]), dtype=torch.long, device=device)
        with torch.no_grad(), autocast(device):
            model(ids)
        layers = [block.mlp.stats for block in model.model.layers]
        from experiments.a5_capacity.diagnostics import summarize_routers

        report['routing'] = summarize_routers(layers)
        report['routing_layers'] = layers
        model.train(was)
    return report


def train(config, *, root=None, output=None, device='cuda'):
    c = config
    if c.schedule not in ('linear', 'constant') or c.epochs < 1 or c.batch < 1 or c.micro_batch < 1:
        raise ValueError('Use linear/constant schedules, positive epochs and batch sizes.')
    train_rows, val = data(c, root)
    torch.set_num_threads(4)
    model = models.build(c, root).to(device)
    optimizer = optimizers.build(model, c)
    rates = [g['lr'] for g in optimizer.param_groups]
    decay = build_masked_weight_decay_parameter_groups(model, c.wd)[0]['params']
    initial = (
        {id(p): p.detach().clone() for p in decay}
        if c.adaptation == 'toward_initialization'
        else {}
    )
    order = np.random.default_rng(c.data_seed).permutation(len(train_rows))
    per_epoch = math.ceil(len(train_rows) / c.batch)
    total = per_epoch * c.epochs
    warmup = min(math.ceil(4800 / c.batch), total - 1)
    if output is None:
        from utils import MODEL_DIR

        output = Path(MODEL_DIR) / 'a6' / c.name
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model.pt').exists():
        raise FileExistsError('Choose a new run name/output directory.')
    history = []
    initial_nll = evaluate(model, val, device=device)
    (output / 'initial.json').write_text(
        json.dumps(dict(step=0, epochs=0.0, validation_nll=initial_nll)) + '\n'
    )
    print(dict(step=0, epochs=0.0, validation_nll=initial_nll), flush=True)
    for step in range(total):
        # Native schedule semantics: step0 LR0, common 4,800-sequence warmup.
        if warmup and step < warmup:
            multiplier = step / warmup
        elif c.schedule == 'constant':
            multiplier = 1.0
        else:
            multiplier = max(0.0, (total - step) / (total - warmup))
        for g, rate in zip(optimizer.param_groups, rates):
            g['lr'] = rate * multiplier
        start = (step % per_epoch) * c.batch
        rows = train_rows[order[start : start + c.batch]]
        model.train()

        def backward():
            for offset in range(0, len(rows), c.micro_batch):
                ids = torch.as_tensor(
                    np.array(rows[offset : offset + c.micro_batch]), dtype=torch.long, device=device
                )
                with autocast(device):
                    logits = model(ids)
                    loss = nll(logits, ids)
                    if hasattr(model, 'auxiliary_loss'):
                        loss = loss + model.auxiliary_loss()
                    loss = loss * len(ids) / len(rows)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()

        optimizer.zero_grad(set_to_none=True)
        rng = (
            torch.cuda.get_rng_state() if str(device).startswith('cuda') else torch.get_rng_state()
        )
        backward()
        if c.optimizer == 'sam':
            ps = [p for p in model.parameters() if p.grad is not None]
            norm = torch.stack([p.grad.float().norm() for p in ps]).norm()
            saved = [p.detach().clone() for p in ps]
            with torch.no_grad():
                for p in ps:
                    p.add_(p.grad, alpha=float(c.rho / (norm + 1e-12)))
            optimizer.zero_grad(set_to_none=True)
            if str(device).startswith('cuda'):
                torch.cuda.set_rng_state(rng)
            else:
                torch.set_rng_state(rng)
            try:
                backward()
            finally:
                with torch.no_grad():
                    for p, value in zip(ps, saved):
                        p.copy_(value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if initial:
            # p <- p - eta*lambda*(p - p0) on AdamW's decay set. The optimizer itself has lambda 0.
            alpha = -optimizer.param_groups[0]['lr'] * c.wd
            with torch.no_grad():
                for p in decay:
                    p.add_(p - initial[id(p)], alpha=alpha)
        optimizer.step()
        if hasattr(model, 'finish_step'):
            model.finish_step()
        if (step + 1) % max(1, total // 16) == 0 or step + 1 == total:
            point = dict(
                step=step + 1,
                epochs=(step + 1) / per_epoch,
                validation_nll=evaluate(model, val, device=device),
            )
            history.append(point)
            (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(point, flush=True)
            _log_wandb(c, point)
    diagnostics = _diagnostics(model, c, train_rows, val, device)
    (output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    if c.save_logits:
        predictions = np.empty((len(val), c.context - 1, len(model.lm_head.weight)), np.float32)
        evaluate(model, val, device=device, predictions=predictions)
        np.savez_compressed(output / 'validation_logits.npz', logits=predictions)
    torch.save(dict(model=model.cpu().state_dict(), config=asdict(c)), output / 'model.pt')
    (output / 'config.json').write_text(json.dumps(asdict(c), indent=2) + '\n')
    (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    _log_wandb(c, dict(final_validation_nll=history[-1]['validation_nll']), finish=True)
    return history


def run(config):
    from modal_utils import MODAL_MODEL_DIR, user_volume

    try:
        return train(config, output=Path(MODAL_MODEL_DIR) / 'a6' / config.name)
    finally:
        user_volume.commit()


def launch(config, gpu='H100'):
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT

    remote = app.function(
        image=build_image(), volumes=VOLUME_MOUNTS, gpu=gpu, cpu=4, memory=32768, timeout=24 * 3600
    )(run)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
        call = remote.spawn(config)
        print('Modal call:', call.object_id)
        return call.object_id
