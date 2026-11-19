"""Lock predictions, then spawn one Modal app. Do not start a second app."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.a6_generalization.common import Config
from experiments.a6_generalization.ledger import lock_prediction

MANIFEST = Path('results/a6_generalization/run_manifest.jsonl')
BASELINE_NLL = 1.2627138263769448
BASELINE_ID = '0baf962ea57fafacec22'
TOKENS_PER_PASS = 30_000 * 1024


def _base(**kwargs):
    cfg = Config(**kwargs)
    return cfg


def wave1_configs():
    configs = []
    notes = []
    for lr, interval in (
        (0.001, [1.270, 1.310]),
        (0.003, [1.268, 1.295]),
        (0.01, [1.275, 1.360]),
    ):
        name = f'a6-moe-lr{lr:g}-s42'
        configs.append(_base(name=name, architecture='moe', lr=lr))
        notes.append(dict(
            problem='P1b', run_name=name, interval=interval, direction='higher',
            numeric_prediction=sum(interval) / 2,
            changed=dict(architecture='moe', lr=lr),
            mechanism=(
                'Top-1 routing assigns each expert about a quarter of the tokens. '
                'Stored MLP capacity roughly triples, but active MLP compute stays near dense d8. '
                'With 30.72M unique tokens, the split of examples across experts should outweigh the extra stored capacity.'
            ),
            diagnostics_expected='Expert loads stay near 0.25 after quantile balancing; validation NLL above the dense baseline.',
            falsification='Falsify if the best of the three MoE runs is below 1.258 or any finished run exceeds 1.45.',
            confidence='medium',
        ))
    for lr, interval in (
        (0.001, [1.265, 1.290]),
        (0.003, [1.260, 1.280]),
        (0.01, [1.265, 1.320]),
    ):
        name = f'a6-loop-lr{lr:g}-s42'
        configs.append(_base(name=name, architecture='loop', lr=lr))
        notes.append(dict(
            problem='P1c', run_name=name, interval=interval, direction='similar',
            numeric_prediction=sum(interval) / 2,
            changed=dict(architecture='loop', lr=lr),
            mechanism=(
                'RMS+recall looping stores one prelude, one shared core, and one coda, '
                'while still executing eight block applications. Parameter sharing is a capacity constraint. '
                'It should land near dense d8, with a small penalty if the shared block is a bottleneck.'
            ),
            diagnostics_expected='Stored parameter count near 11.4M and executed depth 8. Loss within about 0.02 of dense d8.',
            falsification='Falsify if the best loop run is below 1.255 or above 1.300.',
            confidence='medium',
        ))
    for lr, interval in (
        (0.003, [1.280, 1.360]),
        (0.01, [1.265, 1.320]),
        (0.03, [1.270, 1.360]),
    ):
        name = f'a6-hyperball-lr{lr:g}-s42'
        configs.append(_base(name=name, optimizer='hyperball', lr=lr, wd=0.0))
        notes.append(dict(
            problem='P2a-ii', run_name=name, interval=interval, direction='similar',
            numeric_prediction=sum(interval) / 2,
            changed=dict(optimizer='hyperball', lr=lr, wd=0.0),
            mechanism=(
                'Hyperball projects each linear weight onto its initial Frobenius sphere, replacing decay toward zero. '
                'Tuned AdamW WD 3.2 reaches 1.2627 and untuned WD 0.1 reaches 1.4057 at this horizon. '
                'A tuned radius-preserving update should recover most of that gap, closest to WD 3.2.'
            ),
            diagnostics_expected='Linear-layer norms stay at initialization. Best LR is 0.01 or 0.03, not the AdamW default 0.003.',
            falsification='Falsify if the best Hyperball run is above 1.350 or below 1.255.',
            confidence='medium',
        ))
    for rate, interval, direction in (
        (0.05, [1.260, 1.272], 'similar'),
        (0.15, [1.262, 1.285], 'higher'),
        (0.25, [1.270, 1.310], 'higher'),
    ):
        name = f'a6-dropout-p{rate:g}-s42'
        configs.append(_base(name=name, attention_dropout=rate, save_logits=True))
        notes.append(dict(
            problem='P2b-i', run_name=name, interval=interval, direction=direction,
            numeric_prediction=sum(interval) / 2,
            changed=dict(attention_dropout=rate),
            mechanism=(
                'The WD 3.2, 32-epoch dense run is only mildly worse than its best earlier checkpoint, '
                'so attention dropout has little overfitting left to remove. At 0.25 it should act as optimization noise.'
            ),
            diagnostics_expected='Training NLL rises with the dropout rate. Evaluation disables dropout.',
            falsification='Falsify the single-model claim if 0.05 beats 1.258 or if 0.25 beats the no-dropout baseline.',
            confidence='medium',
        ))
    for rho, interval in (
        (0.01, [1.258, 1.272]),
        (0.05, [1.258, 1.275]),
        (0.1, [1.260, 1.285]),
    ):
        name = f'a6-sam-rho{rho:g}-s42'
        configs.append(_base(name=name, optimizer='sam', rho=rho))
        notes.append(dict(
            problem='P3c', run_name=name, interval=interval, direction='similar',
            numeric_prediction=sum(interval) / 2,
            changed=dict(optimizer='sam', rho=rho),
            mechanism=(
                'SAM steps from the gradient at a loss-increasing perturbation. '
                'The supplied batch sweep already shows that lower directional sharpness is not monotone with lower NLL. '
                'Expect a small or null loss change, with rho 0.1 more likely to overshoot.'
            ),
            diagnostics_expected='Directional sharpness at radius 0.05 falls as rho increases. Wall time is about twice AdamW because of the second backward.',
            falsification='Falsify if any SAM run differs from 1.2627 by more than 0.025.',
            confidence='low',
        ))
    ft_grid = {
        'related': {
            1e-4: [1.12, 1.35],
            3e-4: [1.10, 1.32],
            1e-3: [1.12, 1.40],
            3e-3: [1.20, 1.60],
        },
        'scratch': {
            1e-4: [1.45, 1.90],
            3e-4: [1.35, 1.70],
            1e-3: [1.30, 1.55],
            3e-3: [1.28, 1.50],
        },
        'dclm': {
            1e-4: [1.50, 2.10],
            3e-4: [1.40, 1.90],
            1e-3: [1.35, 1.80],
            3e-3: [1.35, 1.85],
        },
        'noise': {
            1e-4: [1.45, 1.85],
            3e-4: [1.35, 1.65],
            1e-3: [1.30, 1.55],
            3e-3: [1.28, 1.52],
        },
    }
    for init, grid in ft_grid.items():
        for lr, interval in grid.items():
            name = f'a6-ft-{init}-n30000-e1-lr{lr:g}-wd3.2-s42'
            initialization = 'random' if init == 'scratch' else init
            configs.append(_base(
                name=name,
                initialization=initialization,
                epochs=1,
                lr=lr,
                wd=3.2,
                adaptation='full',
            ))
            notes.append(dict(
                problem='P4a', run_name=name, interval=interval,
                direction='lower' if init == 'related' else 'higher',
                numeric_prediction=sum(interval) / 2,
                changed=dict(initialization=initialization, epochs=1, lr=lr, wd=3.2),
                mechanism=(
                    'Related-DNA pretraining used about 10B tokens on eukaryotic genic windows that exclude this pool. '
                    'One target pass should preserve a large head start. DCLM transfers only the transformer body, '
                    'so a random DNA embedding and head should look closer to scratch. '
                    'Nucleotide noise can improve optimization conditioning without transferring sequence features. '
                    'The anchor is the measured 32-epoch scratch baseline, not an unmeasured 1-epoch control.'
                ),
                diagnostics_expected='initial.json shows related well below scratch before any target update. The gap shrinks but remains after one pass.',
                falsification=(
                    'Falsify the related-domain claim if the best related 1-epoch run is above 1.40, '
                    'or if the best DCLM run beats the best related run.'
                ),
                confidence='medium' if init == 'related' else 'low',
            ))
    if len({cfg.name for cfg in configs}) != len(configs):
        raise RuntimeError('Wave-1 names are not unique.')
    return configs, notes


def dense_depth3(lr):
    """P1(d) control: same stored count as the six-pass loop. Lock a prediction only at launch."""
    return _base(
        name=f'a6-dense-d3-lr{lr:g}-s42',
        architecture='dense',
        depth=3,
        width=512,
        lr=lr,
    )


def sgd_grid(lrs=(0.001, 0.003, 0.01)):
    """P3(d). Weight decay 3.2 matches AdamW. SGD folds that decay into the gradient."""
    return [
        _base(name=f'a6-sgd-lr{lr:g}-wd3.2-s42', optimizer='sgd', lr=lr, wd=3.2)
        for lr in lrs
    ]


def adaptation_configs(lr):
    """P4(c) on related DNA at 16 passes. Full fine-tuning at weight decay 3.2 is the baseline.

    Does not freeze the embedding by itself. Does not lock a prediction.
    """
    rows = []
    for wd in (0.0, 0.1, 1.0):
        rows.append(_base(
            name=f'a6-ft-related-n30000-e16-lr{lr:g}-wd{wd:g}-s42',
            initialization='related', epochs=16, lr=lr, wd=wd, adaptation='full',
        ))
    for adaptation in ('toward_initialization', 'frozen_head', 'frozen_body'):
        rows.append(_base(
            name=f'a6-ft-related-n30000-e16-lr{lr:g}-wd3.2-{adaptation}-s42',
            initialization='related', epochs=16, lr=lr, wd=3.2, adaptation=adaptation,
        ))
    return rows


def hyperball_finetune_configs(lrs=(3e-4, 3e-3, 0.03)):
    """P4(d). Fine-tune related DNA on the pretrained Frobenius sphere.

    This is not decay toward the pretrained values: the norm stays at the
    loaded checkpoint and the direction can still rotate. Weight decay is 0.
    The grid spans the one-epoch related AdamW learning rate and the
    from-scratch Hyperball learning rate. Does not lock a prediction.
    """
    return [
        _base(
            name=f'a6-ft-related-n30000-e16-hyperball-lr{lr:g}-s42',
            initialization='related',
            epochs=16,
            lr=lr,
            wd=0.0,
            optimizer='hyperball',
            adaptation='full',
        )
        for lr in lrs
    ]


def sample_scaling_configs(lr_by_init, counts=(1000, 3000, 10000)):
    """P4(b). Sixteen passes at pools smaller than the supplied 30k. Does not lock."""
    rows = []
    for init, lr in lr_by_init.items():
        initialization = 'random' if init == 'scratch' else init
        for ntrain in counts:
            rows.append(_base(
                name=f'a6-ft-{init}-n{ntrain}-e16-lr{lr:g}-wd3.2-s42',
                initialization=initialization, epochs=16, lr=lr, wd=3.2, ntrain=ntrain,
                adaptation='full',
            ))
    return rows


def dropout_replicas(rate, seeds=(43, 44, 45, 46)):
    """P2(b)(iii). Same data order as the seed-42 member; only the initialization changes."""
    return [
        _base(
            name=f'a6-dropout-p{rate:g}-s{seed}',
            attention_dropout=rate,
            seed=seed,
            data_seed=42,
            save_logits=True,
        )
        for seed in seeds
    ]


def _lock_all(notes):
    locked = []
    controlled = dict(
        ntrain=30000,
        context=1024,
        batch=64,
        seed=42,
        data_seed=42,
        schedule='linear',
        validation='splits/validation.npy',
        unique_tokens=TOKENS_PER_PASS,
    )
    for note in notes:
        row = lock_prediction(
            prediction_id=note['run_name'],
            problem=note['problem'],
            run_name=note['run_name'],
            baseline_nll=BASELINE_NLL,
            baseline_source=f'measured reference run {BASELINE_ID}',
            changed=note['changed'],
            controlled=controlled,
            direction=note['direction'],
            interval=note['interval'],
            interval_null_reason='',
            numeric_prediction=note['numeric_prediction'],
            diagnostics_expected=note['diagnostics_expected'],
            mechanism=note['mechanism'],
            confidence=note['confidence'],
            falsification=note['falsification'],
            run_identifier=note['run_name'],
        )
        locked.append(row['prediction_id'])
    return locked


def _append_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open('a') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def launch_wave1(*, max_parallel=4, gpu='H100'):
    from experiments.a6_generalization.training import run
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image, secrets

    configs, notes = wave1_configs()
    locked = _lock_all(notes)
    remote = app.function(
        image=build_image(),
        volumes=VOLUME_MOUNTS,
        secrets=secrets(include_wandb=True),
        gpu=gpu,
        cpu=4,
        memory=32768,
        timeout=24 * 3600,
        max_containers=max_parallel,
    )(run)
    launched = []
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
        for cfg in configs:
            call = remote.spawn(cfg)
            exposure = cfg.ntrain * cfg.context * cfg.epochs
            launched.append(dict(
                record_type='spawn',
                timestamp=datetime.now(timezone.utc).isoformat(),
                problem=next(note['problem'] for note in notes if note['run_name'] == cfg.name),
                experiment_id=cfg.name,
                config=asdict(cfg),
                architecture=cfg.architecture,
                initialization=cfg.initialization,
                lr=cfg.lr,
                wd=cfg.wd,
                schedule=cfg.schedule,
                optimizer=cfg.optimizer,
                batch=cfg.batch,
                unique_training_sequences=cfg.ntrain,
                passes=cfg.epochs,
                total_exposure_tokens=exposure,
                model_seed=cfg.seed,
                data_order_seed=cfg.data_seed,
                modal_call_id=call.object_id,
                wandb_id=None,
                checkpoint=f'a6/{cfg.name}/model.pt',
                status='spawned',
                final_validation_nll=None,
                diagnostics=f'a6/{cfg.name}/diagnostics.json',
                failure_reason=None,
                gpu=gpu,
                max_parallel=max_parallel,
            ))
            print('spawned', cfg.name, call.object_id, flush=True)
    _append_manifest(launched)
    print(f'locked {len(locked)} predictions and spawned {len(launched)} runs')
    return launched


def _endpoint(name):
    history = json.loads((Path('results/a6_generalization/runs') / name / 'history.json').read_text())
    return float(history[-1]['validation_nll'])


def wave2_configs():
    """Second seeds, the 4- and 16-epoch fine-tunes, dense depth 3, and SGD.

    Does not lock predictions. Scratch at 16 epochs and weight decay 3.2 is
    already in the supplied grid, so it is not trained again.
    """
    configs = []
    notes = []

    def add(cfg, **note):
        configs.append(cfg)
        notes.append(dict(run_name=cfg.name, numeric_prediction=sum(note['interval']) / 2, **note))

    moe = _endpoint('a6-moe-lr0.003-s42')
    loop = _endpoint('a6-loop-lr0.001-s42')
    hyperball = _endpoint('a6-hyperball-lr0.03-s42')
    sam = _endpoint('a6-sam-rho0.01-s42')
    one_epoch = {
        'related': _endpoint('a6-ft-related-n30000-e1-lr0.0003-wd3.2-s42'),
        'scratch': _endpoint('a6-ft-scratch-n30000-e1-lr0.003-wd3.2-s42'),
        'dclm': _endpoint('a6-ft-dclm-n30000-e1-lr0.003-wd3.2-s42'),
        'noise': _endpoint('a6-ft-noise-n30000-e1-lr0.003-wd3.2-s42'),
    }
    loops = {lr: _endpoint(f'a6-loop-lr{lr:g}-s42') for lr in (0.001, 0.003, 0.01)}

    add(
        _base(name='a6-sam-rho0.01-s43', optimizer='sam', rho=0.01, seed=43),
        problem='P3c-s43', interval=[1.256, 1.270], direction='similar',
        changed=dict(optimizer='sam', rho=0.01, seed=43),
        baseline_nll=sam, baseline_source='measured a6-sam-rho0.01-s42',
        mechanism='The seed-42 SAM run at rho 0.01 matched AdamW inside the dense seed span. A second initialization should stay there. Rho 0.1 was worse, so it is not the run being replicated.',
        diagnostics_expected='Sharpness stays well below the AdamW value 0.100. Validation NLL stays inside the dense seed span.',
        falsification='Falsify if this run moves the AdamW baseline 1.262714 by more than 0.02.',
        confidence='medium',
    )
    add(
        _base(name='a6-moe-lr0.003-s43', architecture='moe', lr=0.003, seed=43),
        problem='P1b-s43', interval=[1.260, 1.280], direction='similar',
        changed=dict(architecture='moe', lr=0.003, seed=43),
        baseline_nll=moe, baseline_source='measured a6-moe-lr0.003-s42',
        mechanism='Learning rate 0.003 was the best of the three MoE runs and finished 0.0057 worse than dense depth 8. The learning-rate 0.001 collapse was a different recipe. A second initialization at the winning learning rate should keep the small deficit.',
        diagnostics_expected='Router load CV stays below the 0.72 of the learning-rate 0.01 run. The endpoint does not repeat that run\'s late collapse.',
        falsification='Falsify if the second seed beats 1.258 or finishes above 1.295.',
        confidence='medium',
    )
    add(
        _base(name='a6-loop-lr0.001-s43', architecture='loop', lr=0.001, seed=43),
        problem='P1c-s43', interval=[1.260, 1.272], direction='similar',
        changed=dict(architecture='loop', lr=0.001, seed=43),
        baseline_nll=loop, baseline_source='measured a6-loop-lr0.001-s42',
        mechanism='The two best loop learning rates agreed to 0.00007 and matched dense depth 8 within seed noise. Seed 43 should do the same at the slightly better learning rate.',
        diagnostics_expected='Stored parameters stay 11,426,688 and executed depth stays 8.',
        falsification='Falsify if the second seed is below 1.255 or above 1.280.',
        confidence='medium',
    )
    add(
        _base(name='a6-hyperball-lr0.03-s43', optimizer='hyperball', lr=0.03, wd=0.0, seed=43),
        problem='P2a-ii-s43', interval=[1.265, 1.285], direction='similar',
        changed=dict(optimizer='hyperball', lr=0.03, wd=0.0, seed=43),
        baseline_nll=hyperball, baseline_source='measured a6-hyperball-lr0.03-s42',
        mechanism='Learning rate 0.03 was the only Hyperball run that did not walk away from an early good point. Its endpoint should replicate. The smaller learning rates are not the selected recipe.',
        diagnostics_expected='Linear-layer norms stay at initialization. The endpoint stays near 1.271 rather than rising toward 1.36.',
        falsification='Falsify if the second seed beats 1.258 or finishes above 1.320.',
        confidence='medium',
    )
    for rate in (0.05, 0.15, 0.25):
        first = _endpoint(f'a6-dropout-p{rate:g}-s42')
        add(
            _base(
                name=f'a6-dropout-p{rate:g}-s43',
                attention_dropout=rate,
                seed=43,
                data_seed=42,
                save_logits=True,
            ),
            problem='P2b-i-s43', interval=[first - 0.006, first + 0.006], direction='similar',
            changed=dict(attention_dropout=rate, seed=43, data_seed=42),
            baseline_nll=first, baseline_source=f'measured a6-dropout-p{rate:g}-s42',
            mechanism='All three seed-42 dropout rates landed inside the dense seed span. A second initialization at the same data order should stay there. The saved logits are for a later ensemble comparison.',
            diagnostics_expected='Validation NLL stays within 0.006 of the seed-42 twin.',
            falsification='Falsify if any second-seed dropout rate moves more than 0.015 from its seed-42 twin.',
            confidence='medium',
        )
    for lr, interval in ((0.001, [1.272, 1.330]), (0.003, [1.272, 1.330]), (0.01, [1.285, 1.360])):
        add(
            dense_depth3(lr),
            problem='P1d', interval=interval, direction='higher',
            changed=dict(architecture='dense', depth=3, width=512, lr=lr),
            baseline_nll=loops[lr], baseline_source=f'measured a6-loop-lr{lr:g}-s42',
            mechanism='Dense depth 3 stores the same 11,426,688 parameters as the six-pass loop and executes three blocks instead of eight. The loop already matched dense depth 8. Removing the extra applications should not improve the loss, and the lost compute should make depth 3 worse.',
            diagnostics_expected='Stored parameters equal the loop. Executed depth is 3, not 8.',
            falsification='Falsify if the best depth-3 run beats 1.262, or if the best is above 1.40.',
            confidence='medium',
        )
    for lr, interval in ((0.001, [1.300, 1.500]), (0.003, [1.280, 1.450]), (0.01, [1.300, 1.550])):
        add(
            _base(name=f'a6-sgd-lr{lr:g}-wd3.2-s42', optimizer='sgd', lr=lr, wd=3.2),
            problem='P3d', interval=interval, direction='higher',
            changed=dict(optimizer='sgd', lr=lr, wd=3.2),
            baseline_nll=BASELINE_NLL, baseline_source=f'measured reference run {BASELINE_ID}',
            mechanism='SGD applies weight decay inside the gradient, on linear weights only, and it has no adaptive second moment. At these learning rates AdamW already works. SGD should be worse. A win would require the coupled decay to happen to suit this loss.',
            diagnostics_expected='The run finishes. The best learning rate is still above the AdamW baseline by more than seed noise.',
            falsification='Falsify if the best SGD run beats 1.266, or if the best is above 1.70.',
            confidence='medium',
        )

    longer = {
        4: {
            'related': (1e-4, 3e-4, 1e-3),
            'scratch': (1e-3, 3e-3, 1e-2),
            'dclm': (1e-3, 3e-3, 1e-2),
            'noise': (1e-3, 3e-3, 1e-2),
        },
        16: {
            'related': (1e-4, 3e-4, 1e-3),
            'dclm': (1e-3, 3e-3, 1e-2),
            'noise': (1e-3, 3e-3, 1e-2),
        },
    }
    intervals = {
        ('related', 4, 1e-4): [1.200, 1.245],
        ('related', 4, 3e-4): [1.180, 1.235],
        ('related', 4, 1e-3): [1.185, 1.245],
        ('scratch', 4, 1e-3): [1.270, 1.310],
        ('scratch', 4, 3e-3): [1.265, 1.310],
        ('scratch', 4, 1e-2): [1.270, 1.400],
        ('dclm', 4, 1e-3): [1.265, 1.310],
        ('dclm', 4, 3e-3): [1.260, 1.310],
        ('dclm', 4, 1e-2): [1.270, 1.400],
        ('noise', 4, 1e-3): [1.280, 1.340],
        ('noise', 4, 3e-3): [1.275, 1.340],
        ('noise', 4, 1e-2): [1.280, 1.420],
        ('related', 16, 1e-4): [1.180, 1.240],
        ('related', 16, 3e-4): [1.160, 1.235],
        ('related', 16, 1e-3): [1.170, 1.240],
        ('dclm', 16, 1e-3): [1.250, 1.300],
        ('dclm', 16, 3e-3): [1.250, 1.300],
        ('dclm', 16, 1e-2): [1.260, 1.360],
        ('noise', 16, 1e-3): [1.270, 1.330],
        ('noise', 16, 3e-3): [1.270, 1.330],
        ('noise', 16, 1e-2): [1.280, 1.400],
    }
    for epochs, grid in longer.items():
        for init, lrs in grid.items():
            for lr in lrs:
                name = f'a6-ft-{init}-n30000-e{epochs}-lr{lr:g}-wd3.2-s42'
                initialization = 'random' if init == 'scratch' else init
                add(
                    _base(
                        name=name, initialization=initialization, epochs=epochs, lr=lr, wd=3.2,
                        adaptation='full',
                    ),
                    problem=f'P4a-e{epochs}', interval=intervals[(init, epochs, lr)],
                    direction='lower' if init == 'related' else 'higher',
                    changed=dict(initialization=initialization, epochs=epochs, lr=lr, wd=3.2),
                    baseline_nll=one_epoch[init],
                    baseline_source=f'measured 1-epoch {init} at its best learning rate',
                    mechanism=(
                        'Related DNA after one pass is already 0.069 better than scratch. '
                        'Four and sixteen passes should keep that lead and can improve the related model further. '
                        'DCLM and noise were best at the top of the one-epoch grid, so the grid adds 0.01. '
                        'Scratch at sixteen epochs and weight decay 3.2 is already in the supplied grid and is not retrained.'
                    ),
                    diagnostics_expected='The best related run stays below the best scratch run at the same horizon.',
                    falsification=(
                        'At 4 epochs, falsify if the best related run is above 1.26 or worse than the best scratch run. '
                        'At 16 epochs, falsify if the best related run is above 1.270 or the best DCLM run beats it.'
                    ),
                    confidence='medium' if init == 'related' else 'low',
                )
    data_order = {
        'related': 3e-4,
        'scratch': 3e-3,
        'dclm': 3e-3,
        'noise': 3e-3,
    }
    data_intervals = {
        'related': [1.220, 1.242],
        'scratch': [1.288, 1.312],
        'dclm': [1.284, 1.308],
        'noise': [1.310, 1.340],
    }
    for init, lr in data_order.items():
        name = f'a6-ft-{init}-n30000-e1-lr{lr:g}-wd3.2-s42-d43'
        initialization = 'random' if init == 'scratch' else init
        add(
            _base(
                name=name, initialization=initialization, epochs=1, lr=lr, wd=3.2,
                adaptation='full', data_seed=43,
            ),
            problem='P4a-d43', interval=data_intervals[init], direction='similar',
            changed=dict(initialization=initialization, epochs=1, lr=lr, wd=3.2, data_seed=43),
            baseline_nll=one_epoch[init],
            baseline_source=f'measured 1-epoch {init} at data order 42',
            mechanism='The pretrained weights are fixed. A second data order is the replication, not a new pretraining seed. The related-versus-scratch gap is 0.069, much larger than a reshuffle of the same 30k chunks.',
            diagnostics_expected='Related stays near 1.229 and well below scratch.',
            falsification='Falsify if the related run at data order 43 exceeds 1.26 or exceeds the scratch run.',
            confidence='high' if init == 'related' else 'medium',
        )
    if len({cfg.name for cfg in configs}) != len(configs):
        raise RuntimeError('Wave-2 names are not unique.')
    wave1, _ = wave1_configs()
    overlap = {cfg.name for cfg in configs} & {cfg.name for cfg in wave1}
    if overlap:
        raise RuntimeError(f'Wave 2 collides with wave 1: {sorted(overlap)}')
    return configs, notes


def wave3_configs():
    """P4(b), P4(c), and P4(d), after the 16-epoch learning rates are known.

    The winning 16-epoch learning rate is 0.001 for related DNA, DCLM, and noise.
    The supplied scratch grid at weight decay 3.2 is also best at 0.001. Full
    fine-tuning of related DNA at that learning rate and weight decay 3.2 already
    exists and is not retrained.
    """
    configs = []
    notes = []
    full_ft = 1.2647764340471477

    def add(cfg, **note):
        note.setdefault('numeric_prediction', sum(note['interval']) / 2)
        note.setdefault('run_name', cfg.name)
        configs.append(cfg)
        notes.append(note)

    for ntrain, interval in ((1000, [1.20, 1.40]), (3000, [1.20, 1.35]), (10000, [1.22, 1.32])):
        add(
            _base(
                name=f'a6-ft-related-n{ntrain}-e16-lr0.001-wd3.2-s42',
                initialization='related', epochs=16, lr=0.001, wd=3.2, ntrain=ntrain,
                adaptation='full',
            ),
            problem='P4b', interval=interval, direction='higher',
            changed=dict(initialization='related', epochs=16, lr=0.001, wd=3.2, ntrain=ntrain),
            baseline_nll=full_ft, baseline_source='measured related 16-epoch LR 0.001 WD 3.2',
            mechanism=(
                'Sixteen passes at learning rate 0.001 kept related DNA near 1.265 on 30k chunks. '
                'A smaller pool has fewer unique sequences, so the loss should be higher, and 10k should beat 1k.'
            ),
            diagnostics_expected='Loss falls as the pool grows from 1k to 10k.',
            falsification='Falsify if every initialization is worse at 10k than at 1k, or if related DNA at 10k is above 1.45.',
            confidence='medium',
        )
    for init, intervals in (
        ('scratch', {1000: [1.30, 1.60], 3000: [1.28, 1.50], 10000: [1.26, 1.40]}),
        ('dclm', {1000: [1.28, 1.55], 3000: [1.26, 1.45], 10000: [1.25, 1.38]}),
        ('noise', {1000: [1.30, 1.60], 3000: [1.28, 1.50], 10000: [1.26, 1.42]}),
    ):
        initialization = 'random' if init == 'scratch' else init
        for ntrain, interval in intervals.items():
            add(
                _base(
                    name=f'a6-ft-{init}-n{ntrain}-e16-lr0.001-wd3.2-s42',
                    initialization=initialization, epochs=16, lr=0.001, wd=3.2, ntrain=ntrain,
                    adaptation='full',
                ),
                problem='P4b', interval=interval, direction='higher',
                changed=dict(initialization=initialization, epochs=16, lr=0.001, wd=3.2, ntrain=ntrain),
                baseline_nll=full_ft, baseline_source='measured related 16-epoch LR 0.001 WD 3.2',
                mechanism=(
                    'At 30k and 16 passes the four initializations landed within about 0.004. '
                    'On a smaller pool the pretrained related model should keep more of its lead over scratch.'
                ),
                diagnostics_expected='Scratch stays worse than related DNA at the same pool size.',
                falsification='Falsify if every initialization is worse at 10k than at 1k, or if related DNA at 10k is above 1.45.',
                confidence='low',
            )
    for wd, interval in ((0.0, [1.25, 1.45]), (0.1, [1.25, 1.40]), (1.0, [1.24, 1.32])):
        add(
            _base(
                name=f'a6-ft-related-n30000-e16-lr0.001-wd{wd:g}-s42',
                initialization='related', epochs=16, lr=0.001, wd=wd, adaptation='full',
            ),
            problem='P4c', interval=interval, direction='similar',
            changed=dict(initialization='related', epochs=16, lr=0.001, wd=wd, adaptation='full'),
            baseline_nll=full_ft, baseline_source='measured related 16-epoch LR 0.001 WD 3.2',
            mechanism=(
                'Weight decay 3.2 pulled the low-learning-rate related runs from about 1.23 up past 1.30. '
                'Less decay should remove less of the pretrained solution. The full weight-decay 3.2 run is the baseline and is not retrained.'
            ),
            diagnostics_expected='A lower weight decay does not repeat the 1.32 climb of learning rate 1e-4.',
            falsification='Falsify if freezing the body beats 1.254, or if every adaptation finishes above 1.40.',
            confidence='medium',
        )
    for adaptation, interval in (
        ('toward_initialization', [1.22, 1.30]),
        ('frozen_head', [1.24, 1.32]),
        ('frozen_body', [1.28, 1.45]),
    ):
        add(
            _base(
                name=f'a6-ft-related-n30000-e16-lr0.001-wd3.2-{adaptation}-s42',
                initialization='related', epochs=16, lr=0.001, wd=3.2, adaptation=adaptation,
            ),
            problem='P4c', interval=interval,
            direction='lower' if adaptation == 'toward_initialization' else 'higher',
            changed=dict(initialization='related', epochs=16, lr=0.001, wd=3.2, adaptation=adaptation),
            baseline_nll=full_ft, baseline_source='measured related 16-epoch LR 0.001 WD 3.2',
            mechanism=(
                'Decay toward the pretrained weights keeps the solution that one pass already improved. '
                'Freezing the body leaves only the embedding and the output head, which is where DCLM had to recover.'
            ),
            diagnostics_expected='The frozen body is worse than full fine-tuning. Decay toward initialization is not worse than decay toward zero.',
            falsification='Falsify if freezing the body beats 1.254, or if every adaptation finishes above 1.40.',
            confidence='medium',
        )
    for lr, interval in ((3e-4, [1.22, 1.32]), (3e-3, [1.23, 1.34]), (0.03, [1.25, 1.40])):
        add(
            _base(
                name=f'a6-ft-related-n30000-e16-hyperball-lr{lr:g}-s42',
                initialization='related', epochs=16, lr=lr, wd=0.0,
                optimizer='hyperball', adaptation='full',
            ),
            problem='P4d', interval=interval, direction='similar',
            changed=dict(initialization='related', epochs=16, lr=lr, wd=0.0, optimizer='hyperball'),
            baseline_nll=full_ft, baseline_source='measured related 16-epoch LR 0.001 WD 3.2',
            mechanism=(
                'Hyperball freezes each linear-weight norm at the loaded checkpoint, so weight decay cannot shrink the pretrained solution. '
                'The direction can still rotate. From-scratch Hyperball needed learning rate 0.03; the one-pass related run preferred 3e-4.'
            ),
            diagnostics_expected='Linear norms match the loaded checkpoint. At least one learning rate stays near the full fine-tuning loss.',
            falsification='Falsify if the best Hyperball fine-tune is above 1.35 or below 1.20.',
            confidence='medium',
        )
    names = [cfg.name for cfg in configs]
    if len(names) != len(set(names)):
        raise RuntimeError('Wave-3 names are not unique.')
    wave1, _ = wave1_configs()
    wave2, _ = wave2_configs()
    taken = {cfg.name for cfg in wave1} | {cfg.name for cfg in wave2}
    overlap = set(names) & taken
    if overlap:
        raise RuntimeError(f'Wave 3 collides with an earlier wave: {sorted(overlap)}')
    return configs, notes


def wave4_configs():
    """Second data order for the selected 4-pass and 16-pass recipes.

    The learning rate stays the one selected on data order 42. The model seed
    stays 42, because the pretrained weights are fixed. Scratch at 16 epochs
    uses the supplied run as its twin and is trained here only at data order 43.
    """
    selected = (
        ('related', 4, 1e-4, 1.228766800943762, [1.217, 1.241], 'medium'),
        ('scratch', 4, 3e-3, 1.2765886756777762, [1.265, 1.289], 'medium'),
        ('dclm', 4, 1e-3, 1.2733154431916773, [1.261, 1.285], 'medium'),
        ('noise', 4, 1e-3, 1.2809555028527975, [1.263, 1.299], 'low'),
        ('related', 16, 1e-3, 1.2647764340471477, [1.253, 1.277], 'medium'),
        ('scratch', 16, 1e-3, 1.2636805066037924, [1.252, 1.276], 'medium'),
        ('dclm', 16, 1e-3, 1.2612853994648903, [1.249, 1.273], 'medium'),
        ('noise', 16, 1e-3, 1.2631211679652332, [1.245, 1.281], 'low'),
    )
    configs, notes = [], []
    for init, epochs, lr, twin, interval, confidence in selected:
        initialization = 'random' if init == 'scratch' else init
        name = f'a6-ft-{init}-n30000-e{epochs}-lr{lr:g}-wd3.2-s42-d43'
        problem = f'P4a-e{epochs}-d43'
        if epochs == 4:
            falsification = (
                'Falsify if related DNA at data order 43 is above 1.26, finishes worse than scratch, '
                'or any run moves more than 0.02 from its data-order-42 twin.'
            )
        else:
            falsification = (
                'Falsify if related DNA at data order 43 finishes below 1.24, '
                'or any run moves more than 0.02 from its data-order-42 twin. '
                'The 0.004 ranking among the four data-order-42 endpoints is not part of this prediction.'
            )
        configs.append(_base(
            name=name, initialization=initialization, epochs=epochs, lr=lr, wd=3.2,
            adaptation='full', data_seed=43,
        ))
        notes.append(dict(
            problem=problem, run_name=name, interval=interval, direction='similar',
            numeric_prediction=twin,
            changed=dict(initialization=initialization, epochs=epochs, lr=lr, wd=3.2, data_seed=43),
            baseline_nll=twin,
            baseline_source=f'measured {epochs}-epoch {init} at data order 42, learning rate {lr:g}',
            mechanism=(
                'The learning rate was selected on data order 42. This run repeats that recipe on data order 43. '
                'The pretrained weights stay fixed. At one pass, related DNA and scratch moved by less than 0.001 '
                'and noise moved by 0.014. Four and sixteen passes should stay near those twins.'
            ),
            diagnostics_expected='The endpoint stays within 0.02 of the data-order-42 twin.',
            falsification=falsification,
            confidence=confidence,
        ))
    names = [cfg.name for cfg in configs]
    if len(names) != len(set(names)):
        raise RuntimeError('Wave-4 names are not unique.')
    taken = {cfg.name for cfg in wave1_configs()[0]}
    taken |= {cfg.name for cfg in wave2_configs()[0]}
    taken |= {cfg.name for cfg in wave3_configs()[0]}
    overlap = set(names) & taken
    if overlap:
        raise RuntimeError(f'Wave 4 collides with an earlier wave: {sorted(overlap)}')
    return configs, notes


def _launch(loader, *, max_parallel=2, gpu='H100'):
    from experiments.a6_generalization.training import run
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image, secrets

    configs, notes = loader()
    locked = []
    for cfg, note in zip(configs, notes):
        controlled = dict(
            ntrain=cfg.ntrain,
            context=1024,
            batch=64,
            seed=cfg.seed,
            data_seed=cfg.data_seed,
            schedule='linear',
            validation='splits/validation.npy',
            unique_tokens=cfg.ntrain * cfg.context,
            epochs=cfg.epochs,
        )
        lock_prediction(
            prediction_id=note['run_name'],
            problem=note['problem'],
            run_name=note['run_name'],
            baseline_nll=note['baseline_nll'],
            baseline_source=note['baseline_source'],
            changed=note['changed'],
            controlled=controlled,
            direction=note['direction'],
            interval=note['interval'],
            interval_null_reason='',
            numeric_prediction=note['numeric_prediction'],
            diagnostics_expected=note['diagnostics_expected'],
            mechanism=note['mechanism'],
            confidence=note['confidence'],
            falsification=note['falsification'],
            run_identifier=note['run_name'],
        )
        locked.append(note['run_name'])
    remote = app.function(
        image=build_image(),
        volumes=VOLUME_MOUNTS,
        secrets=secrets(include_wandb=True),
        gpu=gpu,
        cpu=4,
        memory=32768,
        timeout=24 * 3600,
        max_containers=max_parallel,
    )(run)
    launched = []
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
        for cfg in configs:
            call = remote.spawn(cfg)
            exposure = cfg.ntrain * cfg.context * cfg.epochs
            launched.append(dict(
                record_type='spawn',
                timestamp=datetime.now(timezone.utc).isoformat(),
                problem=next(note['problem'] for note in notes if note['run_name'] == cfg.name),
                experiment_id=cfg.name,
                config=asdict(cfg),
                architecture=cfg.architecture,
                initialization=cfg.initialization,
                lr=cfg.lr,
                wd=cfg.wd,
                schedule=cfg.schedule,
                optimizer=cfg.optimizer,
                batch=cfg.batch,
                unique_training_sequences=cfg.ntrain,
                passes=cfg.epochs,
                total_exposure_tokens=exposure,
                model_seed=cfg.seed,
                data_order_seed=cfg.data_seed,
                modal_call_id=call.object_id,
                wandb_id=None,
                checkpoint=f'a6/{cfg.name}/model.pt',
                status='spawned',
                final_validation_nll=None,
                diagnostics=f'a6/{cfg.name}/diagnostics.json',
                failure_reason=None,
                gpu=gpu,
                max_parallel=max_parallel,
            ))
            print('spawned', cfg.name, call.object_id, flush=True)
    _append_manifest(launched)
    print(f'locked {len(locked)} predictions and spawned {len(launched)} runs')
    return launched


def launch_wave2(*, max_parallel=2, gpu='H100'):
    from experiments.a6_generalization.training import run
    import modal
    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image, secrets

    configs, notes = wave2_configs()
    locked = []
    for cfg, note in zip(configs, notes):
        controlled = dict(
            ntrain=30000,
            context=1024,
            batch=64,
            seed=cfg.seed,
            data_seed=cfg.data_seed,
            schedule='linear',
            validation='splits/validation.npy',
            unique_tokens=cfg.ntrain * cfg.context,
            epochs=cfg.epochs,
        )
        lock_prediction(
            prediction_id=note['run_name'],
            problem=note['problem'],
            run_name=note['run_name'],
            baseline_nll=note['baseline_nll'],
            baseline_source=note['baseline_source'],
            changed=note['changed'],
            controlled=controlled,
            direction=note['direction'],
            interval=note['interval'],
            interval_null_reason='',
            numeric_prediction=note['numeric_prediction'],
            diagnostics_expected=note['diagnostics_expected'],
            mechanism=note['mechanism'],
            confidence=note['confidence'],
            falsification=note['falsification'],
            run_identifier=note['run_name'],
        )
        locked.append(note['run_name'])
    remote = app.function(
        image=build_image(),
        volumes=VOLUME_MOUNTS,
        secrets=secrets(include_wandb=True),
        gpu=gpu,
        cpu=4,
        memory=32768,
        timeout=24 * 3600,
        max_containers=max_parallel,
    )(run)
    launched = []
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT, detach=True):
        for cfg in configs:
            call = remote.spawn(cfg)
            exposure = cfg.ntrain * cfg.context * cfg.epochs
            launched.append(dict(
                record_type='spawn',
                timestamp=datetime.now(timezone.utc).isoformat(),
                problem=next(note['problem'] for note in notes if note['run_name'] == cfg.name),
                experiment_id=cfg.name,
                config=asdict(cfg),
                architecture=cfg.architecture,
                initialization=cfg.initialization,
                lr=cfg.lr,
                wd=cfg.wd,
                schedule=cfg.schedule,
                optimizer=cfg.optimizer,
                batch=cfg.batch,
                unique_training_sequences=cfg.ntrain,
                passes=cfg.epochs,
                total_exposure_tokens=exposure,
                model_seed=cfg.seed,
                data_order_seed=cfg.data_seed,
                modal_call_id=call.object_id,
                wandb_id=None,
                checkpoint=f'a6/{cfg.name}/model.pt',
                status='spawned',
                final_validation_nll=None,
                diagnostics=f'a6/{cfg.name}/diagnostics.json',
                failure_reason=None,
                gpu=gpu,
                max_parallel=max_parallel,
            ))
            print('spawned', cfg.name, call.object_id, flush=True)
    _append_manifest(launched)
    print(f'locked {len(locked)} predictions and spawned {len(launched)} runs')
    return launched


def launch_wave3(*, max_parallel=2, gpu='H100'):
    return _launch(wave3_configs, max_parallel=max_parallel, gpu=gpu)


def launch_wave4(*, max_parallel=2, gpu='H100'):
    return _launch(wave4_configs, max_parallel=max_parallel, gpu=gpu)


if __name__ == '__main__':
    import sys

    if sys.argv[1:] == ['wave2']:
        launch_wave2()
    elif sys.argv[1:] == ['wave3']:
        launch_wave3()
    elif sys.argv[1:] == ['wave4']:
        launch_wave4()
    else:
        raise SystemExit('Refusing to relaunch. Use: python -m experiments.a6_generalization.orchestrate wave4')
