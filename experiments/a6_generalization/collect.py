"""Pull finished A6 runs from the Modal volume and score locked predictions.

history.json is rewritten during training, and a Modal volume can show that
partial file before model.pt exists. A run counts as finished only when its
last logged epoch has reached the locked target. A missing directory means
the run is still queued or running, not that it failed.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import subprocess

from experiments.a6_generalization.ledger import audit, read_rows, record_outcome

VOLUME = 'volume-dl_alchemy'
ENV = 'cs312-daxton'
LOCAL = Path('results/a6_generalization/runs')
MANIFEST = Path('results/a6_generalization/run_manifest.jsonl')


def _modal(*args):
    return subprocess.run(
        ['modal', 'volume', *args, '--env', ENV],
        check=True,
        text=True,
        capture_output=True,
    )


def finished_names():
    listing = _modal('ls', VOLUME, 'ckpts/a6')
    names = []
    for line in listing.stdout.splitlines():
        line = line.strip().rstrip('/')
        if line.startswith('ckpts/a6/'):
            names.append(line.split('/')[-1])
    return names


def fetch(name):
    destination = LOCAL / name
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ('history.json', 'diagnostics.json', 'initial.json', 'config.json'):
        remote = f'ckpts/a6/{name}/{filename}'
        try:
            _modal('get', VOLUME, remote, str(destination / filename), '--force')
        except subprocess.CalledProcessError:
            continue
    return destination


def _append_status(name, status, nll, reason=None):
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    previous = [row for row in rows if row.get('experiment_id') == name]
    if not previous:
        raise KeyError(name)
    row = dict(previous[-1])
    row['record_type'] = 'status'
    row['status'] = status
    row['final_validation_nll'] = nll
    row['failure_reason'] = reason
    row['timestamp'] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open('a') as handle:
        handle.write(json.dumps(row) + '\n')


GROUPS = {
    'P1b': ('a6-moe-lr0.001-s42', 'a6-moe-lr0.003-s42', 'a6-moe-lr0.01-s42'),
    'P1c': ('a6-loop-lr0.001-s42', 'a6-loop-lr0.003-s42', 'a6-loop-lr0.01-s42'),
    'P2a-ii': ('a6-hyperball-lr0.003-s42', 'a6-hyperball-lr0.01-s42', 'a6-hyperball-lr0.03-s42'),
    'P2b-i': ('a6-dropout-p0.05-s42', 'a6-dropout-p0.15-s42', 'a6-dropout-p0.25-s42'),
    'P3c': ('a6-sam-rho0.01-s42', 'a6-sam-rho0.05-s42', 'a6-sam-rho0.1-s42'),
    'P1b-s43': ('a6-moe-lr0.003-s43',),
    'P1c-s43': ('a6-loop-lr0.001-s43',),
    'P2a-ii-s43': ('a6-hyperball-lr0.03-s43',),
    'P2b-i-s43': ('a6-dropout-p0.05-s43', 'a6-dropout-p0.15-s43', 'a6-dropout-p0.25-s43'),
    'P3c-s43': ('a6-sam-rho0.01-s43',),
    'P1d': ('a6-dense-d3-lr0.001-s42', 'a6-dense-d3-lr0.003-s42', 'a6-dense-d3-lr0.01-s42'),
    'P3d': ('a6-sgd-lr0.001-wd3.2-s42', 'a6-sgd-lr0.003-wd3.2-s42', 'a6-sgd-lr0.01-wd3.2-s42'),
}


def comparison_groups(prediction_ids):
    """Every locked run belongs to exactly one family, including the fine-tuning grids.

    A name can contain ``-e16-`` and still be a different experiment: a smaller
    pool, a weight-decay ablation, a freeze, or Hyperball. Those must not be
    scored with the full-fine-tuning family.
    """
    groups = {key: tuple(names) for key, names in GROUPS.items()}
    buckets = {key: [] for key in (
        'P4a', 'P4a-d43', 'P4a-e4', 'P4a-e4-d43', 'P4a-e16', 'P4a-e16-d43', 'P4b', 'P4c', 'P4d',
    )}
    for name in prediction_ids:
        name = str(name)
        if not name.startswith('a6-ft-'):
            continue
        if '-hyperball-' in name:
            buckets['P4d'].append(name)
        elif '-d43' in name and '-e4-' in name:
            buckets['P4a-e4-d43'].append(name)
        elif '-d43' in name and '-e16-' in name:
            buckets['P4a-e16-d43'].append(name)
        elif '-d43' in name:
            buckets['P4a-d43'].append(name)
        elif any(tag in name for tag in ('toward_initialization', 'frozen_head', 'frozen_body')):
            buckets['P4c'].append(name)
        elif '-e16-' in name and '-n30000-' not in name:
            buckets['P4b'].append(name)
        elif '-e16-' in name and '-wd3.2-' not in name:
            buckets['P4c'].append(name)
        elif '-e4-' in name:
            buckets['P4a-e4'].append(name)
        elif '-e16-' in name:
            buckets['P4a-e16'].append(name)
        elif '-e1-' in name:
            buckets['P4a'].append(name)
        else:
            raise ValueError(f'Fine-tuning run {name} has no scoring family.')
    for key, members in buckets.items():
        if members:
            groups[key] = tuple(members)
    return groups


def ready_to_score(history, target_epochs, diagnostics_present):
    """The outcome text is written once, so wait until diagnostics exist.

    history.json is visible during training. diagnostics.json is written
    after the last evaluation, in the same final commit as the checkpoint.
    """
    if not history or not diagnostics_present:
        return False
    return float(history[-1]['epochs']) >= float(target_epochs) - 0.05


def _measured(name, target_epochs):
    path = LOCAL / name / 'history.json'
    if not path.is_file():
        return None
    history = json.loads(path.read_text())
    if not ready_to_score(history, target_epochs, (LOCAL / name / 'diagnostics.json').is_file()):
        return None
    return float(history[-1]['validation_nll'])


_E4_TWINS = {
    'related': 1.228766800943762,
    'scratch': 1.2765886756777762,
    'dclm': 1.2733154431916773,
    'noise': 1.2809555028527975,
}
_E16_TWINS = {
    'related': 1.2647764340471477,
    'scratch': 1.2636805066037924,
    'dclm': 1.2612853994648903,
    'noise': 1.2631211679652332,
}


def _moved(measured, twins, limit):
    return any(
        abs(value - twins[init]) > limit
        for name, value in measured.items()
        for init in twins
        if f'-{init}-' in name
    )


def _family_falsified(problem, measured):
    best = min(measured.values())
    if problem == 'P1b':
        return best < 1.258 or any(value > 1.45 for value in measured.values())
    if problem == 'P1c':
        return best < 1.255 or best > 1.300
    if problem == 'P2a-ii':
        return best > 1.350 or best < 1.255
    if problem == 'P2b-i':
        return measured['a6-dropout-p0.05-s42'] < 1.258 or measured['a6-dropout-p0.25-s42'] < 1.2627138263769448
    if problem == 'P3c':
        return any(abs(value - 1.2627138263769448) > 0.025 for value in measured.values())
    if problem == 'P4a':
        related = min(value for name, value in measured.items() if '-related-' in name)
        dclm = min(value for name, value in measured.items() if '-dclm-' in name)
        return related > 1.40 or dclm < related
    if problem == 'P1b-s43':
        return best < 1.258 or best > 1.295
    if problem == 'P1c-s43':
        return best < 1.255 or best > 1.280
    if problem == 'P2a-ii-s43':
        return best < 1.258 or best > 1.320
    if problem == 'P2b-i-s43':
        twins = {
            'a6-dropout-p0.05-s43': _endpoint('a6-dropout-p0.05-s42'),
            'a6-dropout-p0.15-s43': _endpoint('a6-dropout-p0.15-s42'),
            'a6-dropout-p0.25-s43': _endpoint('a6-dropout-p0.25-s42'),
        }
        return any(abs(measured[name] - twin) > 0.015 for name, twin in twins.items())
    if problem == 'P3c-s43':
        return abs(best - 1.2627138263769448) > 0.02
    if problem == 'P4a-d43':
        related = min(value for name, value in measured.items() if '-related-' in name)
        scratch = min(value for name, value in measured.items() if '-scratch-' in name)
        return related > scratch or related > 1.26
    if problem == 'P4a-e4':
        related = min(value for name, value in measured.items() if '-related-' in name)
        scratch = min(value for name, value in measured.items() if '-scratch-' in name)
        return related > 1.26 or related > scratch
    if problem == 'P4a-e16':
        related = min(value for name, value in measured.items() if '-related-' in name)
        dclm = min(value for name, value in measured.items() if '-dclm-' in name)
        return related > 1.270 or dclm < related
    if problem == 'P4a-e4-d43':
        related = min(value for name, value in measured.items() if '-related-' in name)
        scratch = min(value for name, value in measured.items() if '-scratch-' in name)
        return related > 1.26 or related > scratch or _moved(measured, _E4_TWINS, 0.02)
    if problem == 'P4a-e16-d43':
        related = min(value for name, value in measured.items() if '-related-' in name)
        return related < 1.24 or _moved(measured, _E16_TWINS, 0.02)
    if problem == 'P1d':
        return best < 1.262 or best > 1.40
    if problem == 'P3d':
        return best < 1.266 or best > 1.70
    if problem == 'P4b':
        def at(init, ntrain):
            values = [
                value for name, value in measured.items()
                if f'-{init}-' in name and f'-n{ntrain}-' in name
            ]
            return min(values)
        wrong_way = all(
            at(init, 10000) > at(init, 1000)
            for init in ('related', 'scratch', 'dclm', 'noise')
        )
        return wrong_way or at('related', 10000) > 1.45
    if problem == 'P4c':
        frozen_body = min(value for name, value in measured.items() if 'frozen_body' in name)
        return frozen_body < 1.254 or all(value > 1.40 for value in measured.values())
    if problem == 'P4d':
        return best > 1.35 or best < 1.20
    raise KeyError(problem)


def _endpoint(name):
    history = json.loads((LOCAL / name / 'history.json').read_text())
    return float(history[-1]['validation_nll'])


def linear_lr_fraction(step_completed, total, warmup):
    """Fraction of peak LR used on the step that completed at step_completed.

    The loop sets the multiplier from the zero-based index, then records step+1.
    """
    index = int(step_completed) - 1
    if warmup and index < warmup:
        return index / warmup
    return max(0.0, (total - index) / (total - warmup))


def trajectory_text(history):
    """Describe whether the logged curve was still improving at the end."""
    if not history:
        return ''
    final = history[-1]
    best = min(history, key=lambda row: row['validation_nll'])
    mid = min(history, key=lambda row: abs(float(row['epochs']) - float(final['epochs']) / 2))
    sentence = (
        f"Epoch {float(mid['epochs']):g} was {float(mid['validation_nll']):.4f}."
    )
    if best is not final and float(best['validation_nll']) + 1e-4 < float(final['validation_nll']):
        sentence += (
            f" The minimum was {float(best['validation_nll']):.4f} at epoch {float(best['epochs']):g},"
            ' so the curve rose after that.'
        )
    else:
        sentence += ' The final logged point is the best one.'
    return sentence


def diagnostic_text(report):
    parts = []
    sharp = (report.get('sharpness') or {}).get('max_loss_increase')
    if sharp is not None:
        parts.append(f'Directional sharpness at radius 0.05 is {sharp:.4f}.')
    routing = report.get('routing') or {}
    if 'load_cv' in routing:
        parts.append(
            'Router load CV {load_cv:.3f}, mean entropy {entropy:.3f}, mean max probability {mean_max_probability:.3f}.'.format(
                **routing
            )
        )
    if 'stored_parameters' in report:
        parts.append(f"Stored parameters {int(report['stored_parameters']):,}.")
    if 'active_parameters' in report:
        parts.append(
            f"Active parameters {int(report['active_parameters']):,}, {int(report['experts'])} experts, top-{int(report['top_k'])}."
        )
    if 'executed_depth' in report:
        parts.append(
            f"Executed depth {int(report['executed_depth'])} with {int(report['passes'])} core passes."
        )
    return ' '.join(parts)


def _interpretation(problem, name, value, prediction, measured, falsified):
    baseline = float(prediction['baseline_nll'])
    best_name = min(measured, key=measured.get)
    best = measured[best_name]
    parts = [
        f'Final validation NLL {value:.6f}, {value - baseline:+.6f} versus the locked baseline {baseline:.6f}.',
        f'Best in {problem} is {best_name} at {best:.6f}.',
        'The falsified flag is the family criterion and is copied onto every member.',
    ]
    history_path = LOCAL / name / 'history.json'
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        parts.append(trajectory_text(history))
        if history and 'step' in history[-1] and prediction.get('changed', {}).get('schedule', 'linear') != 'constant':
            total = int(history[-1]['step'])
            warmup = min(math.ceil(4800 / 64), total - 1)
            best = min(history, key=lambda row: row['validation_nll'])
            parts.append(
                f"At that best point the linear schedule was at {linear_lr_fraction(best['step'], total, warmup):.0%} of peak LR."
            )
    diagnostics = LOCAL / name / 'diagnostics.json'
    if diagnostics.is_file():
        text = diagnostic_text(json.loads(diagnostics.read_text()))
        if text:
            parts.append(text)
    if falsified:
        parts.append(f"Falsification criterion: {prediction['falsification']}")
    else:
        parts.append('The family falsification criterion did not fire.')
    return ' '.join(parts)


def pending_downloads(names, predictions, outcomes):
    """Skip runs whose outcome is already in the ledger.

    The group loop does not read a family that already has an outcome, so
    those files do not need another download. A run that is still training
    is always fetched, because its history file is rewritten in place.
    """
    return tuple(
        name for name in names
        if name in predictions and name not in outcomes
    )


def score_finished():
    """Score a prediction only once its whole comparison group has finished.

    Family falsifiers talk about the best run. Writing an outcome early would
    freeze the wrong boolean, because the ledger does not allow a second outcome.
    """
    outcomes = {
        row['prediction_id']
        for row in read_rows()
        if row.get('record_type') == 'outcome'
    }
    predictions = {
        row['prediction_id']: row
        for row in read_rows()
        if row.get('record_type') == 'prediction'
    }
    try:
        names = finished_names()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr)
        return []
    for name in pending_downloads(names, predictions, outcomes):
        fetch(name)
    groups = comparison_groups(predictions)
    scored = []
    for problem, members in groups.items():
        if any(name in outcomes for name in members):
            continue
        measured = {
            name: _measured(name, predictions[name]['changed'].get('epochs', 32))
            for name in members
        }
        if any(value is None for value in measured.values()):
            continue
        falsified = _family_falsified(problem, measured)
        for name, value in measured.items():
            prediction = predictions[name]
            interval = prediction['interval']
            hit = bool(interval[0] <= value <= interval[1]) if interval is not None else None
            point = prediction['numeric_prediction']
            record_outcome(
                prediction_id=name,
                run_name=name,
                measured_nll=value,
                delta=value - float(prediction['baseline_nll']),
                interval_hit=hit,
                interval_hit_reason='' if interval is not None else prediction['interval_null_reason'],
                falsified=falsified,
                signed_error=None if point is None else value - float(point),
                interpretation=_interpretation(problem, name, value, prediction, measured, falsified),
                revised='',
                follow_up='',
                status='measured',
            )
            _append_status(name, 'finished', value)
            scored.append((name, value, hit, falsified))
    print(json.dumps(dict(scored=scored, audit=audit()), indent=2))
    return scored


if __name__ == '__main__':
    score_finished()
