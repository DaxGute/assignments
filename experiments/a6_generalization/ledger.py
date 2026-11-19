"""Append-only prediction ledger.

A locked prediction is a new line. Outcomes are later lines that reference the
same prediction_id. Nothing rewrites a locked prediction.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

LEDGER = Path('results/a6_generalization/prediction_ledger.jsonl')

PREDICTION_FIELDS = (
    'prediction_id',
    'problem',
    'run_name',
    'baseline_nll',
    'baseline_source',
    'changed',
    'controlled',
    'direction',
    'interval',
    'interval_null_reason',
    'numeric_prediction',
    'diagnostics_expected',
    'mechanism',
    'confidence',
    'falsification',
    'run_identifier',
)

OUTCOME_FIELDS = (
    'prediction_id',
    'run_name',
    'measured_nll',
    'delta',
    'interval_hit',
    'interval_hit_reason',
    'falsified',
    'signed_error',
    'interpretation',
    'revised',
    'follow_up',
    'status',
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _append(row):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')


def read_rows():
    if not LEDGER.is_file():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def lock_prediction(**fields):
    missing = [name for name in PREDICTION_FIELDS if name not in fields]
    if missing:
        raise ValueError(f'Prediction is missing {missing}')
    if fields['interval'] is None and not fields['interval_null_reason']:
        raise ValueError('A null interval needs interval_null_reason.')
    if fields['interval'] is not None:
        lo, hi = fields['interval']
        if not lo <= hi:
            raise ValueError('Prediction interval must be ordered.')
    existing = [row for row in read_rows() if row.get('prediction_id') == fields['prediction_id']]
    if any(row.get('record_type') == 'prediction' for row in existing):
        raise ValueError(f'Prediction {fields["prediction_id"]} is already locked.')
    row = dict(fields)
    row.update(record_type='prediction', status='locked', timestamp=_now())
    _append(row)
    return row


def record_outcome(**fields):
    missing = [name for name in OUTCOME_FIELDS if name not in fields]
    if missing:
        raise ValueError(f'Outcome is missing {missing}')
    predictions = [
        row
        for row in read_rows()
        if row.get('record_type') == 'prediction' and row['prediction_id'] == fields['prediction_id']
    ]
    if len(predictions) != 1:
        raise ValueError(f'Outcome {fields["prediction_id"]} needs exactly one locked prediction.')
    if any(
        row.get('record_type') == 'outcome' and row['prediction_id'] == fields['prediction_id']
        for row in read_rows()
    ):
        raise ValueError(f'Outcome {fields["prediction_id"]} already exists.')
    prediction = predictions[0]
    if fields['interval_hit'] is None and not fields['interval_hit_reason']:
        raise ValueError('A null interval_hit needs interval_hit_reason.')
    if prediction['interval'] is None and fields['interval_hit'] is not None:
        raise ValueError('interval_hit must be null when the locked prediction has no interval.')
    if prediction['interval'] is not None and fields['interval_hit'] not in (True, False):
        raise ValueError('A numeric interval needs a boolean interval_hit.')
    row = dict(fields)
    row.update(record_type='outcome', timestamp=_now(), prediction_timestamp=prediction['timestamp'])
    if row['timestamp'] < prediction['timestamp']:
        raise ValueError('Outcome timestamp precedes the locked prediction.')
    _append(row)
    return row


def audit():
    rows = read_rows()
    errors = []
    predictions = {}
    outcomes = {}
    for index, row in enumerate(rows):
        kind = row.get('record_type')
        identifier = row.get('prediction_id')
        if kind == 'prediction':
            if identifier in predictions:
                errors.append(f'duplicate prediction {identifier}')
            predictions[identifier] = row
            if row.get('interval') is None and not row.get('interval_null_reason'):
                errors.append(f'{identifier} null interval without a reason')
        elif kind == 'outcome':
            if identifier in outcomes:
                errors.append(f'duplicate outcome {identifier}')
            outcomes[identifier] = (index, row)
        else:
            errors.append(f'row {index} has record_type {kind}')
    for identifier, (index, outcome) in outcomes.items():
        prediction = predictions.get(identifier)
        if prediction is None:
            errors.append(f'outcome {identifier} has no prediction')
            continue
        if rows.index(prediction) > index:
            errors.append(f'outcome {identifier} appears before its prediction')
        if outcome['timestamp'] < prediction['timestamp']:
            errors.append(f'outcome {identifier} is timestamped before its prediction')
        if prediction['interval'] is None:
            if outcome['interval_hit'] is not None:
                errors.append(f'{identifier} interval_hit must be null')
            if not outcome.get('interval_hit_reason'):
                errors.append(f'{identifier} missing interval_hit_reason')
        elif outcome['interval_hit'] not in (True, False):
            errors.append(f'{identifier} interval_hit must be boolean')
        if outcome.get('falsified') not in (True, False):
            errors.append(f'{identifier} falsified must be boolean')
        if outcome.get('status') not in ('measured', 'failed', 'invalid'):
            errors.append(f'{identifier} bad outcome status {outcome.get("status")}')
    summary = dict(
        predictions=len(predictions),
        outcomes=len(outcomes),
        open=sorted(set(predictions) - set(outcomes)),
        errors=errors,
        ok=not errors,
    )
    return summary
