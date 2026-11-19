"""Check headline report numbers against the released JSON. No training."""

import json
from pathlib import Path

import numpy as np

ROOT = Path('experiments/a6_generalization/references')
REPORT = Path('experiments/a6_generalization/deliverables/A6_ASSIGNMENT_REPORT.md')


def load(name):
    return json.loads((ROOT / f'{name}.json').read_text())


def main():
    report = REPORT.read_text()
    failures = []

    def need(text, label):
        if text not in report:
            failures.append(f'{label}: report is missing {text}')

    parameters = {row['depth']: row for row in load('parameter_scaling')['runs']}
    for depth, shown in (
        (4, '1.269528'),
        (6, '1.262783'),
        (8, '1.262714'),
        (9, '1.259360'),
        (10, '1.261446'),
    ):
        exact = f"{parameters[depth]['validation_nll']:.6f}"
        if not exact.startswith(shown):
            failures.append(f'd{depth}: JSON {exact} does not start with {shown}')
        need(shown, f'd{depth}')
    dense_count = parameters[8]['parameters']
    loop_stored = json.loads(
        Path('results/a6_generalization/runs/a6-loop-lr0.001-s42/diagnostics.json').read_text()
    )['stored_parameters']
    saved = dense_count - loop_stored
    if saved != 19011200 or abs(saved / dense_count - 0.625) > 0.001:
        failures.append(f'loop saves {saved} parameters, not 19,011,200')
    need('19,011,200', 'parameters saved by the loop')
    d3 = json.loads(
        Path('results/a6_generalization/runs/a6-dense-d3-lr0.001-s42/history.json').read_text()
    )
    d3_end = d3[-1]['validation_nll']
    if f'{d3_end:.6f}' != '1.270093':
        failures.append(f'dense depth 3 LR 0.001 is {d3_end:.6f}')
    if min(row['validation_nll'] for row in d3) != d3_end:
        failures.append('dense depth 3 LR 0.001 endpoint is not the minimum')
    loop = json.loads(
        Path('results/a6_generalization/runs/a6-loop-lr0.001-s42/history.json').read_text()
    )
    if f'{(d3_end - loop[-1]["validation_nll"]):.4f}' != '0.0054':
        failures.append('dense depth 3 gap versus the loop is not 0.0054')
    d3_diag = json.loads(
        Path('results/a6_generalization/runs/a6-dense-d3-lr0.001-s42/diagnostics.json').read_text()
    )
    if d3_diag['stored_parameters'] != 11426688:
        failures.append('dense depth 3 stored count is not the loop count')
    if f"{d3_diag['sharpness']['max_loss_increase']:.4f}" != '0.0687':
        failures.append('dense depth 3 sharpness is not 0.0687')
    need('1.270093', 'dense depth 3 LR 0.001')
    need('0.0687', 'dense depth 3 sharpness')
    for lr, shown, delta in (
        ('0.003', '1.272276', '0.0075'),
        ('0.01', '1.284576', '0.0092'),
    ):
        history = json.loads(
            Path(f'results/a6_generalization/runs/a6-dense-d3-lr{lr}-s42/history.json').read_text()
        )
        loop_h = json.loads(
            Path(f'results/a6_generalization/runs/a6-loop-lr{lr}-s42/history.json').read_text()
        )
        end = history[-1]['validation_nll']
        if f'{end:.6f}' != shown:
            failures.append(f'dense depth 3 LR {lr} is {end:.6f}')
        if f'{(end - loop_h[-1]["validation_nll"]):.4f}' != delta:
            failures.append(f'dense depth 3 LR {lr} loop gap is not {delta}')
        if min(row['validation_nll'] for row in history) != end:
            failures.append(f'dense depth 3 LR {lr} endpoint is not the minimum')
        need(shown, f'dense depth 3 LR {lr}')
    for lr, shown in (('0.001', '1.395819'), ('0.003', '1.392914'), ('0.01', '1.396202')):
        history = json.loads(
            Path(f'results/a6_generalization/runs/a6-sgd-lr{lr}-wd3.2-s42/history.json').read_text()
        )
        end = history[-1]['validation_nll']
        if f'{end:.6f}' != shown:
            failures.append(f'SGD LR {lr} is {end:.6f}')
        need(shown, f'SGD LR {lr}')
    drop25 = json.loads(
        Path('results/a6_generalization/runs/a6-dropout-p0.25-s43/history.json').read_text()
    )
    if f"{drop25[-1]['validation_nll']:.6f}" != '1.260817':
        failures.append('dropout 0.25 seed 43 is not 1.260817')
    need('1.260817', 'dropout 0.25 seed 43')
    checks = {
        'a6-ft-related-n30000-e4-lr0.0001-wd3.2-s42': '1.228767',
        'a6-ft-related-n30000-e4-lr0.0001-wd3.2-s42-d43': '1.228823',
        'a6-ft-scratch-n30000-e4-lr0.003-wd3.2-s42-d43': '1.276788',
        'a6-ft-dclm-n30000-e4-lr0.001-wd3.2-s42-d43': '1.273460',
        'a6-ft-noise-n30000-e4-lr0.001-wd3.2-s42-d43': '1.291644',
        'a6-ft-related-n30000-e16-lr0.001-wd3.2-s42-d43': '1.265161',
        'a6-ft-scratch-n30000-e16-lr0.001-wd3.2-s42-d43': '1.263249',
        'a6-ft-dclm-n30000-e16-lr0.001-wd3.2-s42-d43': '1.261844',
        'a6-ft-noise-n30000-e16-lr0.001-wd3.2-s42-d43': '1.262181',
        'a6-ft-scratch-n30000-e4-lr0.003-wd3.2-s42': '1.276589',
        'a6-ft-related-n30000-e16-lr0.001-wd3.2-s42': '1.264776',
        'a6-ft-related-n30000-e16-lr0.0001-wd3.2-s42': '1.324141',
        'a6-ft-dclm-n30000-e16-lr0.001-wd3.2-s42': '1.261285',
        'a6-ft-noise-n30000-e16-lr0.001-wd3.2-s42': '1.263121',
        'a6-ft-related-n30000-e1-lr0.0003-wd3.2-s42-d43': '1.228450',
        'a6-ft-noise-n30000-e1-lr0.003-wd3.2-s42-d43': '1.307638',
    }
    for name, shown in checks.items():
        history = json.loads((Path('results/a6_generalization/runs') / name / 'history.json').read_text())
        if f"{history[-1]['validation_nll']:.6f}" != shown:
            failures.append(f'{name} is {history[-1]["validation_nll"]:.6f}, not {shown}')
        need(shown, name)
    wave3 = {
        'a6-ft-related-n1000-e16-lr0.001-wd3.2-s42': '3.405294',
        'a6-ft-related-n3000-e16-lr0.001-wd3.2-s42': '2.806183',
        'a6-ft-related-n10000-e16-lr0.001-wd3.2-s42': '1.417181',
        'a6-ft-scratch-n1000-e16-lr0.001-wd3.2-s42': '1.311475',
        'a6-ft-scratch-n10000-e16-lr0.001-wd3.2-s42': '1.293627',
        'a6-ft-noise-n10000-e16-lr0.001-wd3.2-s42': '1.285362',
        'a6-ft-related-n30000-e16-lr0.001-wd0-s42': '1.665846',
        'a6-ft-related-n30000-e16-lr0.001-wd3.2-frozen_body-s42': '1.243429',
        'a6-ft-related-n30000-e16-lr0.001-wd3.2-frozen_head-s42': '1.265271',
        'a6-ft-related-n30000-e16-lr0.001-wd3.2-toward_initialization-s42': '1.308736',
        'a6-ft-related-n30000-e16-hyperball-lr0.03-s42': '1.268391',
        'a6-ft-related-n30000-e16-hyperball-lr0.003-s42': '1.794142',
    }
    for name, shown in wave3.items():
        history = json.loads((Path('results/a6_generalization/runs') / name / 'history.json').read_text())
        if f"{history[-1]['validation_nll']:.6f}" != shown:
            failures.append(f'{name} is {history[-1]["validation_nll"]:.6f}, not {shown}')
        need(shown, name)
    body = json.loads(Path(
        'results/a6_generalization/runs/a6-ft-related-n30000-e16-lr0.001-wd3.2-frozen_body-s42/history.json'
    ).read_text())
    if min(row['validation_nll'] for row in body) != body[-1]['validation_nll']:
        failures.append('frozen body endpoint is not the minimum')
    if '1.243429' not in Path('experiments/a6_generalization/deliverables/A6_PRACTICE_QUIZ_SOLUTIONS.md').read_text():
        failures.append('Q20 solution does not cite the frozen-body result')
    report = Path('experiments/a6_generalization/deliverables/A6_ASSIGNMENT_REPORT.md').read_text()
    for phrase in ('1.2292 + 1.1209', 'RMSE 0.0030', 'RMSE 0.223', 'figures/p4_pools.png'):
        if phrase not in report:
            failures.append(f'report is missing the pool-scaling claim {phrase}')
    need('62.5%', 'fraction saved by the loop')

    grid = load('lr_wd_runs')['runs']
    best = min(grid, key=lambda row: row['validation_nll'])
    need(f"{best['validation_nll']:.6f}"[:8], 'best 16-epoch cell')
    if (best['lr'], best['wd']) != (0.01, 0.3):
        failures.append(f"best cell is lr {best['lr']} wd {best['wd']}, not 0.01 and 0.3")
    scratch16 = {
        (row['lr'], f"{row['validation_nll']:.6f}")
        for row in grid
        if row['wd'] == 3.2
    }
    for pair in ((0.001, '1.263681'), (0.003, '1.264844'), (0.01, '1.280172')):
        if pair not in scratch16:
            failures.append(f'16-epoch WD 3.2 missing {pair}')
        need(pair[1], '16-epoch scratch')
    related = json.loads(
        Path('results/a6_generalization/runs/a6-ft-related-n30000-e1-lr0.0003-wd3.2-s42/history.json').read_text()
    )[-1]['validation_nll']
    d9 = parameters[9]['validation_nll']
    scratch16 = next(row['validation_nll'] for row in grid if row['wd'] == 3.2 and row['lr'] == 0.001)
    if abs((scratch16 - related) - 0.0346) > 0.00015:
        failures.append(f'related vs 16-epoch scratch gap is {scratch16 - related:.4f}, not 0.0346')
    if abs((d9 - related) - 0.0303) > 0.00015:
        failures.append(f'related vs d9 gap is {d9 - related:.4f}, not 0.0303')
    need('0.0346', 'related vs 16-epoch scratch')
    need('0.0303', 'related vs d9')
    ensemble_k5 = load('ensemble_curves')['groups']['e32_p0']['ensemble_validation_nll'][4]
    if abs((ensemble_k5 - related) - 0.0252) > 0.00015:
        failures.append(f'related vs ensemble gap is {ensemble_k5 - related:.4f}, not 0.0252')
    need('0.0252', 'related vs ensemble')
    initials = {}
    for init in ('related', 'dclm', 'scratch', 'noise'):
        values = []
        for path in Path('results/a6_generalization/runs').glob(f'a6-ft-{init}-n30000-e1-*/initial.json'):
            values.append(json.loads(path.read_text())['validation_nll'])
        if len(values) < 4 or len(set(values)) != 1:
            failures.append(f'{init} step-0 losses are not one shared value: {values}')
            continue
        initials[init] = values[0]
    shown = {'related': '1.2576', 'dclm': '6.0702', 'scratch': '3.3739', 'noise': '1.4142'}
    for init, text in shown.items():
        if init not in initials:
            continue
        if abs(initials[init] - float(text)) > 5e-5:
            failures.append(f'{init} step-0 is {initials[init]:.4f}, not {text}')
        need(text, f'{init} step-0')
    if 'related' in initials and 'scratch' in initials:
        gap = initials['scratch'] - initials['related']
        if abs(gap - 2.116) > 5e-4:
            failures.append(f'step-0 gap is {gap:.4f}, not 2.116')
        need('2.116', 'step-0 gap')
    bad = json.loads(Path(
        'results/a6_generalization/runs/a6-ft-related-n30000-e1-lr0.003-wd3.2-s42/history.json'
    ).read_text())
    early = min(bad, key=lambda row: abs(float(row['epochs']) - 0.062))
    spike = min(bad, key=lambda row: abs(float(row['epochs']) - 0.186))
    if abs(early['validation_nll'] - 1.2477) > 5e-4:
        failures.append(f'early related LR 0.003 is {early["validation_nll"]:.4f}, not 1.2477')
    if abs(spike['validation_nll'] - 1.3212) > 5e-4:
        failures.append(f'related LR 0.003 spike is {spike["validation_nll"]:.4f}, not 1.3212')
    need('1.2477', 'related early')
    need('1.3212', 'related spike')

    ensemble = load('ensemble_curves')['groups']['e32_p0']['ensemble_validation_nll']
    for value, shown in zip(ensemble, ('1.261927', '1.257089', '1.255512', '1.254726', '1.254255')):
        if f'{value:.6f}' != shown:
            failures.append(f'ensemble {value:.6f} != {shown}')
        need(shown, 'ensemble')
    import numpy as np
    mean = np.asarray(ensemble, dtype=float)
    k = np.arange(1, 6, dtype=float)

    def _form(basis):
        amplitude, offset = np.linalg.lstsq(
            np.column_stack((basis, np.ones_like(k))), mean, rcond=None
        )[0]
        residual = offset + amplitude * basis - mean
        return float(amplitude), float(offset), float(np.sqrt(np.mean(residual ** 2)))

    inv_amplitude, inv_offset, inv_rmse = _form(k ** -1)
    sqrt_offset = _form(k ** -0.5)[1]
    sqrt_rmse = _form(k ** -0.5)[2]
    exp_offset, exp_rmse = _form(np.exp(-k))[1:]
    if abs(inv_rmse - 1.6e-5) > 2e-6 or abs(inv_offset + inv_amplitude / 10 - 1.2533) > 5e-4:
        failures.append(f'1/K ensemble form is offset {inv_offset:.5f}, RMSE {inv_rmse:.3e}')
    if abs(sqrt_offset - 1.2476) > 5e-4 or abs(sqrt_rmse - 3.0e-4) > 2e-5:
        failures.append(f'square-root ensemble form is offset {sqrt_offset:.4f}, RMSE {sqrt_rmse:.3e}')
    if abs((exp_offset - mean[-1]) - 5e-5) > 2e-5 or abs(exp_rmse - 1.2e-4) > 2e-5:
        failures.append(f'exponential ensemble form is {exp_offset - mean[-1]:+.6f} above K=5, RMSE {exp_rmse:.3e}')
    need('1.2476', 'square-root ensemble offset')
    need('1.6\\times 10^{-5}', '1/K ensemble RMSE')
    need('3.0\\times 10^{-4}', 'square-root ensemble RMSE')
    need('1.2\\times 10^{-4}', 'exponential ensemble RMSE')

    batches = sorted(load('batch_scaling')['runs'], key=lambda row: row['batch'])
    for row, shown in zip(batches, ('1.262385', '1.262714', '1.269413', '1.295902', '1.394662')):
        if f"{row['validation_nll']:.6f}" != shown:
            failures.append(f"batch {row['batch']}: {row['validation_nll']:.6f} != {shown}")

    wd = {row['wd']: row['validation_nll'] for row in load('wd_tuning')['runs']}
    if f"{wd[0.1]:.6f}" != '1.405700' or f"{wd[3.2]:.6f}" != '1.262714':
        failures.append(f'32-epoch weight-decay controls are {wd}')

    schedules = load('schedules')['runs']
    constant = schedules['constant']
    longest = max(constant, key=lambda row: row['epochs'])
    history = {round(float(point['epochs']), 5): point['validation_nll'] for point in longest['history']}
    shown_constant = {16: '1.265659', 32: '1.262673', 64: '1.261642', 128: '1.260447', 256: '1.260759'}
    for row in constant:
        epochs = int(row['epochs'])
        logged = history[round(float(epochs), 5)]
        if logged != row['validation_nll']:
            failures.append(f'constant epoch {epochs} is not on the 256-epoch trajectory')
        shown = shown_constant[epochs]
        if f"{logged:.6f}" != shown:
            failures.append(f'constant epoch {epochs}: {logged:.6f} != {shown}')
        need(shown, f'constant {epochs}')
    shown_linear = {
        16: ('1.261350', '0.001414'),
        32: ('1.261138', '0.003517'),
        64: ('1.268315', '0.01'),
        128: ('1.265813', '0.01'),
        256: ('1.266993', '0.01'),
    }
    for row in schedules['linear']:
        loss, lr = shown_linear[int(row['epochs'])]
        if f"{row['validation_nll']:.6f}" != loss:
            failures.append(f"linear epoch {row['epochs']}: {row['validation_nll']:.6f} != {loss}")
        if f"{row['lr']:.6f}".rstrip('0') not in (lr, lr.rstrip('0')) and not f"{row['lr']:.6g}".startswith(lr):
            if abs(row['lr'] - float(lr)) > 5e-7:
                failures.append(f"linear epoch {row['epochs']} lr {row['lr']} != {lr}")
        need(loss, f"linear {row['epochs']}")

    from experiments.a6_generalization.quadratic_fit import fit_quadratic
    import numpy as np

    positive = [row for row in grid if row['wd'] > 0]
    fit = fit_quadratic(
        [row['lr'] for row in positive],
        [row['wd'] for row in positive],
        [row['validation_nll'] for row in positive],
    )
    if f'{fit.rmse:.6f}' != '0.008594':
        failures.append(f'quadratic RMSE {fit.rmse:.6f} != 0.008594')
    need('0.008594', 'quadratic rmse')
    coeff = fit.coefficients
    hessian = np.array([[2 * coeff[3], coeff[4]], [coeff[4], 2 * coeff[5]]])
    step = np.linalg.solve(hessian, -coeff[1:3])
    lr_star = float(np.exp(step[0] + fit.log_centers[0]))
    wd_star = float(np.exp(step[1] + fit.log_centers[1]))
    if not (0.014 < lr_star < 0.016 and 0.17 < wd_star < 0.19):
        failures.append(f'quadratic vertex is LR {lr_star:.4f}, WD {wd_star:.4f}')
    if abs(float(fit.predict(lr_star, wd_star)) - 1.2634) > 5e-4:
        failures.append('quadratic vertex prediction is not 1.2634')
    if np.linalg.eigvalsh(hessian).min() <= 0:
        failures.append('quadratic critical point is not a minimum')
    need('0.015', 'quadratic vertex lr')
    need('0.179', 'quadratic vertex wd')
    need('1.2634', 'quadratic vertex loss')
    solutions = Path('experiments/a6_generalization/deliverables/A6_PRACTICE_QUIZ_SOLUTIONS.md').read_text()
    if '1.265620' not in solutions:
        failures.append('Q22 solution omits the MoE seed-43 endpoint')
    q2 = solutions.split('**Q2.')[1].split('**Q3.')[0]
    q3 = solutions.split('**Q3.')[1].split('**Q4.')[0]
    wd32 = next(row for row in grid if row['wd'] == 3.2 and row['lr'] == 0.003)
    wd1 = next(row for row in grid if row['wd'] == 1.0 and row['lr'] == 0.003)
    if not str(wd32['run_id']).startswith('cc5e09349d') or not str(wd1['run_id']).startswith('8cc78664'):
        failures.append('16-epoch run ids for weight decay 3.2 and 1.0 are not the expected cells')
    if 'cc5e09349d' not in q2 or '8cc78664' in q2:
        failures.append('Q2 cites the weight-decay 1.0 run for the weight-decay 3.2 comparison')
    if '8cc78664' not in q3:
        failures.append('Q3 is missing the weight-decay 1.0 cell id')
    import math
    def _prob(logits):
        shift = max(logits)
        weights = [math.exp(value - shift) for value in logits]
        total = sum(weights)
        return [value / total for value in weights]
    averaged = [(a + b) / 2 for a, b in zip(_prob([0.0, 5.0]), _prob([0.0, 0.0]))]
    probability_nll = -math.log(averaged[1])
    logit_nll = -math.log(_prob([0.0, 2.5])[1])
    if abs(probability_nll - 0.292) > 0.001 or abs(logit_nll - 0.079) > 0.001:
        failures.append(f'logit-average counterexample is {probability_nll:.3f} vs {logit_nll:.3f}')
    q17 = solutions.split('**Q17.')[1].split('**Q18.')[0]
    if '0.292' not in q17 or '0.079' not in q17:
        failures.append('Q17 solution is missing the logit-average counterexample')
    quiz = Path('experiments/a6_generalization/deliverables/A6_PRACTICE_QUIZ.md').read_text()
    q17_quiz = quiz.split('**Q17.')[1].split('**Q18.')[0]
    if 'is the lower loss' in q17_quiz or 'neither loss is guaranteed' not in q17_quiz:
        failures.append('Q17 choice still ranks the probability average as the lower loss')
    for query, shown in (((0.006, 0.5), '1.265'), ((0.03, 0.1), '1.2649')):
        predicted = float(fit.predict(*query))
        if abs(predicted - float(shown)) > 5e-4:
            failures.append(f'quiz surface {query} is {predicted:.4f}, not {shown}')
        if shown not in solutions:
            failures.append(f'solutions missing surface prediction {shown}')
    lr = np.array([row['lr'] for row in positive])
    wd = np.array([row['wd'] for row in positive])
    loss = np.array([row['validation_nll'] for row in positive])
    rounded = (
        1.26665
        - 0.007452 * (np.log(lr) + 5.774)
        - 0.002317 * (np.log(wd) - 0.560)
        + 0.009183 * (np.log(lr) + 5.774) ** 2
        + 0.009434 * (np.log(lr) + 5.774) * (np.log(wd) - 0.560)
        + 0.002754 * (np.log(wd) - 0.560) ** 2
    )
    sharp = json.loads(Path('results/a6_generalization/parameter_sharpness.json').read_text())
    shown_sharp = {
        4: '0.0416', 5: '0.0201', 6: '0.1227', 7: '0.0346',
        8: '0.1000', 9: '0.0447', 10: '0.0268',
    }
    for row in sharp['runs']:
        shown = shown_sharp[row['depth']]
        if f"{row['max_loss_increase']:.4f}" != shown:
            failures.append(f"depth {row['depth']} sharpness {row['max_loss_increase']:.4f} != {shown}")
        need(shown, f"sharpness d{row['depth']}")
    local_b64 = json.loads(Path('results/a6_generalization/b64_sharpness.json').read_text())
    if f"{local_b64['0.05']['max_loss_increase']:.4f}" != '0.1000':
        failures.append('batch-64 local sharpness is not 0.1000')
    released_sharp = {
        16: '0.1203',
        256: '0.1016',
        1024: '0.1699',
        4096: '0.2824',
    }
    for row in batches:
        if row['batch'] == 64:
            continue
        value = row['sharpness']['directional']['0.05']['max_loss_increase']
        shown = released_sharp[row['batch']]
        if f'{value:.4f}' != shown:
            failures.append(f"batch {row['batch']} sharpness {value:.4f} != {shown}")
        need(shown, f"batch {row['batch']} sharpness")

    if float(np.max(np.abs(rounded - fit.predict(lr, wd)))) > 3e-6:
        failures.append('printed quadratic differs from the official fit by more than 3e-6')
    residuals = loss - fit.predict(lr, wd)
    worst = int(np.argmax(np.abs(residuals)))
    if not (np.isclose(lr[worst], 0.001) and np.isclose(wd[worst], 0.3)):
        failures.append('the largest residual is no longer learning rate 0.001, weight decay 0.3')

    epochs = {int(row['epochs']): row for row in load('epoch_scaling')['runs']}
    shown_epochs = {8: ('1.268735', 3.2), 16: ('1.263182', 1.0), 32: ('1.262714', 3.2), 64: ('1.269079', 3.2)}
    for epoch, (shown, wd) in shown_epochs.items():
        row = epochs[epoch]
        if f"{row['validation_nll']:.6f}" != shown or row['wd'] != wd or row['lr'] != 0.003:
            failures.append(f"epoch sweep {epoch}: {row['validation_nll']} wd {row['wd']}")
        need(shown, f'epoch sweep {epoch}')
    long = epochs[64]['history']
    minimum = min(long, key=lambda point: point['validation_nll'])
    if minimum['step'] != 26264 or f"{minimum['validation_nll']:.6f}" != '1.264698':
        failures.append(f"64-epoch minimum is {minimum}, not step 26264 at 1.264698")
    need('1.264698', '64-epoch minimum')
    baseline_history = {point['step']: point['validation_nll'] for point in epochs[32]['history']}
    if f"{baseline_history[5628]:.6f}" != '1.320758' or f"{baseline_history[3752]:.6f}" != '1.292658':
        failures.append('32-epoch epoch-12 spike does not match 1.320758 after 1.292658')
    if min(baseline_history.values()) != epochs[32]['validation_nll']:
        failures.append('32-epoch endpoint is not the logged minimum')
    need('1.320758', 'epoch-12 spike')

    from experiments.a6_generalization.p2_regularization import ensemble_curve

    chunk_losses = np.array(ensemble_curve()['chunk_losses'], dtype=float)
    standard_error = float(chunk_losses.std(axis=1, ddof=1).mean() / np.sqrt(chunk_losses.shape[1]))
    if abs(standard_error - 0.0032) > 0.00015:
        failures.append(f'validation-mean standard error is {standard_error:.4f}, not 0.0032')
    need('0.0032', 'validation standard error')

    runs = Path('results/a6_generalization/runs')
    trained = {
        'a6-moe-lr0.003-s42': '1.268434',
        'a6-moe-lr0.001-s42': '1.367250',
        'a6-loop-lr0.001-s42': '1.264715',
        'a6-hyperball-lr0.03-s42': '1.271385',
        'a6-dropout-p0.05-s42': '1.261245',
        'a6-dropout-p0.25-s42': '1.262897',
        'a6-ft-related-n30000-e1-lr0.0003-wd3.2-s42': '1.229063',
        'a6-ft-scratch-n30000-e1-lr0.003-wd3.2-s42': '1.298103',
        'a6-ft-noise-n30000-e1-lr0.003-wd3.2-s42': '1.321914',
        'a6-sam-rho0.01-s42': '1.261672',
        'a6-sam-rho0.05-s42': '1.263153',
        'a6-sam-rho0.1-s42': '1.273942',
        'a6-moe-lr0.003-s43': '1.265620',
        'a6-sam-rho0.01-s43': '1.263302',
        'a6-loop-lr0.001-s43': '1.263934',
        'a6-hyperball-lr0.03-s43': '1.271424',
    }
    for name, shown in trained.items():
        history = json.loads((runs / name / 'history.json').read_text())
        exact = f"{history[-1]['validation_nll']:.6f}"
        if exact != shown:
            failures.append(f'{name}: history {exact} is not {shown}')
        if shown not in report and f'{float(shown):.4f}' not in report:
            failures.append(f'{name}: report is missing {shown}')
    second = json.loads((runs / 'a6-moe-lr0.003-s43' / 'history.json').read_text())
    if min(row['validation_nll'] for row in second) != second[-1]['validation_nll']:
        failures.append('MoE seed 43 endpoint is not the logged minimum')
    sharp = json.loads((runs / 'a6-moe-lr0.003-s43' / 'diagnostics.json').read_text())
    if f"{sharp['sharpness']['max_loss_increase']:.4f}" != '0.1492':
        failures.append('MoE seed 43 sharpness is not 0.1492')
    need('0.1492', 'MoE seed 43 sharpness')
    if abs((second[-1]['validation_nll'] - 1.2627138263769448) - 0.002906) > 2e-5:
        failures.append('MoE seed 43 gap versus dense is not 0.0029')
    first = json.loads((runs / 'a6-moe-lr0.003-s42' / 'history.json').read_text())
    at20 = {
        round(row['epochs'], 1): row['validation_nll']
        for row in first
    }
    second20 = next(row['validation_nll'] for row in second if round(row['epochs'], 1) == 20)
    if abs((second20 - at20[20.0]) - 0.0092) > 2e-4:
        failures.append('MoE epoch-20 seed gap is not 0.009')
    need('At epoch 20 they differed by 0.009', 'MoE midpoint gap')
    sam43 = json.loads((runs / 'a6-sam-rho0.01-s43' / 'history.json').read_text())
    if min(row['validation_nll'] for row in sam43) != sam43[-1]['validation_nll']:
        failures.append('SAM seed 43 endpoint is not the logged minimum')
    sam_sharp = json.loads((runs / 'a6-sam-rho0.01-s43' / 'diagnostics.json').read_text())
    if f"{sam_sharp['sharpness']['max_loss_increase']:.4f}" != '0.0380':
        failures.append('SAM seed 43 sharpness is not 0.0380')
    need('0.0380', 'SAM seed 43 sharpness')
    loop43 = json.loads((runs / 'a6-loop-lr0.001-s43' / 'history.json').read_text())
    if min(row['validation_nll'] for row in loop43) != loop43[-1]['validation_nll']:
        failures.append('loop seed 43 endpoint is not the logged minimum')
    loop_sharp = json.loads((runs / 'a6-loop-lr0.001-s43' / 'diagnostics.json').read_text())
    if f"{loop_sharp['sharpness']['max_loss_increase']:.4f}" != '0.0572':
        failures.append('loop seed 43 sharpness is not 0.0572')
    need('0.0572', 'loop seed 43 sharpness')
    if abs((loop43[-1]['validation_nll'] - 1.2627138263769448) - 0.001220) > 2e-5:
        failures.append('loop seed 43 gap versus dense is not 0.0012')
    loop42 = json.loads((runs / 'a6-loop-lr0.001-s42' / 'history.json').read_text())
    if abs((loop42[-1]['validation_nll'] - loop43[-1]['validation_nll']) - 0.000781) > 2e-5:
        failures.append('loop seed gap is not 0.0008')
    if 'still required before the tie' in report:
        failures.append('loop second seed is still described as unfinished')
    hb43 = json.loads((runs / 'a6-hyperball-lr0.03-s43' / 'history.json').read_text())
    hb_best = min(hb43, key=lambda row: row['validation_nll'])
    if abs(hb_best['epochs'] - 28) > 0.1 or abs(hb_best['validation_nll'] - 1.266788) > 5e-5:
        failures.append('Hyperball seed 43 best point is not epoch 28')
    if hb43[-1]['validation_nll'] <= hb_best['validation_nll']:
        failures.append('Hyperball seed 43 endpoint is the minimum, but it rose after epoch 28')
    hb_sharp = json.loads((runs / 'a6-hyperball-lr0.03-s43' / 'diagnostics.json').read_text())
    if f"{hb_sharp['sharpness']['max_loss_increase']:.4f}" != '0.0585':
        failures.append('Hyperball seed 43 sharpness is not 0.0585')
    need('0.0585', 'Hyperball seed 43 sharpness')
    need('1.266788', 'Hyperball seed 43 best logged point')
    if 'No second seed of learning rate 0.03 has been run' in report:
        failures.append('Hyperball second seed is still described as unfinished')
    for lr, shown in (('0.003', '0.0905'), ('0.01', '0.0406'), ('0.03', '0.0495')):
        diag = json.loads((runs / f'a6-hyperball-lr{lr}-s42' / 'diagnostics.json').read_text())
        got = diag['sharpness']['max_loss_increase']
        if f'{got:.4f}' != shown:
            failures.append(f'hyperball LR {lr} sharpness {got:.4f} != {shown}')
        need(shown, f'hyperball sharp {lr}')
    hb = json.loads((runs / 'a6-hyperball-lr0.003-s42' / 'history.json').read_text())
    peak = max(hb, key=lambda row: row['validation_nll'])
    if abs(peak['validation_nll'] - 1.6252) > 5e-4 or abs(peak['epochs'] - 28) > 0.1:
        failures.append(
            f"hyperball 0.003 peak is {peak['validation_nll']:.4f} at epoch {peak['epochs']}"
        )
    need('1.6252', 'hyperball peak')
    from experiments.a6_generalization.analysis import chunk_bootstrap

    mixed = np.load('results/a6_generalization/dropout_with_bank_chunks.npz')
    five = mixed['five']
    if abs(float(five.mean()) - 1.2542546561438843) > 1e-12:
        failures.append('five-model ensemble chunks do not match 1.254255')
    for key, six_text, gain_text, lo_text, hi_text in (
        ('p05', '1.254087', '0.000167', '0.000113', '0.000221'),
        ('p15', '1.254293', '−0.000038', '−0.000163', '0.000077'),
        ('p25', '1.254445', '−0.000190', '−0.000350', '−0.000045'),
    ):
        six = mixed[key]
        gain = float((five - six).mean())
        samples = chunk_bootstrap(np.stack([five, six]), repeats=4000, seed=42)
        lo, hi = np.quantile(samples[0] - samples[1], [0.05, 0.95])
        if f'{float(six.mean()):.6f}' != six_text:
            failures.append(f'{key} six-model mean is {six.mean():.6f}, not {six_text}')
        if f'{abs(gain):.6f}' != gain_text.lstrip('−'):
            failures.append(f'{key} gain is {gain:.6f}, not {gain_text}')
        if f'{abs(lo):.6f}' != lo_text.lstrip('−') or f'{abs(hi):.6f}' != hi_text.lstrip('−'):
            failures.append(f'{key} interval is {lo:.6f} to {hi:.6f}')
        need(six_text, f'{key} six-model loss')
        need(gain_text, f'{key} ensemble gain')
        need(lo_text, f'{key} interval low')
        need(hi_text, f'{key} interval high')
    pair = np.load('results/a6_generalization/dropout_p05_pair_chunks.npz')
    s43 = pair['s43']
    s42 = pair['s42']
    best = pair['best']
    both = pair['pair']
    k2 = pair['nodrop_k2']
    if f'{float(s43.mean()):.6f}' != '1.258402':
        failures.append(f"dropout 0.05 seed 43 logits are {s43.mean():.6f}")
    history = json.loads(
        Path('results/a6_generalization/runs/a6-dropout-p0.05-s43/history.json').read_text()
    )
    if abs(history[-1]['validation_nll'] - float(s43.mean())) > 1e-6:
        failures.append('dropout 0.05 seed 43 logits do not match the endpoint')
    if min(row['validation_nll'] for row in history) != history[-1]['validation_nll']:
        failures.append('dropout 0.05 seed 43 endpoint is not the minimum')
    sharp = json.loads(
        Path('results/a6_generalization/runs/a6-dropout-p0.05-s43/diagnostics.json').read_text()
    )
    if f"{sharp['sharpness']['max_loss_increase']:.4f}" != '0.1394':
        failures.append('dropout 0.05 seed 43 sharpness is not 0.1394')
    checks = {
        '0.002843': float((s42 - s43).mean()),
        '0.002403': float((best - s43).mean()),
        '1.256048': float(both.mean()),
        '0.003776': float(((s42 + s43) / 2 - both).mean()),
        '0.001041': float((k2 - both).mean()),
    }
    for shown, got in checks.items():
        if f'{got:.6f}' != shown:
            failures.append(f'dropout pair {shown} computed as {got:.6f}')
        need(shown, f'dropout pair {shown}')
    need('0.1394', 'dropout seed 43 sharpness')
    need('1.255245', 'dropout seed 43 interval')
    p15 = np.load('results/a6_generalization/dropout_p15_pair_chunks.npz')
    s43 = p15['s43']
    s42 = p15['s42']
    best = p15['best']
    both = p15['pair']
    k2 = p15['nodrop_k2']
    p15_checks = {
        '1.258863': float(s43.mean()),
        '0.002739': float((s42 - s43).mean()),
        '0.001942': float((best - s43).mean()),
        '1.257571': float(both.mean()),
        '0.002661': float(((s42 + s43) / 2 - both).mean()),
        '0.000482': abs(float((k2 - both).mean())),
    }
    for shown, got in p15_checks.items():
        if f'{got:.6f}' != shown:
            failures.append(f'dropout 0.15 {shown} computed as {got:.6f}')
        need(shown, f'dropout 0.15 {shown}')
    sharp = json.loads(
        Path('results/a6_generalization/runs/a6-dropout-p0.15-s43/diagnostics.json').read_text()
    )
    if f"{sharp['sharpness']['max_loss_increase']:.4f}" != '0.1315':
        failures.append('dropout 0.15 seed 43 sharpness is not 0.1315')
    history = json.loads(
        Path('results/a6_generalization/runs/a6-dropout-p0.15-s43/history.json').read_text()
    )
    if min(row['validation_nll'] for row in history) != history[-1]['validation_nll']:
        failures.append('dropout 0.15 seed 43 endpoint is not the minimum')
    if abs(history[-1]['validation_nll'] - float(s43.mean())) > 1e-6:
        failures.append('dropout 0.15 seed 43 logits do not match the endpoint')
    gap = chunk_bootstrap(np.stack([both, k2]), repeats=4000, seed=42)
    gap = gap[0] - gap[1]
    lo, hi = np.quantile(gap, [0.05, 0.95])
    if f'{float(lo):.6f}' != '-0.000201' or f'{float(hi):.6f}' != '0.001229':
        failures.append(f'dropout 0.15 pair-minus-K2 interval is {lo:.6f} to {hi:.6f}')
    need('−0.000201', 'dropout 0.15 pair above K=2, low')
    need('0.001229', 'dropout 0.15 pair above K=2, high')
    need('0.1315', 'dropout 0.15 sharpness')
    need('1.255601', 'dropout 0.15 interval')
    need('includes zero', 'dropout 0.15 pair versus K=2')

    def logged(name, epoch):
        history = json.loads((runs / name / 'history.json').read_text())
        return min(history, key=lambda row: abs(float(row['epochs']) - epoch))['validation_nll']

    bumps = (
        ('a6-dropout-p0.25-s42', ((16, '1.2879'), (18, '1.3171'), (20, '1.2888'))),
        ('a6-sam-rho0.1-s42', ((8, '1.2999'), (10, '1.3307'), (12, '1.2971'))),
    )
    for name, points in bumps:
        for epoch, shown in points:
            got = logged(name, epoch)
            if abs(got - float(shown)) > 5e-4:
                failures.append(f'{name} epoch {epoch} is {got:.4f}, not {shown}')
            need(shown, f'{name} e{epoch}')
    from experiments.a6_generalization.collect import linear_lr_fraction

    fractions = {18: '44%', 12: '63%', 10: '69%'}
    for epoch, shown in fractions.items():
        got = linear_lr_fraction(epoch * 469, 15008, 75)
        if f'{round(100 * got):.0f}%' != shown:
            failures.append(f'epoch {epoch} learning-rate fraction is {got:.3f}, not {shown}')
        need(shown, f'lr fraction e{epoch}')
    rises = (
        logged('a6-dropout-p0.25-s42', 18) - logged('a6-dropout-p0.25-s42', 16),
        logged('a6-sam-rho0.1-s42', 10) - logged('a6-sam-rho0.1-s42', 8),
    )
    if any(not 0.025 <= rise <= 0.035 for rise in rises):
        failures.append(f'one-interval bumps are {rises}, not about 0.03')
    need('about 0.03', 'bump size')

    diffs = np.load('results/a6_generalization/dropout_paired_diffs.npz')
    rng = np.random.default_rng(42)
    expected = {
        'p05': ('0.00147', '0.00204', '0.00086'),
        'p15': ('0.00111', '0.00228', '0.00015'),
        'p25': ('0.00018', '0.00127', '0.00180'),
    }
    for key, shown in expected.items():
        values = diffs[key]
        idx = rng.integers(0, len(values), size=(4000, len(values)))
        lo, hi = np.quantile(values[idx].mean(1), [0.05, 0.95])
        got = (f'{abs(values.mean()):.5f}', f'{abs(lo):.5f}', f'{abs(hi):.5f}')
        if got != shown:
            failures.append(f'{key}: paired {got} != {shown}')
        for piece in shown:
            need(piece, f'dropout paired {key}')

    members = np.load('results/a6_generalization/chunk_nll_members.npz')
    order = ['s42', 's43', 's44', 's45', 's46', 'p0.05', 'p0.15', 'p0.25']
    corr = np.corrcoef([members[key] for key in order])
    base = corr[:5, :5]
    base_pairs = base[np.triu_indices(5, k=1)]
    if abs(base_pairs.mean() - 0.996) > 0.001:
        failures.append(f'no-dropout correlation is {base_pairs.mean():.3f}, not 0.996')
    for key, shown, row in (('p0.05', 0.996, 5), ('p0.15', 0.982, 6), ('p0.25', 0.964, 7)):
        got = corr[row, :5].mean()
        if abs(got - shown) > 0.001:
            failures.append(f'{key} correlation is {got:.3f}, not {shown:.3f}')
        need(f'{shown:.3f}', f'{key} correlation')

    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'checked {REPORT} against the released JSON and the finished runs')


if __name__ == '__main__':
    main()
