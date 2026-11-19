"""Stage-0 analysis of the supplied A6 references. No new training."""

from pathlib import Path
import json
import numpy as np

from experiments.a6_generalization.analysis import chunk_bootstrap, power_law, predict
from experiments.a6_generalization.common import runs
from experiments.a6_generalization.p1_architecture import plot_parameters
from experiments.a6_generalization.p2_regularization import ensemble_curve
from experiments.a6_generalization.quadratic_fit import fit_quadratic, plot_example


OUT = Path('results/a6_generalization')


def _fit_sensitivity(x, y):
    fits = {}
    for name, mask in {
        'all': np.ones(len(x), dtype=bool),
        'drop_smallest': np.arange(len(x)) != int(np.argmin(x)),
        'drop_largest': np.arange(len(x)) != int(np.argmax(x)),
    }.items():
        if mask.sum() < 3:
            continue
        fit = power_law(x[mask], y[mask])
        fit['rmse'] = float(np.sqrt(np.mean((predict(fit, x[mask]) - y[mask]) ** 2)))
        fits[name] = fit
    return fits


def parameter_section():
    bundle = runs('parameter_scaling')
    rows = bundle['runs']
    x = np.array([row['parameters'] for row in rows], dtype=float)
    y = np.array([row['validation_nll'] for row in rows], dtype=float)
    order = np.argsort(x)
    ordered = [rows[i] for i in order]
    diffs = np.diff(y[order])
    return dict(
        baseline=bundle['baseline'],
        runs=[
            dict(
                depth=row['depth'],
                parameters=row['parameters'],
                epochs=row['epochs'],
                lr=row['lr'],
                wd=row['wd'],
                validation_nll=row['validation_nll'],
                training_flops=row['training_flops'],
                run_id=row['id'],
            )
            for row in ordered
        ],
        monotonic_in_parameters=bool(np.all(diffs <= 1e-6)),
        adjacent_deltas=[float(value) for value in diffs],
        fit=_fit_sensitivity(x, y),
        selected_lr_values=sorted({row['lr'] for row in rows}),
        selected_epochs=sorted({row['epochs'] for row in rows}),
        note=(
            'Selected recipes all use LR 0.003. Epoch count and weight decay still vary, '
            'so the parameter trend is not a matched-compute comparison. The fitted offset '
            'is a descriptive asymptote inside this fit, not a demonstrated loss.'
        ),
    )


def lr_wd_section():
    bundle = runs('lr_wd_runs')
    rows = bundle['runs']
    positive = [row for row in rows if row['wd'] > 0]
    lr = np.array([row['lr'] for row in positive])
    wd = np.array([row['wd'] for row in positive])
    loss = np.array([row['validation_nll'] for row in positive])
    fit = fit_quadratic(lr, wd, loss)
    product = lr * wd
    # Rank correlation of the product against loss, plus a one-dimensional residual.
    product_order = np.argsort(product)
    loss_order = np.argsort(loss)
    ranks_x = np.empty(len(product))
    ranks_y = np.empty(len(loss))
    ranks_x[product_order] = np.arange(len(product))
    ranks_y[loss_order] = np.arange(len(loss))
    spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    log_product = np.log(product)
    design = np.column_stack((np.ones(len(loss)), log_product, log_product**2))
    coefficient = np.linalg.lstsq(design, loss, rcond=None)[0]
    product_rmse = float(np.sqrt(np.mean((design @ coefficient - loss) ** 2)))
    best = min(positive, key=lambda row: row['validation_nll'])
    reference = [row for row in positive if abs(row['wd'] - 0.1) < 1e-12]
    best_reference = min(reference, key=lambda row: row['validation_nll']) if reference else None
    zero = [row for row in rows if row['wd'] == 0]
    return dict(
        n_runs=len(rows),
        n_positive_wd=len(positive),
        epochs=16,
        lr_values=sorted({float(row['lr']) for row in rows}),
        wd_values=sorted({float(row['wd']) for row in rows}),
        quadratic_rmse=fit.rmse,
        quadratic_coefficients=[float(value) for value in fit.coefficients],
        product_spearman=spearman,
        product_quadratic_rmse=product_rmse,
        best_positive=dict(lr=best['lr'], wd=best['wd'], validation_nll=best['validation_nll'], run_id=best['run_id']),
        best_wd_0_1=None
        if best_reference is None
        else dict(
            lr=best_reference['lr'],
            wd=best_reference['wd'],
            validation_nll=best_reference['validation_nll'],
            run_id=best_reference['run_id'],
        ),
        wd_ratio_best_over_0_1=None if best_reference is None else best['wd'] / best_reference['wd'],
        best_zero_wd=None
        if not zero
        else dict(
            lr=min(zero, key=lambda row: row['validation_nll'])['lr'],
            validation_nll=min(zero, key=lambda row: row['validation_nll'])['validation_nll'],
        ),
        loss_span=float(loss.max() - loss.min()),
    )


def ensemble_section():
    curve = ensemble_curve()
    mean = np.array(curve['ensemble_validation_nll'], dtype=float)
    chunks = np.array(curve['chunk_losses'], dtype=float)
    samples = chunk_bootstrap(chunks, repeats=2000, seed=42)
    # chunk_bootstrap returns (K, repeats) when chunk losses are (K, chunks).
    fit = power_law(np.arange(1, len(mean) + 1, dtype=float), mean)
    fit['rmse'] = float(np.sqrt(np.mean((predict(fit, np.arange(1, 6)) - mean) ** 2)))
    # Leave-one-K-out is impossible below 3 points; compare K=1..4 against K=1..5.
    short = power_law(np.arange(1, 5, dtype=float), mean[:4])
    return dict(
        mean_nll=[float(value) for value in mean],
        bootstrap_mean=[float(value) for value in samples.mean(1)],
        bootstrap_p05=[float(value) for value in np.quantile(samples, 0.05, axis=1)],
        bootstrap_p95=[float(value) for value in np.quantile(samples, 0.95, axis=1)],
        fit_k1_to_k5=fit,
        fit_k1_to_k4=short,
        predicted_k10=float(predict(fit, np.array([10.0]))[0]),
        extrapolation_warning=(
            'Five members do not identify L_infinity. The K=10 number is a fit evaluation, not a measurement.'
        ),
    )


def _final(row):
    if 'validation_nll' in row:
        return row['validation_nll']
    return row['history'][-1]['validation_nll']


def schedule_section():
    bundle = runs('schedules')
    out = {}
    for kind, rows in bundle['runs'].items():
        rows = sorted(rows, key=lambda row: row['epochs'])
        out[kind] = [
            dict(
                epochs=row['epochs'],
                lr=row.get('lr'),
                wd=row.get('wd'),
                validation_nll=_final(row),
                run_id=row.get('id'),
                history_epochs=[point.get('epochs') for point in row.get('history', [])],
            )
            for row in rows
        ]
    constant = {row['epochs']: row for row in out['constant']}
    long = max(bundle['runs']['constant'], key=lambda row: row['epochs'])
    prefix_matches = []
    for point in long['history']:
        short = constant.get(point['epochs'])
        if short is not None:
            prefix_matches.append(
                dict(
                    epochs=point['epochs'],
                    long_history=point['validation_nll'],
                    endpoint=short['validation_nll'],
                    abs_diff=abs(point['validation_nll'] - short['validation_nll']),
                )
            )
    return dict(runs=out, constant_endpoints_match_long_history=prefix_matches)


def batch_section():
    bundle = runs('batch_scaling')
    rows = []
    for row in sorted(bundle['runs'], key=lambda item: item['batch']):
        sharpness = row.get('sharpness') or {}
        directional = (sharpness.get('directional') or {}).get('0.05') or {}
        rows.append(
            dict(
                batch=row['batch'],
                lr=row['lr'],
                wd=row['wd'],
                validation_nll=row['validation_nll'],
                sharpness_radius_0_05=directional.get('max_loss_increase'),
                run_id=row['id'],
                updates=int(np.ceil(row['ntrain'] / row['batch']) * row['epochs']),
            )
        )
    return dict(
        rows=rows,
        missing_sharpness=[row['batch'] for row in rows if row['sharpness_radius_0_05'] is None],
        best_batch=min(rows, key=lambda row: row['validation_nll'])['batch'],
    )


def wd_section():
    bundle = runs('wd_tuning')
    return dict(comparison=bundle['comparison'], runs=bundle['runs'], baseline=bundle['default_baseline'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_parameters('plots/a6_generalization/p1_parameters.png')
    plot_example(runs('lr_wd_runs')['runs'], 'plots/a6_generalization/p2_lr_wd.png')
    summary = dict(
        parameter_scaling=parameter_section(),
        lr_wd=lr_wd_section(),
        wd_tuning=wd_section(),
        ensemble=ensemble_section(),
        schedules=schedule_section(),
        batch=batch_section(),
        epoch_scaling=[
            dict(
                epochs=row['epochs'],
                lr=row['lr'],
                wd=row['wd'],
                validation_nll=row['validation_nll'],
                run_id=row['id'],
            )
            for row in runs('epoch_scaling')['runs']
        ],
    )
    (OUT / 'supplied_analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'parameter_fit': summary['parameter_scaling']['fit']['all'],
        'monotonic': summary['parameter_scaling']['monotonic_in_parameters'],
        'lr_wd_rmse': summary['lr_wd']['quadratic_rmse'],
        'product_spearman': summary['lr_wd']['product_spearman'],
        'best_wd': summary['lr_wd']['best_positive'],
        'ensemble_fit': summary['ensemble']['fit_k1_to_k5'],
        'batch_best': summary['batch']['best_batch'],
        'missing_sharpness': summary['batch']['missing_sharpness'],
    }, indent=2))


if __name__ == '__main__':
    main()
