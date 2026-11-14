"""Evidence-based verifier for A3 experiments and deliverables.

Current-state audit (permits explicitly reported running work):

    uv run --with pypdf python -m experiments.a3_optimization.verify_a3

Final gate (fails until every experiment and source-PDF requirement resolves):

    uv run --with pypdf python -m experiments.a3_optimization.verify_a3 \
        --require-complete
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
A3 = Path(__file__).resolve().parent
RESULTS = A3 / "results"
DELIVERABLES = A3 / "deliverables"


def _load(path: Path):
    return json.loads(path.read_text())


def _record(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def _pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf or run with `uv run --with pypdf`") from exc
    return len(PdfReader(path).pages)


def verify(require_complete: bool) -> tuple[list[dict], list[str]]:
    checks: list[dict] = []
    blockers: list[str] = []

    p1 = _load(RESULTS / "a3-mode-connectivity.json")
    _record(checks, "P1 grid", len(p1) == 30, f"{len(p1)}/30 cells")
    p1_analysis = _load(
        RESULTS / "a3-mode-connectivity-analysis" / "barrier-summary.json"
    )
    _record(
        checks,
        "P1 summary",
        p1_analysis["cells"] == 30
        and p1_analysis["averaging_works_count"] == 26
        and len(p1_analysis["failures"]) == 4,
        "30 cells, 26 connected, 4 failures",
    )

    p1_controls_path = RESULTS / "a3-mode-connectivity-controls.json"
    p1_controls_done = p1_controls_path.exists()
    _record(
        checks,
        "P1 controls",
        p1_controls_done or not require_complete,
        "measured" if p1_controls_done else "running/queued",
    )
    if not p1_controls_done:
        blockers.append("P1(d) control measurements")

    p2 = _load(RESULTS / "a3-p2-analysis.json")
    p2_ok = (
        p2["counts"] == {"expected": 26, "measured": 26}
        and not p2["incomplete"]
        and p2["derived"]["ema"]["gap_closed_fraction"] == 0.7392498292506038
        and p2["derived"]["dip"]["step_4230_regret"] == -0.11722207069396973
        and len(p2["derived"]["schedule_lr_tuning"]) == 4
        and len(p2["derived"]["dip_sweep"]) == 4
        and len(p2["derived"]["horizon_budgets"]) == 2
    )
    _record(
        checks,
        "P2 runs and metrics",
        p2_ok or not require_complete,
        "26/26 locally materialized" if p2_ok else "required sweeps running",
    )
    if not p2_ok:
        blockers.append("P2 LR tuning, dip, and stopping-budget sweeps")

    p3 = _load(RESULTS / "a3-p3-analysis.json")
    p3_done = (
        p3["counts"]["expected"] == 22
        and p3["counts"]["measured"] == 22
    )
    _record(
        checks,
        "P3 configurations",
        p3_done or not require_complete,
        f"{p3['counts']['measured']}/{p3['counts']['expected']} measured",
    )
    if not p3_done:
        blockers.append(
            f"P3(c) configurations ({p3['counts']['incomplete']} incomplete)"
        )

    p4_dir = RESULTS / "a3-inside-the-hessian"
    p4_steps = (0, 3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375)
    stage_a = [p4_dir / f"a-adamw-{step}.json" for step in p4_steps]
    core_c = [p4_dir / f"c-{arm}.jsonl" for arm in ("full", "top", "removed")]
    _record(
        checks,
        "P4 core",
        all(path.exists() for path in stage_a + core_c),
        "13/13 stage-a and 3/3 k=10 subspace files",
    )
    p4_followups = [
        p4_dir / "b-rescale-attn-v4-o0.25.json",
        p4_dir / "b-rescale-attn-v0.25-o4.json",
        p4_dir / "c-k30-top.jsonl",
        p4_dir / "c-k30-removed.jsonl",
    ]
    p4_followups_done = all(path.exists() for path in p4_followups)
    _record(
        checks,
        "P4 targeted follow-ups",
        p4_followups_done or not require_complete,
        "measured" if p4_followups_done else "running/queued",
    )
    if not p4_followups_done:
        blockers.append("P4 exact-gauge and k=30 measurements")

    p4_training = [
        p4_dir / f"b-final-model-d8-lr0.003-tok614M-p4-gauge-v{factor}.json"
        for factor in ("0.25", "0.5", "2", "4")
    ] + [
        p4_dir / "b-final-model-d8-lr0.003-tok614M-p4-persistent-gauge-v0.25.json",
        p4_dir / "b-final-model-d8-lr0.003-tok614M-p4-frozen-norm-gauge-v0.25.json",
        p4_dir / "b-final-model-d8-lr0.003-tok614M-p4-persistent-head-gauge-v0.25.json",
    ]
    p4_training_analysis = _load(p4_dir / "a3-p4-training-analysis.json")
    p4_training_done = (
        all(path.exists() for path in p4_training)
        and len(p4_training_analysis.get("rows", ())) == 7
    )
    _record(
        checks,
        "P4 training-induced sharpening",
        p4_training_done or not require_complete,
        "7/7 measured; 2× training-time sharpness not achieved"
        if p4_training_done
        else "training/measurement pending",
    )
    if not p4_training_done:
        blockers.append("P4(b) training-induced sharpening measurements")

    p5 = _load(RESULTS / "a3-p5-cooldown-barrier.json")
    p5_ok = (
        p5.get("L_R_endpoint_mean") == 3.7496910095214844
        and p5.get("L_C_endpoint_mean") == 3.5674904584884644
        and p5.get("dL_endpoints") == 0.18220055103302002
        and len(p5.get("ledger_ids", ())) == 7
    )
    _record(checks, "P5 endpoint join", p5_ok, "7/7 ledger outcomes measurable")

    ledger_path = DELIVERABLES / "a3-prediction-ledger.csv"
    rows = list(csv.DictReader(ledger_path.open()))
    ids = [row["id"] for row in rows]
    allowed_kinds = {"measured", "predicted", "extrapolated", "fitted"}
    allowed_statuses = {"resolved", "partial", "preregistered"}
    required_text = ("id", "theme", "intervention",
                     "predicted_interval_or_ranking", "mechanism_cite")
    ledger_ok = (
        len(rows) >= 61
        and len(ids) == len(set(ids))
        and all(all(row[field].strip() for field in required_text) for row in rows)
        and all(row["kind"] in allowed_kinds for row in rows)
        and all(row["status"] in allowed_statuses for row in rows)
        and all(
            row["observed"].strip()
            for row in rows
            if row["kind"] == "measured" or row["status"] in {"resolved", "partial"}
        )
    )
    _record(
        checks,
        "Prediction ledger",
        ledger_ok,
        f"{len(rows)} rows, unique IDs and valid evidence states",
    )

    active_sources = (
        "a3-assignment-report.md",
        "a3-experimental-appendix.md",
        "a3-predictive-intuition-guide.md",
        "a3-quantitative-prediction-reference.md",
        "a3-practice.md",
        "a3-prediction-ledger.md",
        "a3-prediction-ledger.csv",
    )
    sources_exist = all((DELIVERABLES / name).exists() for name in active_sources)
    _record(checks, "Six logical deliverables", sources_exist, "all active sources exist")

    guide = (DELIVERABLES / "a3-predictive-intuition-guide.md").read_text()
    counterfactual_sections = guide.split("**E. Counterfactual levers**")[1:]
    counterfactual_counts = [
        sum(line.startswith("- ") for line in section.split(
            "**F. Failure modes / confounds**", 1
        )[0].splitlines())
        for section in counterfactual_sections
    ]
    guide_counterfactuals_ok = (
        len(counterfactual_sections) == 16
        and all(count >= 3 for count in counterfactual_counts)
    )
    _record(
        checks,
        "Guide counterfactual coverage",
        guide_counterfactuals_ok,
        f"{len(counterfactual_sections)} subproblems; minima={min(counterfactual_counts, default=0)}",
    )

    quiz = (DELIVERABLES / "a3-practice.md").read_text()
    quiz_questions = [
        line for line in quiz.splitlines()
        if line.startswith("### Q") and ". " in line
    ]
    quiz_ok = len(quiz_questions) == 30 and "| Q30 | a |" in quiz and "| Q30 | b |" in quiz
    _record(
        checks,
        "Counterfactual practice quiz",
        quiz_ok,
        f"{len(quiz_questions)} questions with answer-sheet/key coverage",
    )

    appendix = (DELIVERABLES / "a3-experimental-appendix.md").read_text()
    reproducibility_markers = (
        "model_seed=42",
        "data_seed=42",
        "614,400,000 tokens",
        "beta_1",
        "beta_2",
        "Checkpoint policy",
        "FP32 with TF32 disabled",
        "W&B",
    )
    missing_repro = [
        marker for marker in reproducibility_markers if marker not in appendix
    ]
    _record(
        checks,
        "Reproducibility metadata",
        not missing_repro,
        "global seeds/config/checkpoint/provenance recorded"
        if not missing_repro else f"missing {missing_repro}",
    )

    required_p1_plots = (
        RESULTS / "a3-mode-connectivity-analysis" / "plots"
        / "endpoint_ensemble_heatmaps.png",
        RESULTS / "a3-mode-connectivity-analysis" / "plots"
        / "interpolation_curves.png",
    )
    _record(
        checks,
        "P1 assignment figures",
        all(path.exists() for path in required_p1_plots),
        "endpoint/ensemble grids and barrier diagnostics",
    )

    required_p2_plots = (
        RESULTS / "a3-p2-analysis-plots" / "p2_schedule_diagnostics.png",
        RESULTS / "a3-p2-analysis-plots" / "p2_schedule_ema.png",
        RESULTS / "a3-p2-analysis-plots" / "p2_dip_rewarm.png",
        RESULTS / "a3-p2-analysis-plots" / "p2_lr_tuning.png",
        RESULTS / "a3-p2-analysis-plots" / "p2_dip_sweep.png",
        RESULTS / "a3-p2-analysis-plots" / "p2_horizon_budgets.png",
    )
    _record(
        checks,
        "P2 assignment figures",
        all(path.exists() for path in required_p2_plots),
        "LR/train/validation, EMA, and dip diagnostics",
    )

    p3_plot_dir = RESULTS / "a3-p3-analysis-plots"
    required_p3_plots = (
        "p3_full_batch_trajectories.png",
        "p3_adamw_preconditioned_trajectory.png",
        "p3_full_lr_scaling.png",
        "p3_full_sample_scaling.png",
        "p3_minibatch_scaling.png",
    )
    missing_p3_plots = [
        name for name in required_p3_plots if not (p3_plot_dir / name).exists()
    ]
    _record(
        checks,
        "P3 diagnostic figures",
        not missing_p3_plots,
        "5/5 measured/fitted diagnostics"
        if not missing_p3_plots else f"missing {missing_p3_plots}",
    )

    p4_plot_dir = RESULTS / "a3-inside-the-hessian" / "plots"
    required_p4_plots = (
        "p4_lambda_max_early.png",
        "p4_family_lambda_early.png",
        "p4_spectrum_density.png",
        "p4_layer_curvature_depth.png",
        "p4_block_energy.png",
        "p4_head_rescale_response.png",
        "p4_training_sharpness.png",
        "p4_subspace_continue.png",
        "p4_subspace_sharpness.png",
        "p4_subspace_k30.png",
    )
    missing_p4_plots = [
        name for name in required_p4_plots if not (p4_plot_dir / name).exists()
    ]
    _record(
        checks,
        "P4 diagnostic figures",
        not missing_p4_plots,
        "10/10 measured diagnostics"
        if not missing_p4_plots else f"missing {missing_p4_plots}",
    )

    p5_plot = RESULTS / "a3-p5-analysis-plots" / "p5_cooldown_barrier.png"
    _record(
        checks,
        "P5 diagnostic figure",
        p5_plot.exists(),
        "measured interpolation/barrier/endpoint diagnostic",
    )

    stale = {}
    for name in active_sources[:-2]:
        text = (DELIVERABLES / name).read_text()
        hits = [token for token in ("`media/", "`internal/", "`docs/") if token in text]
        if hits:
            stale[name] = hits
    _record(
        checks,
        "Active provenance paths",
        not stale,
        "no stale media/internal/docs references" if not stale else repr(stale),
    )

    report_pdf = DELIVERABLES / "a3-assignment-report.pdf"
    study_pdf = DELIVERABLES / "a3-study-pack.pdf"
    report_pages = _pdf_pages(report_pdf)
    study_pages = _pdf_pages(study_pdf)
    _record(
        checks,
        "Rendered PDFs",
        report_pages <= 20 and study_pages > 0,
        f"report={report_pages} pages, study pack={study_pages} pages",
    )

    source_pdfs = (
        tuple(ROOT.rglob("a3_optimization2.pdf")),
        tuple(ROOT.rglob("hyperparameter_scaling_prediction_quiz.pdf")),
    )
    source_pdfs_found = all(source_pdfs)
    _record(
        checks,
        "Authoritative source PDFs",
        source_pdfs_found or not require_complete,
        "found" if source_pdfs_found else "missing from workspace",
    )
    if not source_pdfs_found:
        blockers.append(
            "a3_optimization2.pdf and hyperparameter_scaling_prediction_quiz.pdf"
        )

    return checks, blockers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    checks, blockers = verify(args.require_complete)
    print(json.dumps({"checks": checks, "blockers": blockers}, indent=2))
    return 0 if all(check["passed"] for check in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
