#!/usr/bin/env python3
"""Poll Batch 2 to 9/9; append MEASURED finals as jobs finish; then Batch 3 stage a."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

ROOT = "/Users/daxtongutekunst/Desktop/assignments"
STORE = Path(
    "/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files"
)
STATUS = STORE / "internal/mac-gpu-status.md"
APPENDIX = STORE / "docs/a3-experimental-appendix.md"
REPORT = STORE / "docs/a3-assignment-report.md"
BATCH3_PREP = STORE / "internal/batch3-prep.md"
MEDIA_P2 = STORE / "media/p2"
MEDIA_P3 = STORE / "media/p3"
PARTIAL_JSON = MEDIA_P2 / "batch2_partial_finals.json"
PARTIAL_CSV = MEDIA_P2 / "batch2_partial_finals.csv"
PROGRESS = Path(ROOT) / ".local_monitors/batch2_progress.json"
INTERVAL = 120
TIMEOUT = 6 * 3600

os.chdir(ROOT)
sys.path.insert(0, ROOT)

from modal.exception import NotFoundError
from modal_train import _model_volume_path
from modal_utils import user_volume
from train import training_run_name
from experiments.a3_optimization.p2_river_valley import RUNS as P2
from experiments.a3_optimization.p3_edge_of_stability import RUNS as P3

JOBS = list(P2) + list(P3)
LABELS = [
    "a3-b2-p2-wsd0.0",
    "a3-b2-p2-linear",
    "a3-b2-p2-cos",
    "a3-b2-p2-wsd0.2",
    "a3-b2-p2-wsd0.2-dip",
    "a3-b2-p2-invsqrt",
    "a3-b2-p2-sgdr2000",
    "a3-b2-p3-0-sgd",
    "a3-b2-p3-1-adamw",
]


def snapshot():
    rows = []
    done = partial = missing = 0
    for i, cfg in enumerate(JOBS):
        name = training_run_name(cfg)
        path = _model_volume_path(name)
        try:
            have = {PurePosixPath(e.path).name for e in user_volume.listdir(path)}
        except NotFoundError:
            have = set()
        complete = {"config.json", "model.pt"}.issubset(have)
        if complete:
            done += 1
            st = "DONE"
        elif have:
            partial += 1
            st = "PARTIAL"
        else:
            missing += 1
            st = "MISSING"
        rows.append({"st": st, "i": i, "name": name, "have": have})
    return done, partial, missing, rows


def modal_tasks():
    try:
        out = subprocess.check_output(
            ["modal", "app", "list", "--env", "cs312-daxton"], text=True, timeout=60
        )
    except Exception as e:
        return None, str(e)
    tasks = None
    lines = out.splitlines()
    for idx, line in enumerate(lines):
        if "ap-Izg7CMIJgyf29zQokn4SYv" in line:
            chunk = "\n".join(lines[idx : idx + 3])
            nums = re.findall(r"│\s*(\d+)\s*│", chunk)
            if nums:
                tasks = int(nums[0])
            break
    return tasks, out[:500]


def wandb_records(rows):
    import wandb

    api = wandb.Api(timeout=30)
    # Never regress a volume-complete job's Live line to an older/stale W&B snapshot.
    prior = {}
    if PARTIAL_JSON.exists():
        try:
            for r in json.loads(PARTIAL_JSON.read_text()):
                prior[r["i"]] = r
        except Exception:
            prior = {}
    if PROGRESS.exists():
        try:
            for r in json.loads(PROGRESS.read_text()).get("live", []):
                if r.get("state") == "finished" or r.get("volume_done"):
                    prior.setdefault(r["i"], r)
        except Exception:
            pass
    info = []
    for row in rows:
        if row["st"] == "MISSING":
            continue
        path = _model_volume_path(row["name"])
        if "run_state.json" not in row["have"]:
            continue
        obj = json.loads(b"".join(user_volume.read_file(f"{path}/run_state.json")).decode())
        rid = obj.get("wandb_run_id")
        if not rid:
            continue
        run = api.run(f"{obj['wandb_entity']}/{obj['wandb_project']}/{rid}")
        s = dict(run.summary)
        rec = {
            "i": row["i"],
            "label": LABELS[row["i"]],
            "run_name": row["name"],
            "wandb_id": rid,
            "wandb_url": run.url,
            "state": run.state,
            "train_loss": s.get("train_loss"),
            "val_loss": s.get("val_loss"),
            "optimizer_step": s.get("optimizer_step"),
            "progress": s.get("progress"),
            "volume_done": row["st"] == "DONE",
        }
        old = prior.get(row["i"])
        if row["st"] == "DONE" and old and (
            old.get("state") == "finished" or old.get("volume_done")
        ):
            old_step = old.get("optimizer_step") or 0
            new_step = rec.get("optimizer_step") or 0
            if rec.get("state") != "finished" or new_step < old_step:
                rec = {
                    **old,
                    "wandb_url": rec["wandb_url"],
                    "volume_done": True,
                    "state": "finished" if old.get("state") == "finished" else rec["state"],
                }
        info.append(rec)
    return info


def md_table(rows):
    lines = [
        "| # | label | run_name | W&B | state | train_loss | val_loss | optimizer_step |",
        "|---|-------|----------|-----|-------|------------|----------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['i']} | `{r['label']}` | `{r['run_name']}` | [`{r['wandb_id']}`]({r['wandb_url']}) | {r['state']} | "
            f"{r.get('train_loss')} | {r.get('val_loss')} | {r.get('optimizer_step')} |"
        )
    return "\n".join(lines)


def write_partial_artifacts(kept, now):
    MEDIA_P2.mkdir(parents=True, exist_ok=True)
    MEDIA_P3.mkdir(parents=True, exist_ok=True)
    PARTIAL_JSON.write_text(json.dumps(kept, indent=2) + "\n")
    with PARTIAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "i",
                "label",
                "run_name",
                "wandb_id",
                "state",
                "train_loss",
                "val_loss",
                "optimizer_step",
                "progress",
            ],
        )
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    # p3 subset
    (MEDIA_P3 / "batch2_partial_finals.json").write_text(
        json.dumps([r for r in kept if r["i"] >= 7], indent=2) + "\n"
    )

    inc = f"""### Batch 2 incremental finals (**MEASURED**, {len(kept)}/9 volume-complete as of {now})

{md_table(kept) if kept else "_none yet_"}

CSV: `media/p2/batch2_partial_finals.csv`

"""
    app = APPENDIX.read_text()
    if "### Batch 2 incremental finals" in app:
        app2, n = re.subn(
            r"### Batch 2 incremental finals \(\*\*MEASURED\*\*.*?(?=\n### |\n## |\Z)",
            inc,
            app,
            count=1,
            flags=re.S,
        )
        if n == 0:
            raise SystemExit("incremental replace failed")
    else:
        if "## P2 River-Valley" in app:
            app2 = re.sub(r"(## P2 River-Valley\n)", r"\1\n" + inc, app, count=1)
        else:
            app2 = app + "\n" + inc
    APPENDIX.write_text(app2)


def update_status(done, partial, missing, tasks, live, kept_n):
    now = time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    live_txt = "; ".join(
        f"job{x['i']} `{x['wandb_id']}`: step={x.get('optimizer_step')} progress={x.get('progress')} "
        f"train={x.get('train_loss')} val={x.get('val_loss')} state={x.get('state')}"
        for x in live
        if "wandb_id" in x
    ) or "n/a"
    block = f"""## Batch 2 — P2 + P3 — **{"DONE" if done == 9 else "RUNNING"}** (refreshed {now})

Checkout: **`/Users/daxtongutekunst/Desktop/assignments` ONLY**.

| Field | Value |
|-------|--------|
| Command | `uv run python -m experiments.a3_optimization.launch_batch2 --execute --max-parallel 2` |
| Modal app name | `dl_alchemy-20261011-002207` |
| Modal app ID | `ap-Izg7CMIJgyf29zQokn4SYv` |
| App URL | https://modal.com/apps/cs312-f26/cs312-daxton/ap-Izg7CMIJgyf29zQokn4SYv |
| Jobs | **9** (7 P2 + 2 P3 starter core) |
| **Volume complete** (`config.json`+`model.pt`) | **{done} / 9** |
| Partial | **{partial}** |
| Missing | **{missing}** |
| Modal concurrent tasks | **{tasks if tasks is not None else "unknown"}** |
| Durable monitor | incremental wait_and_finish from Desktop/assignments; timer `batch2-monitor` |
| Live/final W&B (**MEASURED**) | {live_txt} |
| Partial finals artifact | `media/p2/batch2_partial_finals.csv` ({kept_n} rows) |
| Final metrics in appendix | {"filled" if done == 9 else f"**PARTIAL** {kept_n}/9 MEASURED — full fill at 9/9"} |
"""
    text = STATUS.read_text()
    new, n = re.subn(
        r"## Batch 2 — P2 \+ P3 — \*\*(?:RUNNING|DONE)\*\*.*?(?=\n\| job \| call_id|\n## Batch 3|\n## Policy|\Z)",
        block + "\n",
        text,
        count=1,
        flags=re.S,
    )
    if n == 0:
        raise SystemExit("status replace failed")
    STATUS.write_text(new)


def sync_incremental(rows, live):
    """Persist MEASURED finals for volume-complete jobs; rewrite incremental appendix table."""
    by_i = {x["i"]: x for x in live}
    kept = []
    newly = []
    prior = {}
    if PARTIAL_JSON.exists():
        for r in json.loads(PARTIAL_JSON.read_text()):
            prior[r["i"]] = r
    for row in rows:
        if row["st"] != "DONE":
            continue
        rec = by_i.get(row["i"])
        if not rec:
            continue
        kept.append(rec)
        old = prior.get(row["i"])
        if old is None or old.get("train_loss") != rec.get("train_loss") or old.get("state") != rec.get("state"):
            newly.append(rec)
    now = time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    write_partial_artifacts(kept, now)
    return kept, newly



def fill_scorecard(finals):
    """Fill Observed on docs/a3-batch2-prediction-scorecard.md from MEASURED val_loss only.
    P3 sharpness medians left blank if not in W&B summary (do not invent).
    """
    SCORECARD = STORE / "docs/a3-batch2-prediction-scorecard.md"
    by = {r["label"]: r for r in finals}
    def L(lab):
        return by[lab]["val_loss"]
    labs4 = ["a3-b2-p2-wsd0.0", "a3-b2-p2-linear", "a3-b2-p2-cos", "a3-b2-p2-wsd0.2"]
    Ls = {lab: L(lab) for lab in labs4}
    Lbest = min(Ls.values())
    R = {lab: Ls[lab] - Lbest for lab in labs4}
    L_dip = L("a3-b2-p2-wsd0.2-dip")
    L_wsd02 = Ls["a3-b2-p2-wsd0.2"]
    R_dip = L_dip - L_wsd02
    L_tuned = min(Ls["a3-b2-p2-cos"], Ls["a3-b2-p2-wsd0.2"])
    dL_inv = L("a3-b2-p2-invsqrt") - L_tuned
    dL_sgdr = L("a3-b2-p2-sgdr2000") - L_tuned
    dL_sgdr_inv = abs(L("a3-b2-p2-sgdr2000") - L("a3-b2-p2-invsqrt"))

    def in_band(x, lo, hi_excl):
        return lo <= x < hi_excl

    def hitmiss(x, lo, hi_excl):
        return "hit" if in_band(x, lo, hi_excl) else "miss"

    # schedule rank
    sched_ok = min(Ls["a3-b2-p2-cos"], Ls["a3-b2-p2-wsd0.2"], Ls["a3-b2-p2-linear"]) < Ls["a3-b2-p2-wsd0.0"]

    rows = {
        "B2-P2-wsd0.0-R": (R["a3-b2-p2-wsd0.0"], 0.02, 0.08, f"R={R['a3-b2-p2-wsd0.0']:.5f}; L={Ls['a3-b2-p2-wsd0.0']:.5f}; Lbest={Lbest:.5f}"),
        "B2-P2-linear-R": (R["a3-b2-p2-linear"], 0.0, 0.02, f"R={R['a3-b2-p2-linear']:.5f}; L={Ls['a3-b2-p2-linear']:.5f}"),
        "B2-P2-cos-R": (R["a3-b2-p2-cos"], 0.0, 0.02, f"R={R['a3-b2-p2-cos']:.5f}; L={Ls['a3-b2-p2-cos']:.5f}"),
        "B2-P2-wsd0.2-R": (R["a3-b2-p2-wsd0.2"], 0.0, 0.02, f"R={R['a3-b2-p2-wsd0.2']:.5f}; L={Ls['a3-b2-p2-wsd0.2']:.5f}"),
        "B2-P2-dip-R": (R_dip, -0.02, 0.02, f"R_final={R_dip:.5f}; L_dip={L_dip:.5f}; L_wsd0.2={L_wsd02:.5f}; mid-dip R TBD from logs"),
        "B2-P2-invsqrt-dL": (dL_inv, 0.01, 0.08, f"dL={dL_inv:.5f}; L_inv={L('a3-b2-p2-invsqrt'):.5f}; L_tuned={L_tuned:.5f}"),
        "B2-P2-sgdr-dL": (dL_sgdr, 0.01, 0.08, f"dL={dL_sgdr:.5f}; |sgdr-inv|={dL_sgdr_inv:.5f}; L_sgdr={L('a3-b2-p2-sgdr2000'):.5f}"),
    }

    text = SCORECARD.read_text()
    # replace partial context block
    partial = f"""**Partial context only (not Observed):** finals through 9/9 now in `media/p2/batch2_finals.csv` (**MEASURED**).

| Job | Label | Final `val_loss` |
|-----|-------|------------------|
| 0 | `a3-b2-p2-wsd0.0` | **{Ls['a3-b2-p2-wsd0.0']:.5f}** |
| 1 | `a3-b2-p2-linear` | **{Ls['a3-b2-p2-linear']:.5f}** |
| 2 | `a3-b2-p2-cos` | **{Ls['a3-b2-p2-cos']:.5f}** |
| 3 | `a3-b2-p2-wsd0.2` | **{Ls['a3-b2-p2-wsd0.2']:.5f}** |
| 4 | `a3-b2-p2-wsd0.2-dip` | **{L_dip:.5f}** |
| 5 | `a3-b2-p2-invsqrt` | **{L('a3-b2-p2-invsqrt'):.5f}** |
| 6 | `a3-b2-p2-sgdr2000` | **{L('a3-b2-p2-sgdr2000'):.5f}** |
| 7 | `a3-b2-p3-0-sgd` | **{L('a3-b2-p3-0-sgd'):.5f}** (val; λ TBD) |
| 8 | `a3-b2-p3-1-adamw` | **{L('a3-b2-p3-1-adamw'):.5f}** (val; ηλpre TBD) |

"""
    text, n = re.subn(
        r"\*\*Partial context only \(not Observed\):\*\*.*?(?=\n---\n)",
        partial,
        text,
        count=1,
        flags=re.S,
    )
    if n == 0:
        print("WARN: partial context replace failed", flush=True)

    def fill_row(text, pid, observed, hm, note):
        # table rows: | `pid` | ... | PREDICTION | Observed | Hit / miss | Notes |
        pat = rf"(\| `{re.escape(pid)}` \|[^|]+\|[^|]+\|) ([^|]*) \| ([^|]*) \| ([^|]*) \|"
        def repl(m):
            return f"{m.group(1)} {observed} | {hm} | {note} |"
        new, n = re.subn(pat, repl, text, count=1)
        if n == 0:
            print(f"WARN: row {pid} not filled", flush=True)
            return text
        return new

    for pid, (val, lo, hi, note) in rows.items():
        observed = f"**MEASURED** {val:.5f}"
        hm = hitmiss(val, lo, hi)
        # special: linear also predicted L_linear < L_wsd0.0
        if pid == "B2-P2-linear-R" and not (Ls["a3-b2-p2-linear"] < Ls["a3-b2-p2-wsd0.0"]):
            hm = "miss"
        if pid == "B2-P2-sgdr-dL":
            # also check |sgdr-inv| band [0,0.03)
            if not in_band(dL_sgdr_inv, 0.0, 0.03):
                hm = "miss" if hm == "hit" else hm + "+|dL_inv|miss"
            note = note + f"; |sgdr-inv| hit={in_band(dL_sgdr_inv, 0.0, 0.03)}"
        text = fill_row(text, pid, observed, hm, note)

    # P3 rows — leave sharpness blank unless summary has usable keys
    for pid in ("B2-P3-sgd-lam", "B2-P3-adamw-pre"):
        text = fill_row(
            text,
            pid,
            "**PENDING** sharpness median (not in val_loss summary)",
            "—",
            "Do not invent λ / ηλpre; pull from W&B sharpness logs later",
        )

    # alias P2-sched-rank
    text = fill_row(
        text,
        "P2-sched-rank",
        f"**MEASURED** min(cos,wsd0.2,linear)={min(Ls['a3-b2-p2-cos'], Ls['a3-b2-p2-wsd0.2'], Ls['a3-b2-p2-linear']):.5f} < L_wsd0.0={Ls['a3-b2-p2-wsd0.0']:.5f}",
        "hit" if sched_ok else "miss",
        "from finals val_loss",
    )
    # other aliases that need EMA/mid-dip — leave pending note
    for pid, note in (
        ("P2-ema-gap-frac", "EMA vals not in summary — PENDING"),
        ("P2-dip-final-R", f"same as B2-P2-dip-R final R={R_dip:.5f}"),
        ("P2-horizon-dL", f"invsqrt dL={dL_inv:.5f}; sgdr dL={dL_sgdr:.5f}"),
        ("P3-clean-edge", "PENDING sharpness"),
        ("P3-mb-edge", "PENDING sharpness"),
    ):
        hm = hitmiss(R_dip, -0.02, 0.02) if pid == "P2-dip-final-R" else (
            ("hit" if in_band(dL_inv, 0.01, 0.08) and in_band(dL_sgdr, 0.01, 0.08) else "miss")
            if pid == "P2-horizon-dL"
            else "—"
        )
        obs = (
            f"**MEASURED** R={R_dip:.5f}"
            if pid == "P2-dip-final-R"
            else (
                f"**MEASURED** inv={dL_inv:.5f} sgdr={dL_sgdr:.5f}"
                if pid == "P2-horizon-dL"
                else "**PENDING**"
            )
        )
        text = fill_row(text, pid, obs, hm, note)

    SCORECARD.write_text(text)
    print("scorecard_updated", flush=True)

def pull_finals_and_appendix():
    import wandb

    api = wandb.Api(timeout=60)
    finals = []
    for i, cfg in enumerate(JOBS):
        name = training_run_name(cfg)
        path = _model_volume_path(name)
        obj = json.loads(b"".join(user_volume.read_file(f"{path}/run_state.json")).decode())
        rid = obj["wandb_run_id"]
        run = api.run(f"{obj['wandb_entity']}/{obj['wandb_project']}/{rid}")
        s = dict(run.summary)
        finals.append(
            {
                "i": i,
                "label": LABELS[i],
                "run_name": name,
                "wandb_id": rid,
                "wandb_url": run.url,
                "state": run.state,
                "train_loss": s.get("train_loss"),
                "val_loss": s.get("val_loss"),
                "optimizer_step": s.get("optimizer_step"),
                "progress": s.get("progress"),
            }
        )
        print("FINAL", finals[-1], flush=True)

    MEDIA_P2.mkdir(parents=True, exist_ok=True)
    MEDIA_P3.mkdir(parents=True, exist_ok=True)
    (MEDIA_P2 / "batch2_final_metrics.json").write_text(json.dumps(finals, indent=2) + "\n")
    (MEDIA_P3 / "batch2_final_metrics.json").write_text(
        json.dumps([f for f in finals if f["i"] >= 7], indent=2) + "\n"
    )
    with (MEDIA_P2 / "batch2_finals.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "i",
                "label",
                "run_name",
                "wandb_id",
                "state",
                "train_loss",
                "val_loss",
                "optimizer_step",
                "progress",
            ],
        )
        w.writeheader()
        for r in finals:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    (MEDIA_P2 / "batch2_final_metrics.csv").write_text((MEDIA_P2 / "batch2_finals.csv").read_text())

    p2_rows = [f for f in finals if f["i"] < 7]
    p3_rows = [f for f in finals if f["i"] >= 7]
    app = APPENDIX.read_text()
    p2_section = f"""## P2 River-Valley

### Config

| Key | Value |
|-----|-------|
| schedules | `wsd0.0`, `linear`, `cos`, `wsd0.2` |
| `ema_decays` | `(0.99, 0.999)` on schedule arms |
| `lr_dip` | on one `wsd0.2` arm (`dip0.4-0.45-0.5x0.1`) |
| horizon-free | `invsqrt`, `sgdr2000` |
| Batch | Batch 2 · Modal `ap-Izg7CMIJgyf29zQokn4SYv` |

### Runs (**MEASURED** finals from W&B summary)

{md_table(p2_rows)}

Artifacts: `media/p2/batch2_finals.csv`, `media/p2/batch2_final_metrics.json`

"""
    p3_section = f"""## P3 Edge of Stability

### Config

| Key | Value |
|-----|-------|
| Starter core | full-batch SGD + minibatch AdamW sharpness (2 jobs in Batch 2) |
| Batch | Batch 2 · Modal `ap-Izg7CMIJgyf29zQokn4SYv` |

### Runs (**MEASURED** finals from W&B summary)

{md_table(p3_rows)}

Artifacts: `media/p3/batch2_final_metrics.json`

"""
    app2, n1 = re.subn(
        r"## P2 River-Valley\n.*?(?=\n## P3 Edge of Stability\n)",
        p2_section,
        app,
        count=1,
        flags=re.S,
    )
    if n1 == 0:
        raise SystemExit("P2 section not found")
    app3, n2 = re.subn(
        r"## P3 Edge of Stability\n.*?(?=\n## P4 Inside the Hessian\n)",
        p3_section,
        app2,
        count=1,
        flags=re.S,
    )
    if n2 == 0:
        raise SystemExit("P3 section not found")
    APPENDIX.write_text(app3)

    # brief report note (MEASURED only)
    if REPORT.exists():
        rep = REPORT.read_text()
        ts = time.strftime("%Y-%m-%d %H:%M %Z")
        p2_vals = [r["val_loss"] for r in p2_rows if r.get("val_loss") is not None]
        p3_bits = ", ".join(f"{r['label']} val={r['val_loss']}" for r in p3_rows)
        note = (
            f"\n\n### Batch 2 P2/P3 finals (**MEASURED**, {ts})\n\n"
            "See `docs/a3-experimental-appendix.md` P2/P3 tables and `media/p2/batch2_final_metrics.json`.\n"
            f"P2 val_loss range (MEASURED): {min(p2_vals)} … {max(p2_vals)}; "
            f"P3 jobs: {p3_bits}.\n"
        )
        if "### Batch 2 P2/P3 finals" in rep:
            rep2, _ = re.subn(
                r"### Batch 2 P2/P3 finals \(\*\*MEASURED\*\*.*?(?=\n### |\n## |\Z)",
                note.lstrip("\n"),
                rep,
                count=1,
                flags=re.S,
            )
        else:
            rep2 = rep.rstrip() + note
        REPORT.write_text(rep2)
    print("appendix_and_report_updated", flush=True)
    fill_scorecard(finals)
    return finals


def launch_batch3():
    dry = subprocess.run(
        ["uv", "run", "python", "-m", "experiments.a3_optimization.launch_batch3", "--dry-run", "--stage", "a"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    print("DRY_STDOUT\n", dry.stdout[-4000:], flush=True)
    print("DRY_STDERR\n", dry.stderr[-2000:], flush=True)
    exe = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "experiments.a3_optimization.launch_batch3",
            "--execute",
            "--stage",
            "a",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    print("EXEC_STDOUT\n", exe.stdout[-6000:], flush=True)
    print("EXEC_STDERR\n", exe.stderr[-2000:], flush=True)
    (STORE / "internal/batch3-launch.log").write_text(exe.stdout + "\n---STDERR---\n" + exe.stderr)
    prep = BATCH3_PREP.read_text()
    prep2 = prep.replace(
        "| Dry-run executed | Not yet (waiting Batch 2 GPU drain) |\n| Stage `a` submitted | Not yet |\n| Measured Hessian tables | None — do not invent |",
        f"| Dry-run executed | exit={dry.returncode} |\n| Stage `a` submitted | exit={exe.returncode}; log `internal/batch3-launch.log` |\n| Measured Hessian tables | None until fetch — do not invent |",
    )
    m = re.search(r"https://modal.com/apps/[^\s]+/(ap-[A-Za-z0-9]+)", exe.stdout)
    app_id = m.group(1) if m else "see log"
    m2 = re.search(r"app='([^']+)'", exe.stdout)
    app_name = m2.group(1) if m2 else ""
    if "## Launch record" not in prep2:
        prep2 += (
            f"\n\n## Launch record\n\n- Stage `a` submit exit: `{exe.returncode}`\n"
            f"- Modal app name: `{app_name}`\n- Modal app ID/URL fragment: `{app_id}`\n"
            f"- Time: {time.strftime('%Y-%m-%d %H:%M %Z')}\n"
        )
    BATCH3_PREP.write_text(prep2)
    st = STATUS.read_text()
    if "## Batch 3" not in st:
        st += f"""

## Batch 3 — P4 Hessian — **SUBMITTED stage a** (after Batch 2 9/9)

| Field | Value |
|-------|--------|
| Checkout | `/Users/daxtongutekunst/Desktop/assignments` |
| Command | `uv run python -m experiments.a3_optimization.launch_batch3 --execute --stage a` |
| Submit exit | `{exe.returncode}` |
| Modal app | `{app_name}` / `{app_id}` |
| Prep | [`batch3-prep.md`](batch3-prep.md) |
| Fetch | later — `launch_batch3 --stage fetch` when measure jobs finish |
"""
        STATUS.write_text(st)
    return exe.returncode


def main():
    print("WAIT_START cwd=", os.getcwd(), flush=True)
    start = time.time()
    while True:
        done, partial, missing, rows = snapshot()
        tasks, _ = modal_tasks()
        live = wandb_records(rows)
        kept, newly = sync_incremental(rows, live)
        PROGRESS.write_text(
            json.dumps(
                {
                    "done": done,
                    "partial": partial,
                    "missing": missing,
                    "tasks": tasks,
                    "live": live,
                    "kept": len(kept),
                    "newly": [x["i"] for x in newly],
                    "ts": time.time(),
                },
                indent=2,
            )
        )
        update_status(done, partial, missing, tasks, live, len(kept))
        print(
            f"[{int(time.time()-start)}s] done={done}/9 partial={partial} missing={missing} tasks={tasks} kept={len(kept)} newly={ [x['i'] for x in newly] }",
            flush=True,
        )
        for x in live:
            print(
                f"  W&B job{x['i']} {x['wandb_id']} step={x.get('optimizer_step')} progress={x.get('progress')} state={x.get('state')}",
                flush=True,
            )
        if done == 9:
            print("ALL_BATCH2_DONE", flush=True)
            finals = pull_finals_and_appendix()
            update_status(9, 0, 0, tasks, finals, 9)
            rc = launch_batch3()
            print("BATCH3_STAGE_A_RC", rc, flush=True)
            print("READY_UNSUBSCRIBE_TIMER", flush=True)
            return 0 if rc == 0 else 3
        if time.time() - start > TIMEOUT:
            print("TIMEOUT", flush=True)
            return 2
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
