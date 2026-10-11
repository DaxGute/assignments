#!/usr/bin/env python3
"""Wait for 9375; run Mac steps from internal/p4-on-9375-runbook.md §§1–3 + §6 via pipeline."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/daxtongutekunst/Desktop/assignments")
os.chdir(ROOT)
assert os.getcwd() == str(ROOT)
if "assignments-a3-batch1" in os.getcwd():
    raise SystemExit("FORBIDDEN")
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT)
os.environ.setdefault("MODAL_ENVIRONMENT", "cs312-daxton")

from experiments.a3_optimization.p4_inside_the_hessian import STAGES
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, _filename, _volume_results, fetch

STORE = Path(
    "/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files"
)
MEDIA = STORE / "media/p4/stage-a"
OUT = Path("/tmp/a3-batch3-final-9375.out")
PY = str(ROOT / ".venv/bin/python")
a = STAGES["a"]
NAME = "a-adamw-9375"


def ensure(name: str, log: str) -> None:
    pat = f"/Desktop/assignments/.local_monitors/{name}.py"
    if subprocess.call(["pgrep", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        if name == "batch3_stage_pipeline":
            (ROOT / ".local_monitors/batch3_stage_pipeline.lock").unlink(missing_ok=True)
        subprocess.Popen(
            [PY, "-u", str(ROOT / ".local_monitors" / f"{name}.py")],
            cwd=str(ROOT),
            stdout=open(ROOT / ".local_monitors" / log, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"RESPAWN {name}", flush=True)


def volume():
    files = _volume_results()
    done = [j["name"] for j in a if _filename(j) in files]
    miss = [j["name"] for j in a if _filename(j) not in files]
    return done, miss


print("FINAL9375_RUNBOOK_START cwd=", os.getcwd(), flush=True)
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    ensure("longpoll_n4", "batch3_longpoll.log")
    ensure("batch3_stage_pipeline", "batch3_stage_pipeline.log")
    try:
        done, miss = volume()
    except Exception as e:
        print("vol_err", e, flush=True)
        time.sleep(25)
        continue
    if len(done) < 13 and miss:
        time.sleep(25)
        continue

    print("HIT_13_DETECTED", flush=True)

    # §1 copy JSON (Mac → store) — fetch then cp; no invent
    src = RESULTS_DIR / f"{NAME}.json"
    for _ in range(12):
        if src.exists():
            break
        try:
            fetch([NAME])
        except Exception as e:
            print("fetch_err", e, flush=True)
        time.sleep(1)
    if not src.exists():
        print("STOP_NO_9375_FILE", flush=True)
        OUT.write_text(json.dumps({"reason": "blocker_missing_9375", "forbidden_used": False}) + "\n")
        raise SystemExit(3)

    MEDIA.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, MEDIA / f"{NAME}.json")
    shutil.copy2(src, STORE / "media/p4" / f"{NAME}.json")
    full = json.loads(src.read_text())
    lm = full.get("lambda_max")
    print(f"COPIED {NAME} lambda_max={lm}", flush=True)

    # refresh progress
    subprocess.run([PY, "-u", str(ROOT / ".local_monitors/batch3_refresh_once.py")], cwd=str(ROOT), check=False)

    # §2 mid-N (absolute store path — Mac cannot cd /cursor/stores/…)
    mid = STORE / "internal/p4-mid-n-refresh.sh"
    if mid.exists():
        # patch: run scripts with STORE as cwd
        env = {**os.environ, "STORE": str(STORE)}
        subprocess.run(["bash", str(mid)], cwd=str(STORE), env=env, check=False)
    else:
        for script in ("p4-fill-appendix-from-media.py", "p4-update-lambda-plot.py", "p4-score-from-media.py"):
            p = STORE / "internal" / script
            if p.exists():
                subprocess.run(["python3", str(p)], cwd=str(STORE), check=False)

    # §3 dry-run then gated --apply
    score = STORE / "internal/p4-score-from-media.py"
    apply_rc = None
    apply_tail = ""
    if score.exists():
        subprocess.run(["python3", str(score)], cwd=str(STORE), check=False)
        proc = subprocess.run(
            ["python3", str(score), "--apply"],
            cwd=str(STORE),
            capture_output=True,
            text=True,
            timeout=300,
        )
        apply_rc = proc.returncode
        apply_tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-2000:]
        (STORE / "internal/p4-score-apply.log").write_text((proc.stdout or "") + "\n---\n" + (proc.stderr or ""))
        print(f"P4_APPLY rc={apply_rc}", flush=True)

    # §6 via durable pipeline: fetch → b-rescale → fetch → c → fetch
    ensure("batch3_stage_pipeline", "batch3_stage_pipeline.log")
    pipe_deadline = time.time() + 120 * 60
    pipeline_done = False
    while time.time() < pipe_deadline:
        log = (ROOT / ".local_monitors/batch3_stage_pipeline.log").read_text(errors="ignore")
        tail = log[-12000:]
        if "PIPELINE COMPLETE" in tail:
            pipeline_done = True
            print("PIPELINE_COMPLETE_SEEN", flush=True)
            break
        if "EXECUTE b-rescale" in log or "EXECUTE c" in log or "b-rescale" in tail:
            print("PIPELINE_ADVANCING", flush=True)
        ensure("batch3_stage_pipeline", "batch3_stage_pipeline.log")
        time.sleep(20)

    payload = {
        "reason": "complete",
        "n": 13,
        "lambda_max_9375": lm,
        "pipeline_done": pipeline_done,
        "p4_apply_rc": apply_rc,
        "p4_apply_tail": apply_tail,
        "runbook": "internal/p4-on-9375-runbook.md §§1–3 + §6",
        "checkout": str(ROOT),
        "forbidden_used": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    (STORE / "internal/p4-apply-ready-signal.md").write_text(
        "---\n"
        "cursor:\n"
        '  subagentId: "bc-8a994472-d1f4-5aa9-a973-7602cdc5af8b"\n'
        "---\n\n"
        "# P4 --apply / runbook signal after 9375\n\n"
        f"- **λ_max a-adamw-9375:** {lm}\n"
        f"- **p4_apply_rc:** {apply_rc} (2=REFUSED some IDs; 0=ready in principle)\n"
        f"- **pipeline_done:** {pipeline_done}\n"
        "- **Mac checkout:** `/Users/daxtongutekunst/Desktop/assignments` ONLY\n"
        "- **runbook:** `internal/p4-on-9375-runbook.md`\n"
    )
    (STORE / "internal/batch3-n13-complete-report.md").write_text(
        "---\n"
        "cursor:\n"
        '  subagentId: "bc-8a994472-d1f4-5aa9-a973-7602cdc5af8b"\n'
        "---\n\n"
        f"# Batch 3 13/13 — λ_max 9375 = {lm}\n\n"
        f"- pipeline_done: {pipeline_done}\n"
        f"- p4_apply_rc: {apply_rc}\n"
        "- path: Desktop/assignments ONLY\n"
    )
    print("FINAL9375_EXIT", json.dumps({k: payload[k] for k in payload if k != "p4_apply_tail"}), flush=True)
    print("REPORT_LAMBDA", lm, flush=True)
    raise SystemExit(0)

OUT.write_text(json.dumps({"reason": "timeout", "n": 12}) + "\n")
print("FINAL9375_TIMEOUT", flush=True)
raise SystemExit(2)
