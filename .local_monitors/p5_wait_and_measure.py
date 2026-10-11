#!/usr/bin/env python3
"""Wait for P5 continue 4/4, measure, sync to Project store, score+apply."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = "/Users/daxtongutekunst/Desktop/assignments"
STORE = Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from modal_train import _completed_model_exists, _model_volume_path
from experiments.a3_optimization.p5_cooldown_barrier import CONTINUE_RUNS, RESULTS_PATH
from train import training_run_name

APP_ID = "ap-I57w3JnswkRdSl3qU1hMQV"
LOG_PATH = Path(ROOT) / ".local_monitors" / "p5_wait_and_measure.log"
STATUS_PATH = Path(ROOT) / ".local_monitors" / "p5_wait_status.json"
DONE_FLAG = Path(ROOT) / ".local_monitors" / "p5_pipeline_done.json"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def count_done():
    rows = []
    done = 0
    for cfg in CONTINUE_RUNS:
        name = training_run_name(cfg)
        path = _model_volume_path(name)
        exists, _ = _completed_model_exists(cfg)
        try:
            have = sorted(PurePosixPath(e.path).name for e in __import__("modal_utils", fromlist=["user_volume"]).user_volume.listdir(path))
        except Exception as e:
            have = [f"ERR:{e}"]
        if exists:
            done += 1
        rows.append({"name": name, "exists": exists, "have": have, "path": path})
    return done, rows


def run(cmd, cwd=None):
    log(f"$ {' '.join(cmd)} (cwd={cwd or ROOT})")
    proc = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    log(f"rc={proc.returncode}")
    if proc.stdout:
        log("STDOUT:\n" + proc.stdout[-8000:])
    if proc.stderr:
        log("STDERR:\n" + proc.stderr[-4000:])
    return proc.returncode


def main() -> int:
    log(f"cwd={os.getcwd()} app={APP_ID}")
    if os.getcwd() != ROOT:
        log(f"FATAL wrong cwd {os.getcwd()}")
        return 2
    while True:
        done, rows = count_done()
        STATUS_PATH.write_text(json.dumps({"done": done, "total": 4, "rows": rows, "app": APP_ID}, indent=2))
        log(f"continues {done}/4")
        if done >= 4:
            break
        time.sleep(30)

    log("4/4 DONE — measure")
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 50:
        log(f"results already present {RESULTS_PATH}; re-running measure anyway if incomplete")
    rc = run(["uv", "run", "python", "-m", "experiments.a3_optimization.p5_cooldown_barrier", "measure"])
    if rc != 0:
        return rc
    if not RESULTS_PATH.exists():
        log("FATAL missing results json")
        return 3

    media = STORE / "media" / "p5"
    media.mkdir(parents=True, exist_ok=True)
    dest = media / "a3-p5-cooldown-barrier.json"
    dest.write_bytes(RESULTS_PATH.read_bytes())
    log(f"synced {dest} bytes={dest.stat().st_size}")

    data = json.loads(RESULTS_PATH.read_text())
    compact = {
        "B_R": data.get("B_R"),
        "B_C": data.get("B_cool", data.get("B_C")),
        "B_K": data.get("B_const", data.get("B_K")),
        "G_C": data.get("G_cool", data.get("G_C")),
        "dB": data.get("dB"),
        "dL": data.get("dL_endpoints", data.get("dL")),
        "emergent": data.get("emergent"),
        "app_id": APP_ID,
    }
    (media / "p5-metrics.json").write_text(json.dumps(compact, indent=2) + "\n")
    log(f"compact metrics {compact}")

    # score dry-run then apply
    sh = STORE / "internal" / "p5-score-and-apply.sh"
    rc1 = run(["bash", str(sh)], cwd=str(STORE))
    if rc1 not in (0, 2):
        log(f"score dry-run unexpected {rc1}")
    rc2 = run(["bash", str(sh), "--apply"], cwd=str(STORE))
    DONE_FLAG.write_text(json.dumps({
        "app_id": APP_ID,
        "metrics": compact,
        "score_dryrun_rc": rc1,
        "score_apply_rc": rc2,
        "results": str(RESULTS_PATH),
        "store_json": str(dest),
        "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    log(f"pipeline done apply_rc={rc2}")
    return 0 if rc2 == 0 else rc2


if __name__ == "__main__":
    raise SystemExit(main())
