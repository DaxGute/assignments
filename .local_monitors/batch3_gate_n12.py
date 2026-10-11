#!/usr/bin/env python3
"""Wait until stage-a N>11 from Desktop/assignments only; copy + mid-N + report."""
from __future__ import annotations

import json
import os
import re
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
PROG = STORE / "media/p4/stage_a_progress.json"
OUT = Path("/tmp/a3-batch3-gate-n12.out")
PY = str(ROOT / ".venv/bin/python")
BASELINE = 12
PREV = {
    "a-adamw-0",
    "a-adamw-3",
    "a-adamw-15",
    "a-adamw-61",
    "a-adamw-153",
    "a-adamw-305",
    "a-adamw-763",
    "a-adamw-1526",
    "a-adamw-3052",
    "a-adamw-4578",
    "a-adamw-6104",
}
a = STAGES["a"]


def ensure(name: str, log: str) -> None:
    pat = f"/Desktop/assignments/.local_monitors/{name}.py"
    if subprocess.call(["pgrep", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        if name == "batch3_stage_pipeline":
            lock = ROOT / ".local_monitors/batch3_stage_pipeline.lock"
            lock.unlink(missing_ok=True)
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


print("GATE_N12_START cwd=", os.getcwd(), "baseline=", BASELINE, flush=True)
deadline = time.time() + 55 * 60
while time.time() < deadline:
    ensure("longpoll_n4", "batch3_longpoll.log")
    ensure("batch3_stage_pipeline", "batch3_stage_pipeline.log")
    try:
        done, miss = volume()
    except Exception as e:
        print("vol_err", e, flush=True)
        time.sleep(20)
        continue
    n = len(done)
    if n > BASELINE or not miss:
        print(f"N_JUMP_DETECTED n={n}", flush=True)
        subprocess.run([PY, "-u", str(ROOT / ".local_monitors/batch3_refresh_once.py")], cwd=str(ROOT), check=False)
        fresh = [x for x in done if x not in PREV]
        copied = {}
        for name in fresh:
            src = RESULTS_DIR / f"{name}.json"
            for _ in range(6):
                if src.exists():
                    break
                try:
                    fetch([name])
                except Exception as e:
                    print("fetch_err", e, flush=True)
                time.sleep(1)
            if src.exists():
                MEDIA.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, MEDIA / f"{name}.json")
                shutil.copy2(src, STORE / "media/p4" / f"{name}.json")
                lm = json.loads(src.read_text()).get("lambda_max")
                copied[name] = lm
                print(f"COPIED {name} lambda_max={lm}", flush=True)
        for script in ("p4-fill-appendix-from-media.py", "p4-update-lambda-plot.py"):
            p = STORE / "internal" / script
            if p.exists():
                subprocess.run(["python3", str(p)], cwd=str(STORE), check=False)
        d = json.loads(PROG.read_text()) if PROG.exists() else {}
        lams = d.get("lambda_max") or {}
        new = {k: (copied.get(k) if copied.get(k) is not None else lams.get(k)) for k in fresh}
        for k in fresh:
            if new.get(k) is None:
                mp = MEDIA / f"{k}.json"
                if mp.exists():
                    new[k] = json.loads(mp.read_text()).get("lambda_max")
        report = STORE / "internal/batch3-n12-jump-report.md"
        report.write_text(
            "---\n"
            "cursor:\n"
            '  subagentId: "bc-8a994472-d1f4-5aa9-a973-7602cdc5af8b"\n'
            "---\n\n"
            f"# Batch 3 stage-a N↑ 11→{n}\n\n"
            "- **checkout:** `/Users/daxtongutekunst/Desktop/assignments` ONLY\n"
            "- **forbidden_used:** false\n"
            f"- **new DONE / λ_max:** {new}\n"
            f"- **n:** {n} / 13\n"
            f"- **live_expect:** {d.get('live_expect')}\n"
            f"- **gpu_utils:** {d.get('gpu_utils')}\n"
            f"- **miss:** {d.get('miss') or miss}\n"
            f"- **complete:** {not miss}\n"
        )
        # bump baselines in longpoll/waiter for continued watch
        for path, pat, repl in (
            (ROOT / ".local_monitors/longpoll_n4.py", r"baseline\s*=\s*\d+", f"baseline = {n}"),
            (ROOT / ".local_monitors/batch3_waiter_n8.py", r"BASELINE\s*=\s*\d+", f"BASELINE = {n}"),
        ):
            t = path.read_text()
            path.write_text(re.sub(pat, repl, t, count=1))
        payload = {
            "reason": "complete" if not miss else "n_jump",
            "from": BASELINE,
            "n": n,
            "lambda_max_new": new,
            "miss": miss,
            "live_expect": d.get("live_expect"),
            "gpu_utils": d.get("gpu_utils"),
            "checkout": str(ROOT),
            "forbidden_used": False,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print("GATE_N12_EXIT", json.dumps(payload), flush=True)
        if not miss:
            print("HIT_13_PIPELINE", flush=True)
        raise SystemExit(0)
    time.sleep(20)

OUT.write_text(json.dumps({"reason": "timeout", "n": BASELINE}) + "\n")
print("GATE_N12_TIMEOUT", flush=True)
raise SystemExit(2)
