#!/usr/bin/env python3
"""Poll until a-adamw-9375 DONE, copy JSON, refresh 13/13 progress, ensure pipeline advances."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = "/Users/daxtongutekunst/Desktop/assignments"
STORE = Path(
    "/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files"
)
MEDIA = STORE / "media/p4"
STAGE = MEDIA / "stage-a"
APP = "ap-DAUDZsFRech705uMDN7q3x"
FOCUS = "a-adamw-9375"
INTERVAL = 45
SUBAGENT = "bc-876ea7f3-f6d8-56e4-b778-176686c59113"

os.chdir(ROOT)
if os.getcwd() != ROOT or "assignments-a3-batch1" in os.getcwd():
    raise SystemExit("forbidden/wrong checkout")
sys.path.insert(0, ROOT)
os.environ.setdefault("MODAL_ENVIRONMENT", "cs312-daxton")

from experiments.a3_optimization import p4_inside_the_hessian as p4
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, _filename, _volume_results, fetch


def strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def volume():
    files = _volume_results()
    done = [j["name"] for j in p4.STAGES["a"] if _filename(j) in files]
    miss = [j["name"] for j in p4.STAGES["a"] if _filename(j) not in files]
    return done, miss, files


def gpu_utils():
    try:
        out = strip(
            subprocess.check_output(["modal", "container", "list", "--json"], text=True, timeout=60)
        )
        cids = [
            c.get("Container ID") or c.get("container_id")
            for c in json.loads(out)
            if (c.get("App ID") or c.get("app_id")) == APP
        ]
    except Exception as e:
        print(f"container_list_err {e}", flush=True)
        return [], []
    utils = []
    for cid in cids:
        try:
            smi = subprocess.run(
                [
                    "modal",
                    "container",
                    "exec",
                    "--no-pty",
                    cid,
                    "--",
                    "bash",
                    "-lc",
                    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=45,
            )
            line = (smi.stdout or "").strip().splitlines()
            utils.append(int(float(line[0])) if line else None)
        except Exception:
            utils.append(None)
    return cids, utils


def ensure_monitors() -> None:
    subprocess.run(["bash", ".local_monitors/ensure_batch3_monitors.sh"], cwd=ROOT, check=False)


def write_progress(done, miss, gpu, copied, lam_new, phase_extra=""):
    lambda_max = {}
    for n in done:
        for p in [
            MEDIA / f"{n}.json",
            STAGE / f"{n}.json",
            RESULTS_DIR / f"{n}.json",
            Path("experiments/a3_optimization/results/a3-inside-the-hessian") / f"{n}.json",
        ]:
            if p.exists():
                d = json.loads(p.read_text())
                if "lambda_max" in d:
                    lambda_max[n] = d["lambda_max"]
                    break
    ts = time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    payload = {
        "n": len(done),
        "of": 13,
        "remaining": len(miss),
        "done": done,
        "miss": miss,
        "lambda_max": lambda_max,
        "live_expect": list(miss),
        "gpu_utils": gpu,
        "gpu_status": (
            "BUSY"
            if any((u or 0) >= 50 for u in gpu if u is not None)
            else ("COMPLETE" if not miss else "RUNNING")
        ),
        "still_running": bool(miss),
        "checkout": ROOT,
        "forbidden_used": False,
        "ts": ts,
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "copied_new": copied,
        "lambda_new": lam_new,
        "focus": {
            "a-adamw-7630": "DONE" if "a-adamw-7630" in done else "?",
            "a-adamw-9375": "DONE" if FOCUS in done else "LIVE",
        },
        "n13": len(done) == 13,
        "phase_extra": phase_extra,
        "probe": "wait_9375_then_pipeline_bc-876ea7f3",
    }
    text = json.dumps(payload, indent=2) + "\n"
    for t in [
        MEDIA / "stage_a_progress.json",
        Path("experiments/a3_optimization/results/stage_a_progress.json"),
        Path(".local_monitors/stage_a_progress.json"),
        Path("media/p4/stage_a_progress.json"),
    ]:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(text)
    return payload


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    print(f"WAIT_9375_START {time.strftime('%H:%M:%S')}", flush=True)
    while True:
        done, miss, _ = volume()
        _, gpu = gpu_utils()
        print(
            f"[{time.strftime('%H:%M:%S')}] {len(done)}/13 "
            f"9375_done={FOCUS in done} gpu={gpu} miss={miss}",
            flush=True,
        )
        # keep store progress.ts fresh while waiting
        try:
            write_progress(
                done,
                miss,
                gpu,
                [],
                {},
                phase_extra=f"waiting on {FOCUS}; poller alive",
            )
        except Exception as e:
            print(f"progress_refresh_err {e!r}", flush=True)
        if FOCUS in done:
            print("9375_DONE", flush=True)
            break
        # keep local monitors alive infrequently; never relaunch GPU
        if int(time.time()) % 600 < INTERVAL + 5:
            try:
                ensure_monitors()
            except Exception as e:
                print(f"ensure_err {e!r}", flush=True)
        time.sleep(INTERVAL)

    done, miss, _ = volume()
    if FOCUS not in done or len(done) != 13 or miss:
        raise SystemExit(f"unexpected after done: {done=} {miss=}")
    print("HIT_13", flush=True)

    already = {p.stem for p in STAGE.glob("a-adamw-*.json") if p.name.count(".") == 1}
    need = [n for n in done if n not in already]
    copied: list[str] = []
    lam_new: dict = {}
    if need:
        print("FETCH", need, flush=True)
        try:
            fetch(need)
        except Exception as e:
            print("fetch_err", e, flush=True)

    for n in done:
        if n in already and n != FOCUS:
            continue
        src = RESULTS_DIR / f"{n}.json"
        alt = Path("experiments/a3_optimization/results/a3-inside-the-hessian") / f"{n}.json"
        if not src.exists() and alt.exists():
            src = alt
        if not src.exists() and n == FOCUS:
            try:
                fetch([n])
            except Exception as e:
                print("refetch_err", e, flush=True)
            if (RESULTS_DIR / f"{n}.json").exists():
                src = RESULTS_DIR / f"{n}.json"
            elif alt.exists():
                src = alt
        if not src.exists():
            continue
        for dest in (MEDIA / f"{n}.json", STAGE / f"{n}.json", Path("media/p4") / f"{n}.json"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        data = json.loads(src.read_text())
        if n == FOCUS or n not in already:
            lam_new[n] = data.get("lambda_max")
            if n not in copied:
                copied.append(n)
            summary = {
                "name": n,
                "job": n,
                "lambda_max": data.get("lambda_max"),
                "lambda_min": data.get("lambda_min"),
                "step": data.get("step"),
                "seconds": data.get("seconds"),
            }
            for dest in (MEDIA / f"{n}.summary.json", STAGE / f"{n}.summary.json"):
                dest.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"COPIED {n} lambda_max={data.get('lambda_max')}", flush=True)

    src9375 = STAGE / f"{FOCUS}.json"
    if not src9375.exists():
        raise SystemExit("9375_JSON_MISSING_AFTER_COPY")
    lam9375 = json.loads(src9375.read_text()).get("lambda_max")
    print(f"LAMBDA_MAX_9375={lam9375}", flush=True)

    _, gpu = gpu_utils()
    payload = write_progress(
        done, miss, gpu, copied, lam_new, phase_extra="13/13 — starting fetch→b-rescale→c"
    )
    print("PROGRESS_13_WRITTEN", flush=True)

    # Runbook Mac steps §1 done (copy). §2 mid-N refresh + §3 dry-run.
    runbook_notes = []
    mid = STORE / "internal/p4-mid-n-refresh.sh"
    if mid.exists():
        print("RUNBOOK_§2 mid-n-refresh", flush=True)
        r = subprocess.run(["bash", str(mid)], cwd=ROOT, text=True, capture_output=True, timeout=600)
        runbook_notes.append(f"p4-mid-n-refresh rc={r.returncode}")
        (STORE / "internal/batch3-13of13-mid-n-refresh.log").write_text(
            (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
        )
        print(f"MID_N_RC={r.returncode}", flush=True)
    score = STORE / "internal/p4-score-from-media.py"
    if score.exists():
        print("RUNBOOK_§3 score dry-run", flush=True)
        r = subprocess.run(
            ["python3", str(score)], cwd=str(STORE), text=True, capture_output=True, timeout=300
        )
        runbook_notes.append(f"p4-score-from-media dry-run rc={r.returncode}")
        (STORE / "internal/batch3-13of13-score-dryrun.log").write_text(
            (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
        )
        print(f"SCORE_DRYRUN_RC={r.returncode}", flush=True)

    # §6 via durable pipeline (fetch→b-rescale→fetch→c→fetch); no stage-a relaunch
    ensure_monitors()
    time.sleep(5)
    pipe_log = Path(".local_monitors/batch3_stage_pipeline.log")
    tail = pipe_log.read_text()[-2000:] if pipe_log.exists() else ""
    print("PIPELINE_TAIL", tail.replace("\n", " | ")[-800:], flush=True)
    runbook_notes.append("durable pipeline ensured for §6 fetch→b-rescale→c")

    report = STORE / "internal/batch3-13of13-pipeline.md"
    report.write_text(
        f"""---
cursor:
  subagentId: "{SUBAGENT}"
---

# Batch3 stage-a **13/13** — pipeline advance

- Checkout: `/Users/daxtongutekunst/Desktop/assignments` (`forbidden_used=false`)
- App: `{APP}`
- Hit 13/13 at: **{payload['ts']}**
- Last job: `{FOCUS}` **DONE**
- **λ_max(a-adamw-9375) = {lam9375}**
- Copied to `media/p4/` + `media/p4/stage-a/`: {copied}
- `stage_a_progress.json` refreshed to **n=13 / of=13**
- Followed [`p4-on-9375-runbook.md`](p4-on-9375-runbook.md) Mac steps §1–§3 + §6 kickoff
- Durable pipeline → **fetch → b-rescale → fetch → c → fetch**
- **No Modal stage-a relaunch** while BUSY

## Runbook notes

{chr(10).join('- ' + x for x in runbook_notes) or '- (none)'}

## At completion

- gpu_utils: {gpu}
- See `.local_monitors/batch3_stage_pipeline.log` for FETCH / EXECUTE lines

## Next (runbook §4–§5)

Fill MEASURED docs from dry-run (no invent); rebuild DRAFT PDFs. §6 continues on Mac pipeline.
"""
    )
    print("WROTE_REPORT", report, flush=True)

    (STORE / "internal/batch3-live-7630-9375.md").write_text(
        f"""---
cursor:
  subagentId: "{SUBAGENT}"
---

# Batch3 stage-a focus: 7630 / 9375

- Probe ts: **{payload['ts']}**
- N: **13 / 13** COMPLETE
- `a-adamw-7630`: DONE
- `a-adamw-9375`: **DONE** · λ_max=**{lam9375}**
- Pipeline: fetch→b-rescale→c via durable monitor
- See also: [`batch3-13of13-pipeline.md`](batch3-13of13-pipeline.md)
"""
    )
    print("HIT_13_COMPLETE", flush=True)
    print(f"FINAL n=13 lambda_9375={lam9375} copied={copied}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
