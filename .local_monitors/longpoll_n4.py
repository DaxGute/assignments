#!/usr/bin/env python3
"""Long-poll stage-a volume N; copy new JSON to store media/p4/stage-a/.

Does NOT relaunch Modal stage a. On 13/13, ensures durable pipeline monitor
(fetch → b-rescale → fetch → c → fetch) via ensure_batch3_monitors.sh.
Checkout: /Users/daxtongutekunst/Desktop/assignments ONLY.
"""
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
    raise SystemExit("FORBIDDEN checkout")
os.environ["PYTHONPATH"] = str(ROOT)
os.environ.setdefault("MODAL_ENVIRONMENT", "cs312-daxton")
sys.path.insert(0, str(ROOT))

from experiments.a3_optimization.p4_inside_the_hessian import STAGES
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, _filename, _volume_results, fetch

STORE = Path(
    "/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files"
)
STATUS = STORE / "internal/mac-gpu-status.md"
STAGE_A = STORE / "media/p4/stage-a"
MEDIA = STORE / "media/p4"
STAGE_A.mkdir(parents=True, exist_ok=True)
PROG = STORE / "media/p4/stage_a_progress.json"
LOCAL_PROG = ROOT / ".local_monitors/batch3_progress.json"
RESULT = Path("/tmp/a3-batch3-longpoll-result.json")
APP = "ap-DAUDZsFRech705uMDN7q3x"
a = STAGES["a"]
baseline = 13
# Long-lived enough for this turn; durable pipeline covers post-13/13.
deadline = time.time() + 6 * 3600
interval = 90


def strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def memscan_live(cid: str, miss: list[str]) -> str | None:
    """Best-effort: pick a still-pending a-adamw-* name from container memory."""
    script = (
        "import re\n"
        "jobs=set()\n"
        "try:\n"
        " maps=open('/proc/2/maps');b=open('/proc/2/mem','rb');regs=[]\n"
        " for line in maps:\n"
        "  p=line.split()\n"
        "  if len(p)<2 or 'r' not in p[1] or 'nvidia' in line: continue\n"
        "  a,c=p[0].split('-');s,e=int(a,16),int(c,16)\n"
        "  if 0<e-s<=64*1024*1024: regs.append((s,e))\n"
        " for s,e in regs:\n"
        "  try:\n"
        "   b.seek(s);d=b.read(e-s)\n"
        "  except Exception: continue\n"
        "  jobs.update(re.findall(rb'a-adamw-\\d+', d))\n"
        "except Exception: pass\n"
        "print(','.join(sorted(x.decode() for x in jobs)))\n"
    )
    try:
        p = subprocess.run(
            ["modal", "container", "exec", "--no-pty", cid, "--", "python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, "MODAL_ENVIRONMENT": "cs312-daxton"},
        )
        lines = [L.strip() for L in (p.stdout or "").splitlines() if L.strip()]
        jobs = [x for x in (lines[0].split(",") if lines else []) if x.startswith("a-adamw-")]
        pending = [j for j in jobs if j in miss]
        return pending[-1] if pending else (jobs[-1] if jobs else None)
    except Exception:
        return None


env = {**os.environ, "MODAL_ENVIRONMENT": "cs312-daxton"}
print(f"LONGPOLL_DURABLE_START cwd={os.getcwd()} baseline={baseline}", flush=True)

while time.time() < deadline:
    if os.getcwd() != str(ROOT):
        raise SystemExit("WRONG_CWD")
    got = fetch([j["name"] for j in a])
    copied_new = []
    lam = {}
    for name in got:
        src = RESULTS_DIR / f"{name}.json"
        if not src.exists():
            continue
        if not str(src).startswith(str(ROOT) + "/"):
            raise SystemExit(f"FORBIDDEN_PATH {src}")
        dst = STAGE_A / f"{name}.json"
        is_new = not dst.exists()
        shutil.copy2(src, dst)
        shutil.copy2(src, MEDIA / f"{name}.json")
        r = json.loads(src.read_text())
        lam[name] = r.get("lambda_max")
        s = (
            json.dumps(
                {
                    "job": name,
                    "lambda_max": r.get("lambda_max"),
                    "lambda_min": r.get("lambda_min"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                indent=2,
            )
            + "\n"
        )
        (STAGE_A / f"{name}.summary.json").write_text(s)
        (MEDIA / f"{name}.summary.json").write_text(s)
        if is_new:
            copied_new.append(name)
            print("NEW_JSON", name, "lambda_max=", r.get("lambda_max"), flush=True)

    files = _volume_results()
    done = [j["name"] for j in a if _filename(j) in files]
    miss = [j["name"] for j in a if _filename(j) not in files]
    apps = json.loads(
        strip(
            subprocess.check_output(
                ["modal", "app", "list", "--env", "cs312-daxton", "--json"], text=True
            )
        )
    )
    b3 = next(x for x in apps if (x.get("app_id") or x.get("App ID")) == APP)
    tasks = b3.get("tasks") if b3.get("tasks") is not None else b3.get("Tasks")
    state = b3.get("state") or b3.get("State")
    desc = b3.get("description") or b3.get("Description")
    conts = json.loads(
        strip(
            subprocess.check_output(
                ["modal", "container", "list", "--env", "cs312-daxton", "--json"], text=True
            )
        )
    )
    cids = [
        c.get("Container ID") or c.get("container_id")
        for c in conts
        if (c.get("App ID") or c.get("app_id")) == APP
    ]
    utils = []
    gpu_rows = []
    live_expect: list[str] = []
    for cid in cids:
        util = mem = None
        try:
            p = subprocess.run(
                [
                    "modal",
                    "container",
                    "exec",
                    "--no-pty",
                    cid,
                    "--",
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=18,
                env=env,
            )
            m = re.search(
                r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
                strip((p.stdout or "") + "\n" + (p.stderr or "")),
            )
            if m:
                util = int(m.group(1))
                mem = f"{m.group(2)}/{m.group(3)} MiB"
                utils.append(util)
        except Exception:
            pass
        live_job = memscan_live(cid, miss)
        if live_job and live_job not in live_expect:
            live_expect.append(live_job)
        gpu_rows.append({"id": cid, "gpu_util": util, "mem": mem, "live_job": live_job})

    now = time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    n = len(done)
    increased = n > baseline
    complete = not miss
    busy = sum(1 for u in utils if u >= 50)
    try:
        tasks_i = int(str(tasks))
    except Exception:
        tasks_i = -1
    if cids and (busy == len(cids) or (not utils and tasks_i > 0)):
        gpu_status = "BUSY"
    elif busy:
        gpu_status = "PARTIAL"
    elif cids and utils and all(u <= 2 for u in utils):
        gpu_status = "IDLE?"
    elif cids:
        gpu_status = "UNKNOWN_SMI_FAIL"
    else:
        gpu_status = "NO_CONTAINERS"

    prog = {
        "n": n,
        "of": 13,
        "remaining": len(miss),
        "baseline": baseline,
        "increased": increased,
        "complete": complete,
        "done": done,
        "miss": miss,
        "copied_new": copied_new,
        "lambda_max": lam,
        "tasks": tasks,
        "containers": gpu_rows,
        "gpu_utils": utils,
        "gpu_status": gpu_status,
        "still_running": n < 13 and (busy > 0 or tasks_i > 0),
        "live_expect": live_expect,
        "ts": now,
        "checkout": str(ROOT),
        "forbidden_used": False,
    }
    PROG.write_text(json.dumps(prog, indent=2) + "\n")
    LOCAL_PROG.write_text(
        json.dumps(
            {
                "phase": "stage_a" if miss else "stage_a_complete",
                "done": done,
                "miss": miss,
                "tasks": tasks,
                "state": state,
                "containers": cids,
                "media_stage_a": sorted(
                    p.stem for p in STAGE_A.glob("a-adamw-*.json") if p.name.count(".") == 1
                ),
                "copied_this_tick": copied_new,
                "gpu_status": gpu_status,
                "ts": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    Path("/tmp/a3-batch3-progress.json").write_text(json.dumps(prog, indent=2))

    # Do not nest ** inside the heading phase (breaks other writers' replace regex).
    phase = (
        "STAGE a COMPLETE"
        if complete
        else (
            f"RUNNING stage a — {n}/13 N↑"
            if increased
            else f"RUNNING stage a — {n}/13 WAIT ({gpu_status})"
        )
    )
    phase_heading = phase  # already free of nested bold
    gpu_md = (
        "; ".join(
            f"`{g['id']}` GPU={g['gpu_util'] if g['gpu_util'] is not None else 'n/a'}%"
            + (f" ({g['mem']})" if g["mem"] else "")
            for g in gpu_rows
        )
        or "_none_"
    )
    block = f"""## Batch 3 — P4 Hessian — **{phase_heading}** (refreshed {now})

| Field | Value |
|-------|--------|
| Checkout | `/Users/daxtongutekunst/Desktop/assignments` ONLY |
| Forbidden | `assignments-a3-batch1` **not used** |
| Modal app | `{desc}` / **`{APP}`** |
| App state | {state} |
| Modal tasks | **{tasks}** |
| Live jobs (probed) | {", ".join(f"`{x}`" for x in live_expect) or "_unknown_"} — no relaunch |
| GPU status | **{gpu_status}** |
| GPU util | {gpu_md} |
| Stage `a` volume | **{n} / 13** (remaining **{len(miss)}**){" ⬆️ from "+str(baseline) if increased else ""} |
| Finished | {", ".join(f"`{x}`" for x in done)} |
| Pending | {", ".join(f"`{x}`" for x in miss) or "_none_"} |
| Copied new | {", ".join(f"`{x}`" for x in copied_new) or "_none_"} |
| Next | on 13/13: fetch → b-rescale → fetch → c → fetch |

"""
    text = STATUS.read_text()
    new, nn = re.subn(
        r"## Batch 3 — P4 Hessian[\s\S]*?(?=\n## |\Z)",
        block + "\n",
        text,
        count=1,
    )
    if nn == 1:
        new = re.sub(
            r"\| Batch 3 \|[^\n]*\|",
            f"| Batch 3 | **{n}/13** {'COMPLETE' if complete else 'WAIT'}; tasks={tasks} |",
            new,
            count=1,
        )
        STATUS.write_text(new)

    print(
        f"tick N={n}/13 utils={utils} live={live_expect} tasks={tasks} copied={copied_new}",
        flush=True,
    )

    # Re-ensure monitors only on N↑ / 13/13 (avoid thrash). Watchdog covers crashes.
    if increased or complete:
        reason = "complete" if complete else "n_jump"
        payload = {
            "reason": reason,
            "from": baseline,
            "n": n,
            "done": done,
            "miss": miss,
            "copied_new": copied_new,
            "lambda_max": {k: lam.get(k) for k in copied_new},
            "tasks": tasks,
            "gpu_utils": utils,
            "gpu_status": gpu_status,
            "ts": now,
            "checkout": str(ROOT),
            "forbidden_used": False,
        }
        RESULT.write_text(json.dumps(payload, indent=2) + "\n")
        print("EXIT_EVENT", reason, flush=True)
        if complete:
            # Never inline-run a second pipeline; ensure durable monitor holds the lock.
            subprocess.run(["bash", str(ensure)], cwd=str(ROOT), timeout=60, check=False)
        print("FINAL", json.dumps(payload), flush=True)
        if complete:
            break
        baseline = n  # continue watching for further N increases
    time.sleep(interval)
else:
    payload = {
        "reason": "timeout",
        "n": baseline,
        "checkout": str(ROOT),
        "forbidden_used": False,
        "ts": time.strftime("%Y-%m-%d %H:%M PDT"),
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n")
    print("EXIT_EVENT timeout", flush=True)
    print("FINAL", json.dumps(payload), flush=True)
    # On timeout, still leave durable pipeline/watchdog running.
    subprocess.run(
        ["bash", str(ROOT / ".local_monitors/ensure_batch3_monitors.sh")],
        cwd=str(ROOT),
        timeout=60,
        check=False,
    )
