#!/usr/bin/env python3
"""One-shot: volume poll, copy missing stage-a JSON to store, refresh progress/status."""
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
STATUS = STORE / "internal/mac-gpu-status.md"
PROG = STORE / "media/p4/stage_a_progress.json"
LOCAL = ROOT / ".local_monitors/stage_a_progress.json"
APP = "ap-DAUDZsFRech705uMDN7q3x"
a = STAGES["a"]
MEDIA.mkdir(parents=True, exist_ok=True)


def strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def modal_info():
    out = strip(
        subprocess.check_output(
            ["modal", "app", "list", "--env", "cs312-daxton", "--json"], text=True, timeout=60
        )
    )
    for x in json.loads(out):
        if (x.get("app_id") or x.get("App ID")) == APP:
            return (
                x.get("tasks") if x.get("tasks") is not None else x.get("Tasks", "?"),
                x.get("state") or x.get("State", "?"),
            )
    return "?", "missing"


def containers():
    out = strip(
        subprocess.check_output(
            ["modal", "container", "list", "--env", "cs312-daxton", "--json"], text=True, timeout=60
        )
    )
    return [
        c.get("Container ID") or c.get("container_id")
        for c in json.loads(out)
        if (c.get("App ID") or c.get("app_id")) == APP
    ]


def smi(cid):
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
            timeout=30,
        )
        m = re.search(
            r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            strip((p.stdout or "") + "\n" + (p.stderr or "")),
        )
        if m:
            return int(m.group(1)), f"{m.group(2)}/{m.group(3)} MiB"
    except Exception:
        pass
    return None, None


def memscan(cid, miss):
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
        )
        lines = [L.strip() for L in (p.stdout or "").splitlines() if L.strip()]
        jobs = [x for x in (lines[0].split(",") if lines else []) if x.startswith("a-adamw-")]
        pending = [j for j in jobs if j in miss]
        return pending[-1] if pending else (jobs[-1] if jobs else None)
    except Exception:
        return None


files = _volume_results()
done = [j["name"] for j in a if _filename(j) in files]
miss = [j["name"] for j in a if _filename(j) not in files]
print(f"VOLUME {len(done)}/13 done={done}", flush=True)
print(f"MISS {miss}", flush=True)

already = {p.stem for p in MEDIA.glob("a-adamw-*.json") if p.name.count(".") == 1}
need = [n for n in done if n not in already]
copied = []
for attempt in range(6):
    still = [n for n in need if n not in copied]
    if not still:
        break
    try:
        fetch(still)
    except Exception as e:
        print(f"fetch_err {e}", flush=True)
    for name in still:
        src = RESULTS_DIR / f"{name}.json"
        for _ in range(5):
            if src.exists():
                break
            time.sleep(1)
        if src.exists():
            shutil.copy2(src, MEDIA / f"{name}.json")
            shutil.copy2(src, STORE / "media/p4" / f"{name}.json")
            d = json.loads(src.read_text())
            copied.append(name)
            print(f"COPIED {name} lambda_max={d.get('lambda_max')}", flush=True)
    time.sleep(1)

lambda_max = {}
for n in done:
    p = MEDIA / f"{n}.json"
    if p.exists():
        d = json.loads(p.read_text())
        lm = d.get("lambda_max")
        if lm is None and isinstance(d.get("result"), dict):
            lm = d["result"].get("lambda_max")
        if lm is not None:
            lambda_max[n] = lm

tasks, state = modal_info()
cids = containers()
cout = []
live = []
utils = []
for cid in cids:
    util, mem = smi(cid)
    lj = memscan(cid, set(miss))
    cout.append({"id": cid, "gpu_util": util, "mem": mem, "live_job": lj})
    if util is not None:
        utils.append(util)
    if lj and lj not in live:
        live.append(lj)

still = bool(cids) or str(tasks) not in ("0", "?")
progress = {
    "n": len(done),
    "of": 13,
    "remaining": len(miss),
    "baseline": len(done),
    "increased": False,
    "complete": not miss,
    "done": done,
    "miss": miss,
    "copied_new": copied,
    "lambda_max": lambda_max,
    "tasks": str(tasks),
    "app_state": state,
    "containers": cout,
    "gpu_utils": utils,
    "gpu_status": "BUSY" if (utils and max(utils) >= 50) or still else "IDLE",
    "still_running": still,
    "live_expect": live,
    "ts": time.strftime("%Y-%m-%d %H:%M PDT"),
    "checkout": str(ROOT),
    "forbidden_used": False,
}
PROG.write_text(json.dumps(progress, indent=2) + "\n")
LOCAL.write_text(json.dumps(progress, indent=2) + "\n")

now = time.strftime("%Y-%m-%d %H:%M PDT")
phase = "STAGE a COMPLETE — fetching" if not miss else "RUNNING stage a"
new_lams = {k: lambda_max[k] for k in copied if k in lambda_max}
live_s = "; ".join(
    f"`{(c['id'] or '').replace('ta-01M4M945', 'ta-…')}`→`{c.get('live_job')}` util={c.get('gpu_util')}"
    for c in cout
)
block = f"""## Batch 3 — P4 Hessian — **{phase}** (refreshed {now})

| Field | Value |
|-------|--------|
| Checkout | `/Users/daxtongutekunst/Desktop/assignments` ONLY |
| Modal app | **`{APP}`** |
| App state | {state} |
| Modal tasks | **{tasks}** |
| Containers | {", ".join(f"`{c}`" for c in cids) or "_none_"} |
| Stage `a` volume | **{len(done)} / 13** |
| Finished | {", ".join(f"`{x}`" for x in done)} |
| Pending | {", ".join(f"`{x}`" for x in miss) or "_none_"} |
| GPU utils | **{utils}** ({progress['gpu_status']}) |
| live_expect | {", ".join(f"`{x}`" for x in live) or "_unknown_"} |
| Scoring | P4-a-* **PENDING** — no invent |
| Next | on 13/13: fetch → b-rescale → fetch → c → fetch |
| Extra | copied={copied}; λ_new={new_lams}; LIVE {live_s}; WAIT no relaunch |

"""
text = STATUS.read_text()
new, ns = re.subn(r"## Batch 3 — P4 Hessian[\s\S]*?(?=\n## |\Z)", block + "\n", text, count=1)
if ns == 1:
    new = re.sub(
        r"\| Batch 3 \|[^\n]*\|",
        f"| Batch 3 | **{phase}** — `{APP}`; stage a {len(done)}/13; GPUs {utils} |",
        new,
        count=1,
    )
    STATUS.write_text(new)

print(
    json.dumps(
        {
            "n": len(done),
            "copied": copied,
            "lambda_max_new": new_lams,
            "lam1526": lambda_max.get("a-adamw-1526"),
            "lam3052": lambda_max.get("a-adamw-3052"),
            "live_expect": live,
            "gpu_utils": utils,
            "miss": miss,
            "complete": not miss,
        },
        indent=2,
    ),
    flush=True,
)
