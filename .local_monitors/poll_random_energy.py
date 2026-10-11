#!/usr/bin/env python3
"""Poll a-random-energy until JSON/Tasks=0 or hard-block idle."""
from __future__ import annotations
import json, re, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DESKTOP = Path("/Users/daxtongutekunst/Desktop/assignments")
STORE = Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
START = datetime(2026, 10, 10, 21, 56, 22, tzinfo=ZoneInfo("America/Los_Angeles"))
APP = "ap-Kbcc9LczZmezf6dUaOJF0l"
CONTAINERS = [
    "ta-01M4MN3FHDG16J920JTBY0588R",
    "ta-01M4MN3FHDPEFVHM4KB1JF4ZZR",
]
LOG = DESKTOP / ".local_monitors/random-energy-poll-loop.jsonl"
TZ = ZoneInfo("America/Los_Angeles")


def now():
    return datetime.now(TZ)


def elapsed_min():
    return round((now() - START).total_seconds() / 60, 1)


def run(cmd, timeout=90):
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def app_tasks():
    code, out = run(["modal", "app", "list"])
    tasks = None
    present = APP in out
    for line in out.splitlines():
        if APP not in line:
            continue
        m = re.search(r"│\s*(\d+)\s*│\s*2026-", line)
        if m:
            tasks = int(m.group(1))
            break
        parts = [p.strip() for p in line.split("│")]
        for p in parts:
            if p.isdigit():
                tasks = int(p)
    stopped = present and ("stopped" in out.split(APP, 1)[-1][:80])
    return {"present": present, "tasks": tasks, "stopped_hint": stopped, "raw_line": next((l for l in out.splitlines() if APP in l), "")}


def volume_random():
    code = (
        "from experiments.a3_optimization.hessian_jobs import _volume_results\n"
        "done=sorted(_volume_results())\n"
        "print('COUNT', len(done))\n"
        "print('HAS1526', 'a-random-energy-1526.json' in done)\n"
        "print('HAS9375', 'a-random-energy-9375.json' in done)\n"
    )
    p = subprocess.run(
        ["uv", "run", "python", "-c", code],
        cwd=str(DESKTOP), text=True, capture_output=True, timeout=120,
    )
    out = p.stdout
    d = {"count": None, "has1526": False, "has9375": False, "rc": p.returncode}
    for line in out.splitlines():
        if line.startswith("COUNT"):
            d["count"] = int(line.split()[1])
        if line.startswith("HAS1526"):
            d["has1526"] = line.split()[1] == "True"
        if line.startswith("HAS9375"):
            d["has9375"] = line.split()[1] == "True"
    return d


def gpu_util(cid):
    p = subprocess.run(
        [
            "modal", "container", "exec", cid, "--", "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True, capture_output=True, timeout=90,
    )
    blob = (p.stdout or "") + "\n" + (p.stderr or "")
    for line in blob.splitlines():
        if "%" in line and "MiB" in line:
            parts = [x.strip() for x in line.split(",")]
            try:
                util = int(parts[0].replace("%", "").strip())
            except ValueError:
                continue
            return {"util": util, "mem": parts[1] if len(parts) > 1 else "?", "total": parts[2] if len(parts) > 2 else "?", "raw": line.strip()}
    return {"error": "no-parse", "rc": p.returncode, "raw": blob[:300]}


def container_active():
    code, out = run(["modal", "container", "list"])
    active = [c for c in CONTAINERS if c[:18] in out or c in out]
    return active


def write_status(rec, action):
    gpus = rec["gpus"]
    rows = []
    for cid, g in gpus.items():
        if "util" in g:
            rows.append(f"| `{cid}` | 2026-10-10 21:56 PDT | **{g['util']}%**, {g.get('mem','?')} / {g.get('total','?')} |")
        else:
            rows.append(f"| `{cid}` | 2026-10-10 21:56 PDT | util n/a ({g.get('error','?')}) |")
    gpu_table = "\n".join(rows) if rows else "| (none active) | — | — |"
    utils = [g["util"] for g in gpus.values() if "util" in g]
    util_str = " / ".join(f"{u}%" for u in utils) if utils else "n/a"
    el = rec["elapsed_min"]
    tasks = rec["tasks"]
    vol = rec["volume"]
    hard = action == "HARD_BLOCK"
    status_line = (
        f"**HARD BLOCK — GPUs idle, Tasks={tasks}, no JSON**"
        if hard
        else f"**STILL RUNNING (~{el} min) — not a hard hang** — Tasks={tasks}, volume JSON absent, GPUs busy ({util_str}). Not scored. No invented HIT/MISS. **No apply**."
    )
    if action == "APPLY":
        status_line = f"**READY — volume JSON present / Tasks={tasks}** — sole-owner apply in progress."

    launch = STORE / "internal/p4-block-energy-launch-status.md"
    old = launch.read_text()
    end = old.find("\n---\n", 3)
    fm = old[: end + 5] if end != -1 else "---\ncursor:\n  subagentId: \"bc-17ba57d4-9b71-5a62-a2ec-de239d99c11a\"\n---\n"
    body = f'''
# P4 block-energy — a-random-energy launch status

**Status:** {status_line}

**Runbook:** [`p4-block-energy-fill-on-complete.md`](p4-block-energy-fill-on-complete.md) · [`p4-block-energy-merge-and-score.py`](p4-block-energy-merge-and-score.py)  
**Timer2 poll:** [`p4-block-energy-timer2-poll.md`](p4-block-energy-timer2-poll.md)

## Path lock

| Field | Value |
|-------|-------|
| cwd | `/Users/daxtongutekunst/Desktop/assignments` |
| batch1 in path? | **no** |
| Modal env | `cs312-daxton` |

## Live poll — {rec['ts']} (~{el} min)

| Field | Value |
|-------|-------|
| App ID | `{APP}` |
| State | ephemeral (detached) / see Tasks |
| **Tasks** | **{tasks}** |
| Created | 2026-10-10 21:56:22-07:00 |
| **Elapsed** | **~{el} min** |
| Dashboard | https://modal.com/apps/cs312-f26/cs312-daxton/{APP} |

### Containers

{gpu_table}

### Volume / local

| Artifact | Present? |
|----------|----------|
| Volume `a-random-energy-1526.json` | **{'YES' if vol.get('has1526') else 'NO'}** |
| Volume `a-random-energy-9375.json` | **{'YES' if vol.get('has9375') else 'NO'}** |
| Volume file count | {vol.get('count')} |
| Local Desktop `a-random-energy-*.json` | check after fetch |

## Escalate decision

| Condition | Met? |
|-----------|------|
| Volume JSON both | **{'yes' if vol.get('has1526') and vol.get('has9375') else 'no'}** |
| Tasks=2 + GPUs busy | **{'yes' if action=='CONTINUE' else 'n/a'}** |
| GPUs idle + Tasks=2 + no JSON | **{'YES — HARD BLOCK' if hard else 'no'}** |

## Coordination

Sole-owner apply: `bc-285f1cf8-…`. No invent. Never batch1.
'''
    launch.write_text(fm + "\n" + body.lstrip("\n"))

    poll = STORE / "internal/p4-block-energy-timer2-poll.md"
    poll.write_text(f'''---
cursor:
  subagentId: "bc-285f1cf8-afd2-5ba3-854a-c7720b255b33"
---

# P4 block-energy — timer2 continuous poll

**Status:** {status_line}

| Field | Value |
|-------|-------|
| cwd | `/Users/daxtongutekunst/Desktop/assignments` |
| App | `{APP}` |
| Poll ts | {rec['ts']} |
| Elapsed | ~{el} min |
| Tasks | {tasks} |
| GPU util | {util_str} |
| JSON 1526/9375 | {vol.get('has1526')} / {vol.get('has9375')} |
| Action | `{action}` |

Shared: [`p4-block-energy-launch-status.md`](p4-block-energy-launch-status.md)
''')
    if hard:
        diag = STORE / "internal/p4-block-energy-hard-block-diagnosis.md"
        diag.write_text(f'''---
cursor:
  subagentId: "bc-285f1cf8-afd2-5ba3-854a-c7720b255b33"
---

# P4 block-energy HARD BLOCK diagnosis

**Triggered:** GPUs idle AND Tasks=2 AND no volume JSON.

| Field | Value |
|-------|-------|
| App | `{APP}` |
| Poll ts | {rec['ts']} |
| Elapsed | ~{el} min |
| Tasks | {tasks} |
| GPU util | {util_str} (all ≤5%) |
| Volume 1526 | NO |
| Volume 9375 | NO |
| Containers | {', '.join(rec.get('active') or [])} |

## Evidence

```json
{json.dumps(rec, indent=2)}
```

## Implication

Jobs appear stuck without finishing (no commit of `a-random-energy-*.json`). Do **not** invent HIT/MISS. Operator decision needed (kill/relaunch vs wait). Sole-owner apply blocked.
''')


def poll_once():
    ts = now()
    app = app_tasks()
    vol = volume_random()
    active = container_active()
    # if containers gone but APP still listed with tasks, try known IDs anyway
    targets = active or (CONTAINERS if (app.get("tasks") or 0) > 0 else [])
    gpus = {c: gpu_util(c) for c in targets}
    rec = {
        "ts": ts.isoformat(timespec="seconds"),
        "elapsed_min": elapsed_min(),
        "tasks": app.get("tasks"),
        "volume": vol,
        "gpus": gpus,
        "active": active,
        "app_line": app.get("raw_line"),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.open("a").write(json.dumps(rec) + "\n")

    has_json = bool(vol.get("has1526") and vol.get("has9375"))
    tasks = app.get("tasks")
    utils = [g["util"] for g in gpus.values() if "util" in g]
    # idle only if we successfully read util and all ≤5
    idle = bool(utils) and all(u <= 5 for u in utils)
    app_gone = not app.get("present") or tasks == 0

    if has_json or (app_gone and has_json):
        action = "APPLY"
    elif has_json is False and (tasks == 0 or app_gone):
        # finished but no JSON — hard failure
        action = "HARD_BLOCK"
        # refine: if tasks 0 and no json, that's also hard
    elif tasks == 2 and idle and not has_json:
        action = "HARD_BLOCK"
    elif has_json:
        action = "APPLY"
    else:
        action = "CONTINUE"

    # If tasks is 0/None and containers gone and no JSON -> hard block
    if not has_json and (tasks == 0 or (not active and tasks is None)):
        # only hard if app stopped
        if not active and not has_json:
            # check if app still running with tasks
            if tasks == 2:
                action = "CONTINUE" if not idle else "HARD_BLOCK"
            elif tasks == 0 or not app.get("present"):
                action = "HARD_BLOCK"

    if has_json:
        action = "APPLY"

    write_status(rec, action)
    print(json.dumps({"action": action, "elapsed_min": rec["elapsed_min"], "tasks": tasks, "utils": utils, "has_json": has_json}, indent=2))
    return action, rec


if __name__ == "__main__":
    action, rec = poll_once()
    sys.exit(0 if action != "ERROR" else 1)
