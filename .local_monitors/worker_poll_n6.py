#!/usr/bin/env python3
"""Long-poll stage-a from Desktop/assignments ONLY. Never relaunch Modal stage a."""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path("/Users/daxtongutekunst/Desktop/assignments")
os.chdir(ROOT)
assert os.getcwd() == str(ROOT)
if "assignments-a3-batch1" in os.getcwd():
    raise SystemExit("FORBIDDEN")
os.environ["PYTHONPATH"] = str(ROOT)
os.environ.setdefault("MODAL_ENVIRONMENT", "cs312-daxton")
sys.path.insert(0, str(ROOT))

from experiments.a3_optimization.p4_inside_the_hessian import STAGES
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, _filename, _volume_results, fetch

STORE = Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
STAGE_A = STORE / "media/p4/stage-a"
MEDIA = STORE / "media/p4"
STAGE_A.mkdir(parents=True, exist_ok=True)
PROG = STORE / "media/p4/stage_a_progress.json"
LOCAL = ROOT / ".local_monitors/batch3_progress.json"
RESULT = Path("/tmp/a3-batch3-longpoll-result.json")
MID = STORE / "internal/p4-mid-n-refresh.sh"
APP = "ap-DAUDZsFRech705uMDN7q3x"
a = STAGES["a"]
baseline = 12
deadline = time.time() + 3 * 3600
interval = 45
env = {**os.environ, "MODAL_ENVIRONMENT": "cs312-daxton"}

def strip(s): return re.sub(r"\x1b\[[0-9;]*m", "", s)

def smi_one(cid):
    for cmd in (
        ["modal", "container", "exec", "--no-pty", cid, "--", "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        ["modal", "container", "exec", cid, "--", "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
    ):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=22, env=env)
            text = strip((p.stdout or "") + "\n" + (p.stderr or ""))
            m = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
            if m:
                return int(m.group(1)), f"{m.group(2)}/{m.group(3)} MiB"
        except Exception:
            pass
    return None, None

def discover_jobs(cid):
    py = (
        "import re\njobs=set()\n"
        "maps=open('/proc/2/maps')\nregs=[]\n"
        "for line in maps:\n p=line.split()\n"
        " if len(p)<2 or 'r' not in p[1] or 'nvidia' in line: continue\n"
        " a,c=p[0].split('-'); s,e=int(a,16),int(c,16)\n"
        " if 0<e-s<=64*1024*1024: regs.append((s,e))\n"
        "b=open('/proc/2/mem','rb')\n"
        "for s,e in regs:\n"
        " try:\n  b.seek(s); d=b.read(e-s)\n except Exception: continue\n"
        " jobs.update(re.findall(rb'a-adamw-\\d+', d))\n"
        "print(','.join(sorted(x.decode() for x in jobs)))\n"
    )
    try:
        p = subprocess.run(["modal","container","exec","--no-pty",cid,"--","python3","-c",py],
                           capture_output=True, text=True, timeout=50, env=env)
        lines = [L.strip() for L in (p.stdout or "").splitlines() if L.strip()]
        return [x for x in (lines[0].split(",") if lines else []) if x.startswith("a-adamw-")]
    except Exception:
        return []

def mid_n_refresh():
    if MID.exists():
        # script may cd to wrong store path; run pieces locally
        for script in ("p4-fill-appendix-from-media.py", "p4-update-lambda-plot.py", "p4-score-from-media.py"):
            sp = STORE / "internal" / script
            if sp.exists():
                subprocess.run([sys.executable, str(sp)], cwd=str(STORE), timeout=180, check=False)
        # also try shell if it works with STORE path
        subprocess.run(["bash", str(MID)], cwd=str(STORE), timeout=180, check=False)

print(f"WORKER_POLL_N12_START cwd={os.getcwd()} baseline={baseline}", flush=True)
(ROOT / ".local_monitors/batch3_longpoll.pid").write_text(str(os.getpid()))

last_reported = baseline
while time.time() < deadline:
    if os.getcwd() != str(ROOT) or "assignments-a3-batch1" in os.getcwd():
        raise SystemExit("FORBIDDEN_CWD")
    got = fetch([j["name"] for j in a])
    copied_new, lam = [], {}
    for name in got:
        src = RESULTS_DIR / f"{name}.json"
        if not src.exists() or not str(src).startswith(str(ROOT)+"/"):
            continue
        dst = STAGE_A / f"{name}.json"
        is_new = not dst.exists()
        shutil.copy2(src, dst)
        shutil.copy2(src, MEDIA / f"{name}.json")
        r = json.loads(src.read_text())
        lam[name] = r.get("lambda_max")
        summary = json.dumps({"job": name, "lambda_max": r.get("lambda_max"), "lambda_min": r.get("lambda_min"),
                              "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n"
        (STAGE_A / f"{name}.summary.json").write_text(summary)
        (MEDIA / f"{name}.summary.json").write_text(summary)
        if is_new:
            copied_new.append(name)
            print("NEW_JSON", name, "lambda_max=", r.get("lambda_max"), flush=True)

    files = _volume_results()
    done = [j["name"] for j in a if _filename(j) in files]
    miss = [j["name"] for j in a if _filename(j) not in files]
    done_set = set(done)
    apps = json.loads(strip(subprocess.check_output(["modal","app","list","--env","cs312-daxton","--json"], text=True, timeout=60)))
    b3 = next(x for x in apps if (x.get("app_id") or x.get("App ID")) == APP)
    tasks = b3.get("tasks") if b3.get("tasks") is not None else b3.get("Tasks")
    state = b3.get("state") or b3.get("State")
    conts = json.loads(strip(subprocess.check_output(["modal","container","list","--env","cs312-daxton","--json"], text=True, timeout=60)))
    cids = [c.get("Container ID") or c.get("container_id") for c in conts if (c.get("App ID") or c.get("app_id")) == APP]

    utils, gpu_rows, live_expect = [], [], []
    for cid in cids:
        util, mem = smi_one(cid)
        jobs = discover_jobs(cid)
        pending = [j for j in jobs if j not in done_set]
        live = pending[-1] if pending else None
        if util is not None:
            utils.append(util)
        if live and live not in live_expect:
            live_expect.append(live)
        gpu_rows.append({"id": cid, "gpu_util": util, "mem": mem, "live_job": live,
                         "jobs_seen": jobs, "pending_seen": pending})
    if len(live_expect) < len(cids):
        for m in miss:
            if m not in live_expect:
                live_expect.append(m)
            if len(live_expect) >= max(len(cids), 1):
                break
    live_expect = [x for x in live_expect if x not in done_set]

    n = len(done)
    increased = n > last_reported
    complete = not miss
    try: tasks_i = int(str(tasks))
    except Exception: tasks_i = -1
    busy = sum(1 for u in utils if u >= 50)
    if utils and len(utils) == len(cids) and all(u >= 50 for u in utils):
        gpu_status = "BUSY"
    elif busy:
        gpu_status = "PARTIAL"
    elif cids and utils and all(u <= 2 for u in utils):
        gpu_status = "IDLE?"
    elif cids and not utils:
        gpu_status = "UNKNOWN_SMI_FAIL"
    elif not cids:
        gpu_status = "NO_CONTAINERS"
    else:
        gpu_status = "UNKNOWN_SMI_FAIL"

    prior_lam = {}
    if PROG.exists():
        try: prior_lam = json.loads(PROG.read_text()).get("lambda_max") or {}
        except Exception: pass
    lam_all = dict(prior_lam)
    for name in done:
        if name in lam: lam_all[name] = lam[name]
        elif name not in lam_all:
            p = STAGE_A / f"{name}.json"
            if p.exists():
                try: lam_all[name] = json.loads(p.read_text()).get("lambda_max")
                except Exception: pass

    now = time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    still_running = n < 13 and (busy > 0 or tasks_i > 0 or bool(cids))
    prog = {
        "n": n, "of": 13, "remaining": len(miss), "baseline": baseline,
        "increased": n > baseline, "complete": complete, "done": done, "miss": miss,
        "copied_new": copied_new, "lambda_max": lam_all,
        "tasks": tasks, "containers": gpu_rows, "gpu_utils": utils, "gpu_status": gpu_status,
        "live_expect": live_expect, "still_running": still_running,
        "ts": now, "timer": "worker-poll-n6",
        "checkout": str(ROOT), "forbidden_used": False,
    }
    PROG.write_text(json.dumps(prog, indent=2) + "\n")
    LOCAL.write_text(json.dumps({
        "phase": "stage_a" if miss else "stage_a_complete",
        "done": done, "miss": miss, "tasks": tasks, "state": state, "containers": cids,
        "gpu_utils": utils, "gpu_status": gpu_status, "live_expect": live_expect,
        "copied_this_tick": copied_new, "ts": time.time(),
    }, indent=2) + "\n")
    print(f"tick N={n}/13 utils={utils} live={live_expect} tasks={tasks} copied={copied_new}", flush=True)

    # NEVER relaunch stage-a while GPUs busy / still_running
    if increased or complete:
        # mid-N when new DONE beyond 305 (n>6)
        if n > 6:
            print("MID_N_REFRESH", n, flush=True)
            mid_n_refresh()
        reason = "complete" if complete else "n_jump"
        payload = {
            "reason": reason, "from": last_reported, "n": n, "done": done, "miss": miss,
            "copied_new": copied_new,
            "lambda_max": {k: lam_all.get(k) for k in done},
            "new_lambda_max": {k: lam.get(k) for k in copied_new},
            "live_expect": live_expect, "tasks": tasks, "gpu_utils": utils,
            "gpu_status": gpu_status, "ts": now, "checkout": str(ROOT), "forbidden_used": False,
        }
        RESULT.write_text(json.dumps(payload, indent=2) + "\n")
        print("EXIT_EVENT", reason, flush=True)
        print("FINAL", json.dumps(payload), flush=True)
        last_reported = n
        if complete:
            subprocess.run(["bash", str(ROOT / ".local_monitors/ensure_batch3_monitors.sh")],
                           cwd=str(ROOT), timeout=60, check=False)
            break
        # continue toward 13/13 after n_jump
    time.sleep(interval)
else:
    payload = {"reason": "timeout", "n": last_reported, "checkout": str(ROOT),
               "forbidden_used": False, "ts": time.strftime("%Y-%m-%d %H:%M PDT")}
    RESULT.write_text(json.dumps(payload, indent=2) + "\n")
    print("EXIT_EVENT timeout", flush=True)
    print("FINAL", json.dumps(payload), flush=True)

print("WORKER_POLL_N6_DONE", flush=True)
