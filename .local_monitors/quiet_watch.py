import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path("/Users/daxtongutekunst/Desktop/assignments")
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT)
sys.path.insert(0, str(ROOT))
from experiments.a3_optimization.p4_inside_the_hessian import STAGES
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, _filename, _volume_results, fetch
STORE = Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
STATUS = STORE/"internal/mac-gpu-status.md"
STAGE_A = STORE/"media/p4/stage-a"; MEDIA = STORE/"media/p4"; STAGE_A.mkdir(parents=True, exist_ok=True)
PROG = STORE/"media/p4/stage_a_progress.json"
ALERT = Path("/tmp/a3-batch3-alert.json")
APP = "ap-DAUDZsFRech705uMDN7q3x"
a = STAGES["a"]
baseline = int(json.loads(PROG.read_text()).get("n") or 3) if PROG.exists() else 3
def strip(s): return re.sub(r"\x1b\[[0-9;]*m", "", s)
env = {**os.environ, "MODAL_ENVIRONMENT": "cs312-daxton"}
print(f"QUIET_WATCH start baseline={baseline}", flush=True)
while True:
    got = fetch([j["name"] for j in a]); copied_new=[]; lam={}
    for name in got:
        src = RESULTS_DIR/f"{name}.json"
        if not src.exists(): continue
        dst = STAGE_A/f"{name}.json"; is_new = not dst.exists()
        shutil.copy2(src, dst); shutil.copy2(src, MEDIA/f"{name}.json")
        r=json.loads(src.read_text()); lam[name]=r.get("lambda_max")
        s=json.dumps({"job":name,"lambda_max":r.get("lambda_max"),"lambda_min":r.get("lambda_min"),"ts":time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2)+"\n"
        (STAGE_A/f"{name}.summary.json").write_text(s); (MEDIA/f"{name}.summary.json").write_text(s)
        if is_new: copied_new.append(name)
    files=_volume_results(); done=[j["name"] for j in a if _filename(j) in files]; miss=[j["name"] for j in a if _filename(j) not in files]
    apps=json.loads(strip(subprocess.check_output(["modal","app","list","--env","cs312-daxton","--json"], text=True)))
    b3=next(x for x in apps if (x.get("app_id") or x.get("App ID"))==APP)
    tasks=b3.get("tasks") if b3.get("tasks") is not None else b3.get("Tasks")
    state=b3.get("state") or b3.get("State"); desc=b3.get("description") or b3.get("Description")
    conts=json.loads(strip(subprocess.check_output(["modal","container","list","--env","cs312-daxton","--json"], text=True)))
    cids=[c.get("Container ID") or c.get("container_id") for c in conts if (c.get("App ID") or c.get("app_id"))==APP]
    utils=[]
    for cid in cids:
        try:
            p=subprocess.run(["modal","container","exec",cid,"--","nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=18, env=env)
            m=re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", strip((p.stdout or "")+"\n"+(p.stderr or "")))
            if m: utils.append(int(m.group(1)))
        except Exception:
            pass
    n=len(done); complete=not miss; increased=n>baseline
    collapse = (n < 13) and ((not cids) or (utils and max(utils) < 20) or (tasks in (0,"0",None) and n < 13))
    now=time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    # always keep progress/status accurate quietly
    phase="STAGE a COMPLETE" if complete else (f"RUNNING stage a — **{n}/13** N↑" if increased else f"RUNNING stage a — **{n}/13** WAIT")
    gpu_md="; ".join(f"`{cid}` GPU={u}%" for cid,u in zip(cids, utils)) if utils else (", ".join(f"`{c}`" for c in cids) or "_none_")
    prog={"n":n,"of":13,"remaining":len(miss),"complete":complete,"done":done,"miss":miss,"copied_new":copied_new,"lambda_max":lam,"tasks":tasks,"gpu_utils":utils,"ts":now}
    PROG.write_text(json.dumps(prog, indent=2)+"\n")
    Path("/tmp/a3-batch3-progress.json").write_text(json.dumps(prog, indent=2))
    block=f"""## Batch 3 — P4 Hessian — **{phase}** (refreshed {now})

| Field | Value |
|-------|--------|
| Checkout | `/Users/daxtongutekunst/Desktop/assignments` ONLY |
| Modal app | `{desc}` / **`{APP}`** |
| App state | {state} |
| Modal tasks | **{tasks}** |
| GPU util | {gpu_md} |
| Stage `a` volume | **{n} / 13** (remaining **{len(miss)}**){" ⬆️ from "+str(baseline) if increased else ""} |
| Finished | {", ".join(f"`{x}`" for x in done)} |
| Pending | {", ".join(f"`{x}`" for x in miss) or "_none_"} |
| Copied new | {", ".join(f"`{x}`" for x in copied_new) or "_none_"} |
| Next | ping only on N↑ or GPU collapse; 13/13 → fetch → b-rescale → c |

"""
    text=STATUS.read_text(); new,nn=re.subn(r"## Batch 3 — P4 Hessian — \*\*[^*]+\*\*.*?(?=\n## |\Z)", block+"\n", text, count=1, flags=re.S)
    if nn==1:
        new=re.sub(r"\| Batch 3 \|[^\n]*\|", f"| Batch 3 | **{n}/13** {'COMPLETE' if complete else 'WAIT'}; tasks={tasks} |", new, count=1)
        STATUS.write_text(new)
    if increased:
        print(f"PING N_JUMP {baseline}->{n} new={copied_new} tasks={tasks}", flush=True)
        ALERT.write_text(json.dumps({"type":"n_jump","from":baseline,"to":n,"new":copied_new,"done":done,"ts":now}, indent=2))
        baseline=n
    if collapse:
        print(f"PING GPU_COLLAPSE n={n} tasks={tasks} utils={utils} cids={cids}", flush=True)
        ALERT.write_text(json.dumps({"type":"gpu_collapse","n":n,"tasks":tasks,"utils":utils,"cids":cids,"ts":now}, indent=2))
    if complete:
        print("PING COMPLETE_13_13", flush=True)
        ALERT.write_text(json.dumps({"type":"complete","n":13,"ts":now}, indent=2))
        subprocess.run([str(ROOT/".venv/bin/python"),"-u",str(ROOT/".local_monitors/batch3_stage_pipeline.py")])
        break
    time.sleep(180)  # quieter: 3 min
print("QUIET_WATCH end", flush=True)
