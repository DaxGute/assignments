#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, time
from pathlib import PurePosixPath
ROOT = "/Users/daxtongutekunst/Desktop/assignments"
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from modal.exception import NotFoundError
from modal_train import _model_volume_path
from modal_utils import user_volume
from train import training_run_name
from experiments.a3_optimization.p2_river_valley import RUNS as P2
from experiments.a3_optimization.p3_edge_of_stability import RUNS as P3
JOBS=list(P2)+list(P3)
PROGRESS=f"{ROOT}/.local_monitors/batch2_progress.json"
ALIVE=f"{ROOT}/.local_monitors/batch2_monitor.alive"
INTERVAL=180
TIMEOUT=8*3600

def snapshot():
    rows=[]; done=partial=missing=0
    for cfg in JOBS:
        name=training_run_name(cfg); path=_model_volume_path(name)
        try: have={PurePosixPath(e.path).name for e in user_volume.listdir(path)}
        except NotFoundError: have=set()
        complete={"config.json","model.pt"}.issubset(have)
        if complete: done+=1; st="DONE"
        elif have: partial+=1; st="PARTIAL"
        else: missing+=1; st="MISSING"
        rows.append((st,name,len(have)))
    return done,partial,missing,rows

def main():
    print("B2_MONITOR_START root=", ROOT, "cwd=", os.getcwd(), flush=True)
    assert os.getcwd()==ROOT
    start=time.time()
    while True:
        done,partial,missing,rows=snapshot()
        elapsed=int(time.time()-start)
        print(f"[{elapsed}s] done={done}/9 partial={partial} missing={missing}", flush=True)
        for st,name,n in rows:
            if st!="DONE": print(f"  {st} have={n} {name}", flush=True)
        payload=json.dumps({"done":done,"partial":partial,"missing":missing,"elapsed":elapsed,"rows":rows,"root":ROOT,"ts":time.time()})
        open(PROGRESS,"w").write(payload)
        open("/tmp/a3-batch2-progress.json","w").write(payload)
        open(ALIVE,"w").write(f"pid={os.getpid()} root={ROOT} done={done}/9 elapsed={elapsed}\n")
        if done==9:
            print("ALL_BATCH2_DONE", flush=True); print("READY_FOR_METRICS_AND_BATCH3", flush=True); return 0
        if elapsed>TIMEOUT:
            print("TIMEOUT", flush=True); return 2
        time.sleep(INTERVAL)
if __name__=="__main__":
    raise SystemExit(main())
