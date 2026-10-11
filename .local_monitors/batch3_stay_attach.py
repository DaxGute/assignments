#!/usr/bin/env python3
import json, time, subprocess, os, sys
from pathlib import Path
ROOT=Path("/Users/daxtongutekunst/Desktop/assignments")
os.chdir(ROOT)
assert os.getcwd()==str(ROOT)
OUT=Path("/tmp/a3-batch3-final-9375.out")
PROG=Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files/media/p4/stage_a_progress.json")
STORE=Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
PYB=str(ROOT/".venv/bin/python")
print("STAY_ATTACHED_V2_START", flush=True)
deadline=time.time()+3*3600
while time.time()<deadline:
    for name,log in [("batch3_final_9375","batch3_final_9375.log"),("longpoll_n4","batch3_longpoll.log"),("batch3_stage_pipeline","batch3_stage_pipeline.log")]:
        if subprocess.call(["pgrep","-f",f"/Desktop/assignments/.local_monitors/{name}.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)!=0:
            if name=="batch3_stage_pipeline":
                (ROOT/".local_monitors/batch3_stage_pipeline.lock").unlink(missing_ok=True)
            subprocess.Popen([PYB,"-u",str(ROOT/".local_monitors"/f"{name}.py")], cwd=str(ROOT), stdout=open(ROOT/".local_monitors"/log,"a"), stderr=subprocess.STDOUT, start_new_session=True)
            print("RESPAWN", name, flush=True)
    if OUT.exists():
        d=json.loads(OUT.read_text())
        print("DONE_SIGNAL", json.dumps(d), flush=True)
        (STORE/"internal/p4-apply-ready-signal.md").write_text("---\ncursor:\n  subagentId: \"bc-8a994472-d1f4-5aa9-a973-7602cdc5af8b\"\n---\n\n# P4 --apply ready\n\n"+json.dumps(d,indent=2)+"\n")
        print("P4_APPLY_READY_SIGNAL_WRITTEN", flush=True)
        raise SystemExit(0)
    try:
        p=json.loads(PROG.read_text())
        if int(p.get("n") or 0)>=13 or p.get("complete"):
            print("PROGRESS_HIT_13_WAITING_OUT", flush=True)
    except Exception:
        pass
    time.sleep(20)
print("STILL_WAITING_TIMEOUT", flush=True)
raise SystemExit(2)
