#!/usr/bin/env python3
import json, os, re, shutil, subprocess, time
from pathlib import Path

ROOT = Path("/Users/daxtongutekunst/Desktop/assignments")
os.chdir(ROOT)
assert os.getcwd() == str(ROOT)
STORE = Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
RESULTS = ROOT / "experiments/a3_optimization/results/a3-inside-the-hessian"
STAGE_B = STORE / "media/p4/stage-b"
STAGE_C = STORE / "media/p4/stage-c"
MEDIA = STORE / "media/p4"
STAGE_B.mkdir(parents=True, exist_ok=True)
STAGE_C.mkdir(parents=True, exist_ok=True)
STATUS = STORE / "internal/batch3-bc-status.md"
AUTOFILL = STORE / "internal/p4-bc-score-and-apply.sh"
INVAR_PY = STORE / "internal/p4-score-invar-from-media.py"
C_PY = STORE / "internal/p4-score-c-from-media.py"
APP_C = "ap-fITMulBWHgvvL54utzP8Wc"
APP_B = "ap-lSeukZDR7yWisgODr61i1k"
HEADS = ["b-rescale-head-x0.25", "b-rescale-head-x0.5", "b-rescale-head-x2"]
C_NAMES = ["c-full", "c-top", "c-removed"]
FLAG = ROOT / ".local_monitors/first_head_arm.flag"
PIPE = ROOT / ".local_monitors/batch3_stage_pipeline.log"
ALIVE = ROOT / ".local_monitors/dual_bc_wait.alive"

def strip(s): return re.sub(r"\x1b\[[0-9;]*m", "", s)

def partial_rows(c_part):
    """Return markdown rows with volume partial byte sizes + mtimes."""
    import sys
    from pathlib import PurePosixPath
    from datetime import datetime
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from modal_utils import user_volume
    from experiments.a3_optimization.hessian_jobs import VOLUME_RESULTS
    meta = {}
    try:
        for e in user_volume.listdir(VOLUME_RESULTS):
            name = PurePosixPath(e.path).name
            meta[name] = (int(e.size), int(e.mtime))
    except Exception:
        pass
    lines = []
    now = time.time()
    for n in ("c-full.jsonl.partial", "c-top.jsonl.partial", "c-removed.jsonl.partial"):
        if n in meta:
            size, mt = meta[n]
            mtime_s = datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
            age = f"{(now-mt)/60:.1f} min"
            lines.append(f"| `{n}` | **{size}** | {mtime_s} | {age} |")
        elif n in c_part:
            lines.append(f"| `{n}` | **?** | present | — |")
        else:
            lines.append(f"| `{n}` | — | not present | — |")
    return "\n".join(lines)


def vol():
    out = subprocess.check_output(
        [str(ROOT / ".venv/bin/python"), "-c",
         "from experiments.a3_optimization.hessian_jobs import _volume_results; print('\\n'.join(sorted(_volume_results())))"],
        cwd=str(ROOT), text=True, timeout=90)
    return [x for x in out.splitlines() if x.strip()]

def app_row(app_id):
    raw = strip(subprocess.check_output(["modal", "app", "list", "--env", "cs312-daxton"], text=True, timeout=60))
    for L in raw.splitlines():
        if app_id in L:
            return L.strip()
    return "missing"

def tasks_from_row(row):
    parts = [p.strip() for p in row.split("│") if p.strip()]
    return parts[3] if len(parts) >= 4 else "?"

def state_from_row(row):
    parts = [p.strip() for p in row.split("│") if p.strip()]
    return parts[2] if len(parts) >= 3 else "?"

def fetch():
    from experiments.a3_optimization.hessian_jobs import fetch as F
    return F(None)

def copy_b(names):
    copied = []
    for n in names:
        src = RESULTS / f"{n}.json"
        if src.exists():
            shutil.copy2(src, STAGE_B / src.name)
            shutil.copy2(src, MEDIA / src.name)
            copied.append(src.name)
    return copied

def copy_c(names):
    copied = []
    for n in names:
        for ext in (".jsonl", ".json"):
            src = RESULTS / f"{n}{ext}"
            if src.exists() and not src.name.endswith(".partial"):
                shutil.copy2(src, STAGE_C / src.name)
                shutil.copy2(src, MEDIA / src.name)
                copied.append(src.name)
    return copied

def run_autofill():
    if AUTOFILL.exists():
        print("AUTOFILL_SH", flush=True)
        r = subprocess.run(["bash", str(AUTOFILL)], cwd=str(STORE), capture_output=True, text=True, timeout=300)
        print((r.stdout or "")[-2000:], flush=True)
        if r.stderr:
            print("AUTOFILL_ERR", (r.stderr or "")[-800:], flush=True)
        print("AUTOFILL_EC", r.returncode, flush=True)
        return r.returncode
    print("AUTOFILL_FALLBACK", flush=True)
    for py in (INVAR_PY, C_PY):
        if py.exists():
            r = subprocess.run(["python3", str(py)], cwd=str(STORE), capture_output=True, text=True, timeout=300)
            print(py.name, "ec", r.returncode, flush=True)
    return 0

def refresh(heads, c_done, c_part, extra=""):
    """Watch (bc_watch6) owns batch3-bc-status.md — dual only heartbeats."""
    now = time.strftime("%Y-%m-%d %H:%M PDT")
    ALIVE.write_text(now + "\n")
    print("DUAL_ALIVE", now, "heads=", heads, "c=", c_done, c_part, extra, flush=True)

def main():
    print("DUAL_WAIT_FILE", os.getcwd(), flush=True)
    seen = set()
    c_complete = False
    first = FLAG.exists()
    autofilled = set()
    deadline = time.time() + 4 * 3600
    last = ""
    last_st = 0
    while time.time() < deadline:
        files = vol()
        heads = [h for h in HEADS if f"{h}.json" in files]
        c_done = [n for n in C_NAMES if f"{n}.jsonl" in files or f"{n}.json" in files]
        c_part = [f for f in files if f.startswith("c-") and f.endswith(".partial")]
        new = [h for h in heads if h not in seen]
        if new:
            fetch()
            copied = copy_b(new + ["b-rescale-reference"])
            for h in new:
                seen.add(h)
                p = RESULTS / f"{h}.json"
                d = json.loads(p.read_text()) if p.exists() else {}
                print("HEAD_ARM_LANDED", h, d.get("scale"), d.get("lambda_max"), copied, flush=True)
                if not first:
                    FLAG.write_text(json.dumps({
                        "first": h, "scale": d.get("scale"), "lambda_max": d.get("lambda_max"),
                        "probe_loss": d.get("probe_loss"),
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "app": APP_B,
                    }, indent=2))
                    first = True
                    print("FIRST_HEAD_ARM", h, flush=True)
            # invar already scored from x0.25 — fetch/copy only for later heads
            refresh(heads, c_done, c_part, f"fetched heads {new}; stage-b copy only (no invent/rescore)")
            last_st = time.time()
        log = PIPE.read_text(errors="ignore") if PIPE.exists() else ""
        if (len(c_done) >= 3 or "PIPELINE COMPLETE" in log[-20000:]) and not c_complete:
            fetch()
            copied = copy_c(C_NAMES)
            c_complete = True
            print("C_FETCHED", c_done, copied, flush=True)
            run_autofill()
            refresh(heads, c_done, c_part, f"c fetched/copied={copied}; autofill")
            last_st = time.time()
        if time.time() - last_st > 120:
            refresh(heads, c_done, c_part, "waiting")
            last_st = time.time()
        msg = f"heads={heads} c={c_done}/{c_part} | {app_row(APP_B)} | {app_row(APP_C)}"
        if msg != last:
            print(time.strftime("%H:%M:%S"), msg, flush=True)
            last = msg
        if len(seen) >= 3 and c_complete:
            print("DUAL_COMPLETE", flush=True)
            return 0
        # keep pipeline without double-launch
        try:
            text = (ROOT / ".local_monitors/batch3_stage_pipeline.lock").read_text()
            m = re.search(r"pid=(\d+)", text)
            if m:
                os.kill(int(m.group(1)), 0)
        except ProcessLookupError:
            r = subprocess.run(["pgrep", "-f", "/Desktop/assignments/.local_monitors/batch3_stage_pipeline.py"], capture_output=True, text=True)
            if not (r.stdout or "").strip():
                print("RESPAWN_PIPELINE_ONCE", flush=True)
                (ROOT / ".local_monitors/batch3_stage_pipeline.lock").unlink(missing_ok=True)
                subprocess.Popen(
                    [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / ".local_monitors/batch3_stage_pipeline.py")],
                    cwd=str(ROOT),
                    stdout=open(ROOT / ".local_monitors/batch3_stage_pipeline.log", "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        time.sleep(45)
    print("DUAL_TIMEOUT", last, flush=True)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
