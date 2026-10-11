#!/usr/bin/env python3
"""Watch remaining b-heads (fetch/copy only) + c-removed DONE (score-and-apply). P5 HOLD."""
import json, os, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path("/Users/daxtongutekunst/Desktop/assignments")
os.chdir(ROOT)
assert os.getcwd() == str(ROOT)
sys.path.insert(0, str(ROOT))

STORE = Path("/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files")
STATUS = STORE / "internal/batch3-bc-status.md"
SCORE_APPLY = STORE / "internal/p4-bc-score-and-apply.sh"
ALIVE = ROOT / ".local_monitors/dual_bc_wait.alive"
SEEN = ROOT / ".local_monitors/bc_watch6_seen.json"
RESULTS_DIR = ROOT / "experiments/a3_optimization/results/a3-inside-the-hessian"
APP_C, APP_B = "ap-fITMulBWHgvvL54utzP8Wc", "ap-lSeukZDR7yWisgODr61i1k"
HEADS = ["b-rescale-head-x0.25", "b-rescale-head-x0.5", "b-rescale-head-x2"]
C_NAMES = ["c-full", "c-top", "c-removed"]
SUBAGENT = "bc-19d41b67-450b-5cfc-8551-e8c572def4e1"
CYCLES, SLEEP = 40, 90

from modal_utils import user_volume
from experiments.a3_optimization.hessian_jobs import VOLUME_RESULTS, _volume_results, fetch

def strip(s): return re.sub(r"\x1b\[[0-9;]*m", "", s)

def app_row(app_id):
    raw = strip(subprocess.check_output(["modal", "app", "list", "--env", "cs312-daxton"], text=True, timeout=60))
    for L in raw.splitlines():
        if app_id in L:
            return L.strip()
    return "missing"

def tasks_state(row):
    parts = [p.strip() for p in row.split("│") if p.strip()]
    return (parts[2] if len(parts) >= 3 else "?", parts[3] if len(parts) >= 4 else "?")

def to_int(x):
    try: return int(str(x).strip())
    except Exception: return -1

def dual_pid():
    r = subprocess.run(["pgrep", "-f", r"dual_bc_wait\.py"], capture_output=True, text=True)
    xs = [x for x in (r.stdout or "").split() if x.isdigit()]
    return xs[-1] if xs else "missing"

def watch_pid():
    r = subprocess.run(["pgrep", "-f", r"bc_watch6\.py"], capture_output=True, text=True)
    xs = [x for x in (r.stdout or "").split() if x.isdigit()]
    # prefer other than self
    me = str(os.getpid())
    xs = [x for x in xs if x != me] or xs
    return xs[-1] if xs else me

def ensure_dual():
    if dual_pid() != "missing":
        return dual_pid()
    log = open(ROOT / ".local_monitors/dual_bc_wait.log", "a")
    p = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / ".local_monitors/dual_bc_wait.py")],
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / ".local_monitors/dual_bc_wait.pid").write_text(str(p.pid) + "\n")
    time.sleep(1)
    return dual_pid()

def snap():
    meta = {}
    for e in user_volume.listdir(VOLUME_RESULTS):
        name = PurePosixPath(e.path).name
        meta[name] = {"size": int(e.size), "mtime": int(e.mtime)}
    return meta, sorted(_volume_results())

def fmt(name, meta, now_ts):
    m = meta.get(name)
    if not m: return "—", "not present", "—"
    return m["size"], datetime.fromtimestamp(m["mtime"]).strftime("%Y-%m-%d %H:%M:%S"), f'{(now_ts-m["mtime"])/60:.1f} min'

def load_seen():
    if SEEN.exists():
        try: return set(json.loads(SEEN.read_text()).get("heads", []))
        except Exception: pass
    # seed with already-fetched x0.25 if in media
    seeded = set()
    for h in HEADS:
        if (STORE / "media/p4/stage-b" / f"{h}.json").exists() or (RESULTS_DIR / f"{h}.json").exists():
            # only seed if present on volume too — handled later
            seeded.add(h)
    return seeded

def save_seen(seen):
    SEEN.write_text(json.dumps({"heads": sorted(seen), "ts": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))

def copy_heads(names):
    copied = []
    for base in (ROOT / "media/p4", STORE / "media/p4"):
        (base / "stage-b").mkdir(parents=True, exist_ok=True)
    for n in names:
        src = RESULTS_DIR / f"{n}.json"
        if src.exists():
            for d in (ROOT / "media/p4/stage-b", STORE / "media/p4/stage-b", ROOT / "media/p4", STORE / "media/p4"):
                shutil.copy2(src, d / src.name)
            copied.append(f"{src.name}:{src.stat().st_size}")
    return copied

def copy_c(names):
    copied = []
    for base in (ROOT / "media/p4", STORE / "media/p4"):
        (base / "stage-c").mkdir(parents=True, exist_ok=True)
    for n in names:
        for ext in (".jsonl", ".json"):
            src = RESULTS_DIR / f"{n}{ext}"
            if src.exists() and not src.name.endswith(".partial"):
                for d in (ROOT / "media/p4/stage-c", STORE / "media/p4/stage-c", ROOT / "media/p4", STORE / "media/p4"):
                    shutil.copy2(src, d / src.name)
                copied.append(f"{src.name}:{src.stat().st_size}")
    return copied

def run_score_and_apply():
    r = subprocess.run(["bash", str(SCORE_APPLY)], cwd=str(STORE), capture_output=True, text=True, timeout=600)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    draft = [ln for ln in out.splitlines() if any(k in ln for k in
             ("DRAFT", "HIT", "MISS", "INSUFFICIENT", "autofill", "apply_exit", "applied", "paste", "P4-c"))]
    return r.returncode, out, "\n".join(draft[-60:])

def write_status(cycle, meta, vol, rc, rb, st_c, t_c, st_b, t_b, prev, done_rm,
                 delta, flat_for, signal, verdict, c_done, c_part, heads, head_rows,
                 actions, apply_ec, draft_snip, note):
    now = time.strftime("%Y-%m-%d %H:%M PDT")
    now_ts = time.time()
    ALIVE.write_text(now + "\n")
    psz, pmt, page = fmt("c-removed.jsonl.partial", meta, now_ts)
    dsz, dmt, dage = fmt("c-removed.jsonl", meta, now_ts)
    body = f"""---
cursor:
  subagentId: "{SUBAGENT}"
---

# Batch 3 b/c status (Desktop)

Checkout: **`/Users/daxtongutekunst/Desktop/assignments` ONLY**  
Refreshed: **{now}**  
Watch cycle: **{cycle}/{CYCLES}** — **bc_watch6 owns this file**  
Monitor: **dual_bc_wait pid={dual_pid()}** (ALIVE only; no status overwrite)  
Signal: **{signal}**

## Modal apps

| Stage | App ID | State | Tasks |
|-------|--------|-------|------:|
| stage c | `{APP_C}` | {st_c} | **{t_c}** |
| hand-rescale | `{APP_B}` | {st_b} | **{t_b}** |

Rows: `{rc}`  
`{rb}`

## c-removed growth

| Field | Value |
|-------|------:|
| prior partial bytes | **{prev if prev is not None else '—'}** |
| partial bytes | **{psz}** |
| partial mtime | {pmt} |
| age | {page} |
| **Δ bytes** | **{delta if delta is not None else '—'}** |
| done `c-removed.jsonl` | {'**YES** '+str(dsz)+'B' if done_rm else 'no'} |

## Stage-c / stage-b volume

| File | Bytes | mtime | Age | State |
|------|------:|-------|----:|-------|
| `c-full.jsonl` | **{fmt('c-full.jsonl', meta, now_ts)[0]}** | {fmt('c-full.jsonl', meta, now_ts)[1]} | {fmt('c-full.jsonl', meta, now_ts)[2]} | {'DONE' if 'c-full.jsonl' in vol else '—'} |
| `c-top.jsonl` | **{fmt('c-top.jsonl', meta, now_ts)[0]}** | {fmt('c-top.jsonl', meta, now_ts)[1]} | {fmt('c-top.jsonl', meta, now_ts)[2]} | {'DONE' if 'c-top.jsonl' in vol else '—'} |
| `c-removed.jsonl` | **{dsz}** | {dmt} | {dage} | {'DONE' if done_rm else '—'} |
| `c-removed.jsonl.partial` | **{psz}** | {pmt} | {page} | {'PARTIAL' if 'c-removed.jsonl.partial' in vol else '—'} |

| Glob | Count | Names |
|------|------:|-------|
| `c-*` done | **{len(c_done)}**/3 | {', '.join(c_done) or '—'} |
| `c-*` partial | {len(c_part)} | {', '.join(c_part) or '—'} |
| `b-rescale-head-*` | **{len(heads)}**/3 | {', '.join(heads) or '—'} |

### Head arms

| File | Bytes | scale |
|------|------:|-------|
{chr(10).join(f'| `{fn}` | {sz if sz is not None else "—"} | `{scale}` |' for fn, sz, scale in head_rows)}

## Actions this cycle

{actions or '- (poll only)'}

- `p4-bc-score-and-apply.sh` exit: {apply_ec}
- Snip:
```
{draft_snip or '(none)'}
```

## Diagnosis

- Verdict: **{verdict}**
- Signal: **{signal}**
- Bytes delta: **{delta if delta is not None else '—'}**
- Note: {note}
- Invar: already DRAFT MISS from x0.25 (parent applying). Later heads → fetch/copy only.
- **P5 HOLD**. No invent.
"""
    STATUS.write_text(body)
    print(f"STATUS_WROTE cycle={cycle} signal={signal} delta={delta} partial={psz} heads={heads}", flush=True)

def main():
    print("WATCH6_START", os.getcwd(), "heads-copy + c-score-and-apply", flush=True)
    print("DUAL", ensure_dual(), flush=True)
    seen = load_seen()
    # mark x0.25 seen if on volume (already handled)
    c_scored = False
    prev_size = None
    last_change = time.time()
    note = "watching x0.5/x2 + c-removed"

    for cycle in range(1, CYCLES + 1):
        meta, vol = snap()
        rc, rb = app_row(APP_C), app_row(APP_B)
        st_c, t_c = tasks_state(rc)
        st_b, t_b = tasks_state(rb)
        tci, tbi = to_int(t_c), to_int(t_b)

        partial = meta.get("c-removed.jsonl.partial")
        done_meta = meta.get("c-removed.jsonl")
        done_rm = done_meta is not None or "c-removed.jsonl" in vol
        cur = partial["size"] if partial else (done_meta["size"] if done_meta else None)
        delta = None if (prev_size is None or cur is None) else (cur - prev_size)
        if delta is not None and delta > 0:
            last_change = time.time()
        flat_for = time.time() - last_change

        c_done = [n for n in C_NAMES if f"{n}.jsonl" in vol or f"{n}.json" in vol]
        c_part = [f for f in vol if f.startswith("c-") and f.endswith(".partial")]
        heads = [h for h in HEADS if f"{h}.json" in vol]
        # seed seen for heads already present at start of this run once
        if cycle == 1:
            for h in heads:
                if h not in seen and (STORE / "media/p4/stage-b" / f"{h}.json").exists():
                    seen.add(h)
            save_seen(seen)

        head_rows = []
        for h in HEADS:
            fn = f"{h}.json"
            if fn in vol:
                local = RESULTS_DIR / fn
                if not local.exists():
                    try:
                        local.write_bytes(b"".join(user_volume.read_file(f"{VOLUME_RESULTS}/{fn}")))
                    except Exception as e:
                        head_rows.append((fn, meta.get(fn, {}).get("size"), f"read_err:{e}"))
                        continue
                try:
                    scale = json.loads(local.read_text()).get("scale", {})
                except Exception as e:
                    scale = f"parse_err:{e}"
                head_rows.append((fn, meta.get(fn, {}).get("size"), scale))
            else:
                head_rows.append((fn, None, "absent"))

        stuck = (not done_rm) and tci == 0 and flat_for > 300 and cur is not None
        actions = []
        apply_ec = None
        draft_snip = "(none)"
        new_heads = [h for h in heads if h not in seen]

        # 1) new heads → fetch + stage-b copy only (no invent / no invar rescore)
        if new_heads:
            print("NEW_HEADS", new_heads, flush=True)
            try:
                got = fetch(None) or []
                actions.append(f"- fetch: {got}")
            except Exception as e:
                actions.append(f"- fetch_err: {e}")
            # ensure new files local
            for h in new_heads:
                fn = f"{h}.json"
                local = RESULTS_DIR / fn
                if not local.exists() and fn in vol:
                    local.write_bytes(b"".join(user_volume.read_file(f"{VOLUME_RESULTS}/{fn}")))
            copied = copy_heads(new_heads + (["b-rescale-reference"] if "b-rescale-reference.json" in vol else []))
            actions.append(f"- stage-b copy only: {copied}")
            for h in new_heads:
                seen.add(h)
            save_seen(seen)
            note = f"fetched heads {new_heads}; invar already MISS from x0.25 — no rescore"

        # 2) c-removed DONE → fetch + score-and-apply (P4-c)
        if done_rm and not c_scored:
            print("C_REMOVED_DONE score-and-apply", flush=True)
            try:
                got = fetch(None) or []
                actions.append(f"- c fetch: {got}")
            except Exception as e:
                actions.append(f"- c fetch_err: {e}")
            copied = copy_c(c_done)
            actions.append(f"- stage-c copy: {copied}")
            apply_ec, out, draft_snip = run_score_and_apply()
            actions.append(f"- score-and-apply ec={apply_ec}")
            print("SCORE_APPLY_EC", apply_ec, flush=True)
            print(draft_snip, flush=True)
            c_scored = True
            note = f"c-removed DONE; score-and-apply ec={apply_ec}"

        if done_rm and len(heads) >= 3 and c_scored:
            signal, verdict = "ADVANCED", "complete"
        elif done_rm or new_heads or (delta is not None and delta > 0) or heads:
            signal = "ADVANCED" if (done_rm or new_heads or (delta is not None and delta > 0)) else "STILL_WAITING"
            if heads and not done_rm and (delta is None or delta == 0) and not new_heads:
                signal = "STILL_WAITING" if not (delta and delta > 0) else "ADVANCED"
            if delta is not None and delta > 0:
                signal = "ADVANCED"
            if done_rm or new_heads:
                signal = "ADVANCED"
            verdict = "alive_writing" if (tci >= 1 or c_part) else ("c_done" if done_rm else "waiting")
        elif stuck:
            signal, verdict = "BLOCKED", "stuck_suspect"
        else:
            signal, verdict = "STILL_WAITING", "alive_writing" if (tci >= 1 or tbi >= 1) else "queued"

        write_status(cycle, meta, vol, rc, rb, st_c, t_c, st_b, t_b, prev_size, done_rm,
                     delta, flat_for, signal, verdict, c_done, c_part, heads, head_rows,
                     "\n".join(actions), apply_ec, draft_snip, note)

        print(json.dumps({
            "cycle": cycle, "partial": cur, "delta": delta, "done_rm": done_rm,
            "tasks_c": t_c, "tasks_b": t_b, "heads": heads, "new_heads": new_heads,
            "signal": signal, "c_scored": c_scored,
        }), flush=True)

        prev_size = cur
        if done_rm and c_scored and len(heads) >= 3:
            print("WATCH6_STOP all heads + c scored", flush=True)
            for name in ("p4-c-score-dryrun.md", "p4-b-score-dryrun.md"):
                p = STORE / "internal" / name
                if p.exists():
                    print(f"==== {name} ====", flush=True)
                    print(p.read_text()[:2500], flush=True)
            return 0
        # if c scored but heads incomplete, keep watching for heads
        if cycle < CYCLES:
            time.sleep(SLEEP)

    print("WATCH6_DONE_MAX_CYCLES", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
