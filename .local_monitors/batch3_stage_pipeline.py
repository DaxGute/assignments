#!/usr/bin/env python3
"""Monitor Batch 3: a → fetch → b-rescale → fetch → c → fetch.

Checkout: /Users/daxtongutekunst/Desktop/assignments ONLY.
Does not invent Hessian numbers; only records volume/fetch counts and Modal app IDs.
Pipeline order matches internal/batch3-stage-plan.md (do not skip; fetch between GPU stages).
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

ROOT = "/Users/daxtongutekunst/Desktop/assignments"
STORE = Path(
    "/Users/daxtongutekunst/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/files"
)
STATUS = STORE / "internal/mac-gpu-status.md"
PREP = STORE / "internal/batch3-prep.md"
MEDIA_STAGE_A = STORE / "media/p4/stage-a"
PROGRESS = Path(ROOT) / ".local_monitors/batch3_progress.json"
LOG = Path(ROOT) / ".local_monitors/batch3_stage_pipeline.log"
LOCK = Path(ROOT) / ".local_monitors/batch3_stage_pipeline.lock"
APP_A = "ap-DAUDZsFRech705uMDN7q3x"
INTERVAL = 120
TIMEOUT = 10 * 3600

os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("MODAL_ENVIRONMENT", "cs312-daxton")

from experiments.a3_optimization import p4_inside_the_hessian as p4
from experiments.a3_optimization.hessian_jobs import RESULTS_DIR, _filename, _volume_results, fetch


def log(msg: str) -> None:
    """Always persist to LOG (sandbox may redirect stdout to a unix socket)."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    # Prefer direct file write; also print when stdout is a TTY (interactive debug).
    try:
        with LOG.open("a") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass
    try:
        if sys.stdout.isatty():
            print(line, flush=True)
    except Exception:
        pass


def acquire_lock() -> object:
    """Single-instance lock so 13/13 advance cannot double-launch b-rescale/c."""
    import fcntl

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = LOCK.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(f"another batch3_stage_pipeline holds {LOCK}")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} ts={time.time()}\n")
    fh.flush()
    return fh


def modal_apps():
    out = subprocess.check_output(
        ["modal", "app", "list", "--env", "cs312-daxton", "--json"], text=True, timeout=60
    )
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def _norm_app(a: dict) -> dict:
    """Normalize modal --json keys (Title Case vs snake_case across CLI versions)."""
    return {
        "app_id": a.get("app_id") or a.get("App ID"),
        "description": a.get("description") or a.get("Description") or "",
        "state": a.get("state") or a.get("State") or "unknown",
        "tasks": a.get("tasks") if a.get("tasks") is not None else a.get("Tasks", "?"),
    }


def app_info(app_id: str):
    for a in modal_apps():
        n = _norm_app(a)
        if n.get("app_id") == app_id:
            return n
    # fallback: plain text table
    try:
        plain = subprocess.check_output(
            ["modal", "app", "list", "--env", "cs312-daxton"], text=True, timeout=60
        )
    except Exception:
        return None
    if app_id not in plain:
        return None
    idx = plain.find(app_id)
    chunk = plain[idx : idx + 300]
    tasks = "?"
    tm = re.search(r"│\s*(\d+)\s*│", chunk)
    if tm:
        tasks = tm.group(1)
    state = "unknown"
    if "ephemeral" in chunk:
        state = "ephemeral (detached)"
    elif "stopped" in chunk:
        state = "stopped"
    name = ""
    nm = re.search(r"(dl[-_]alchemy[^\s│]+)", chunk)
    if nm:
        name = nm.group(1)
    return {"app_id": app_id, "description": name, "state": state, "tasks": tasks}


def stage_status(stage: str):
    jobs = p4.STAGES.get(stage, [])
    done_files = _volume_results()
    done = [j["name"] for j in jobs if _filename(j) in done_files]
    miss = [j["name"] for j in jobs if _filename(j) not in done_files]
    partial = sorted(f for f in done_files if f.endswith(".partial"))
    return done, miss, partial, done_files


def stage_a_media_names() -> list[str]:
    MEDIA_STAGE_A.mkdir(parents=True, exist_ok=True)
    # only real job JSON (`a-adamw-3.json`), not `*.summary.json`
    return sorted(
        p.stem
        for p in MEDIA_STAGE_A.glob("a-adamw-*.json")
        if p.name.count(".") == 1
    )


def copy_new_stage_a(done: list[str]) -> list[str]:
    """Fetch newly finished stage-a jobs and copy JSON into media/p4/stage-a/. No invent."""
    MEDIA_STAGE_A.mkdir(parents=True, exist_ok=True)
    already = set(stage_a_media_names())
    need = [n for n in done if n not in already]
    if not need:
        return []
    got = fetch(need)
    copied = []
    for name in got:
        src = RESULTS_DIR / f"{name}.json"
        if not src.exists():
            continue
        dest = MEDIA_STAGE_A / f"{name}.json"
        shutil.copy2(src, dest)
        # keep sibling mirror under media/p4/ for older readers
        shutil.copy2(src, STORE / "media/p4" / f"{name}.json")
        copied.append(name)
        log(f"COPIED media/p4/stage-a/{name}.json bytes={src.stat().st_size}")
    return copied


def update_status(phase: str, *, done_a, miss_a, partial, tasks, state, name, extra=""):
    now = time.strftime("%Y-%m-%d %H:%M PDT", time.localtime())
    media = stage_a_media_names()
    media_cell = ", ".join(f"`media/p4/stage-a/{x}.json`" for x in media) or "_none_"
    # Keep phase free of nested ** so heading replace stays unambiguous.
    phase_safe = phase.replace("**", "")
    block = f"""## Batch 3 — P4 Hessian — **{phase_safe}** (refreshed {now})

| Field | Value |
|-------|--------|
| Checkout | `/Users/daxtongutekunst/Desktop/assignments` ONLY |
| Stage `a` Modal app | `{name}` / **`{APP_A}`** |
| App URL | https://modal.com/apps/cs312-f26/cs312-daxton/{APP_A} |
| App state | {state} |
| Modal tasks (stage-a app) | **{tasks}** |
| Stage `a` volume results | **{len(done_a)} / {len(done_a)+len(miss_a)}** |
| Stage `a` done | {", ".join(f"`{x}`" for x in done_a) or "_none_"} |
| Stage `a` pending | {", ".join(f"`{x}`" for x in miss_a) or "_none_"} |
| media/p4/stage-a | {media_cell} |
| Partial files | {", ".join(f"`{x}`" for x in partial) or "_none_"} |
| Hessian metrics | **not inventing** — no full P4-a-* score until series lands |
| Pipeline | a → fetch → b-rescale → fetch → c → fetch |
| Extra | {extra or "—"} |
| Prep | [`batch3-prep.md`](batch3-prep.md) |

"""
    text = STATUS.read_text()
    # Match through end of Batch-3 section; tolerate nested ** inside phase titles.
    pat = r"## Batch 3 — P4 Hessian[\s\S]*?(?=\n## |\Z)"
    if re.search(pat, text):
        new, n = re.subn(pat, block + "\n", text, count=1)
        if n != 1:
            log(f"status Batch 3 replace failed n={n}; appending")
            new = text.rstrip() + "\n\n" + block
    else:
        new = text.rstrip() + "\n\n" + block
    # keep Batch 2 pointer accurate (phase may contain markdown; keep short)
    short = f"RUNNING stage a {len(done_a)}/{len(done_a)+len(miss_a)}" if miss_a else phase
    new = re.sub(
        r"\| Batch 3 \|[^\n]*\|",
        f"| Batch 3 | **{short}** — `{APP_A}`; stage a {len(done_a)}/{len(done_a)+len(miss_a)} |",
        new,
        count=1,
    )
    STATUS.write_text(new)


def run_stage(stage: str) -> int:
    log(f"EXECUTE stage={stage}")
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "experiments.a3_optimization.launch_batch3",
            "--execute",
            "--stage",
            stage,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    (STORE / f"internal/batch3-{stage}-launch.log").write_text(
        proc.stdout + "\n---STDERR---\n" + proc.stderr
    )
    log(f"STAGE_RC {stage}={proc.returncode}")
    m = re.search(r"https://modal.com/apps/[^\s]+/(ap-[A-Za-z0-9]+)", proc.stdout)
    if m:
        log(f"STAGE_APP {stage}={m.group(1)}")
    if proc.stdout:
        log("STDOUT_TAIL " + proc.stdout[-1500:].replace("\n", " | "))
    if proc.stderr:
        log("STDERR_TAIL " + proc.stderr[-800:].replace("\n", " | "))
    return proc.returncode


def run_fetch() -> list:
    log("EXECUTE fetch")
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "experiments.a3_optimization.launch_batch3",
            "--stage",
            "fetch",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    (STORE / "internal/batch3-fetch.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    log(f"FETCH_RC={proc.returncode} out={proc.stdout.strip()[-500:]}")
    # stdout should print the list
    try:
        # fetch prints a python list
        text = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "[]"
        got = eval(text, {"__builtins__": {}}) if text.startswith("[") else []
    except Exception:
        got = []
    return got


def wait_stage_a():
    start = time.time()
    last_n = -1
    while True:
        done, miss, partial, _ = stage_status("a")
        info = app_info(APP_A) or {}
        tasks = info.get("tasks", "?")
        state = info.get("state", "unknown")
        name = info.get("description", "dl-alchemy-a3-hessian")
        copied = []
        if len(done) != last_n:
            # volume N increased (or first observation): fetch+copy new JSONs
            try:
                copied = copy_new_stage_a(done)
            except Exception as e:
                log(f"COPY_ERR {e!r}")
        media = stage_a_media_names()
        PROGRESS.write_text(
            json.dumps(
                {
                    "phase": "stage_a",
                    "done": done,
                    "miss": miss,
                    "partial": partial,
                    "tasks": tasks,
                    "state": state,
                    "media_stage_a": media,
                    "copied_this_tick": copied,
                    "ts": time.time(),
                },
                indent=2,
            )
        )
        extra = (
            "13/13 volume JSON present; starting fetch→b-rescale→fetch→c→fetch"
            if not miss
            else (f"copied {copied}" if copied else f"media/p4/stage-a has {len(media)} file(s)")
        )
        # refresh status every loop so task count stays current
        update_status(
            "RUNNING stage a" if miss else "STAGE a COMPLETE — fetching",
            done_a=done,
            miss_a=miss,
            partial=partial,
            tasks=tasks,
            state=state,
            name=name,
            extra=extra,
        )
        if len(done) != last_n:
            last_n = len(done)
            log(f"stage_a {len(done)}/{len(done)+len(miss)} tasks={tasks} state={state} media={len(media)}")
        else:
            log(f"still stage_a {len(done)}/{len(done)+len(miss)} tasks={tasks}")
        if not miss:
            # final copy sweep before advancing pipeline
            copy_new_stage_a(done)
            return done
        if time.time() - start > TIMEOUT:
            raise SystemExit("TIMEOUT waiting stage a")
        time.sleep(INTERVAL)


def append_prep(note: str):
    text = PREP.read_text()
    stamp = time.strftime("%Y-%m-%d %H:%M %Z")
    text = text.rstrip() + f"\n\n### Pipeline progress ({stamp})\n\n{note}\n"
    PREP.write_text(text)


def active_hessian_apps():
    out = []
    for a in modal_apps():
        n = _norm_app(a)
        desc = (n.get("description") or "").lower()
        state = (n.get("state") or "").lower()
        if "hessian" in desc and "ephemeral" in state:
            out.append(n)
    return out


def wait_stage_volume(stage: str, done_a: list) -> None:
    """Block until all jobs for `stage` have volume results (do not skip)."""
    jobs = p4.STAGES.get(stage) or []
    if not jobs:
        log(f"{stage}: no jobs in starter STAGES (noop)")
        return
    while True:
        d, m, p, _ = stage_status(stage)
        active = active_hessian_apps()
        tasks = sum(int(a.get("tasks") or 0) for a in active) if active else 0
        update_status(
            f"RUNNING {stage}",
            done_a=done_a,
            miss_a=[],
            partial=p,
            tasks=tasks,
            state=active[0]["state"] if active else "n/a",
            name=active[0]["description"] if active else "",
            extra=f"{stage} volume {len(d)}/{len(d)+len(m)}; done={d}; miss={m}",
        )
        log(f"wait {stage} {len(d)}/{len(d)+len(m)} tasks={tasks}")
        if not m:
            return
        time.sleep(INTERVAL)


def main():
    if os.getcwd() != ROOT:
        raise SystemExit("wrong cwd")
    if "assignments-a3-batch1" in os.getcwd():
        raise SystemExit("forbidden checkout assignments-a3-batch1")
    lock_fh = acquire_lock()
    log(f"START cwd={os.getcwd()} pid={os.getpid()} lock={LOCK}")
    # Never re-submit stage a; only wait for volume 13/13 then advance.
    done_a = wait_stage_a()
    got = run_fetch()
    log(f"FETCHED_COUNT={len(got)} names={got}")
    append_prep(
        f"- Stage a complete ({len(done_a)}/13). Fetch returned {len(got)} names "
        "(no Hessian numbers invented)."
    )

    # Per batch3-stage-plan.md: b-rescale → fetch → c → fetch (do not skip).
    for stage in ("b-rescale", "c"):
        done, miss, partial, _ = stage_status(stage)
        info = app_info(APP_A) or {}
        update_status(
            f"LAUNCHING {stage}",
            done_a=done_a,
            miss_a=[],
            partial=partial,
            tasks=info.get("tasks", "?"),
            state=info.get("state", "?"),
            name=info.get("description", ""),
            extra=f"about to execute `{stage}` (pending={miss}; must not skip)",
        )
        # Skip re-execute if volume already complete (idempotent resume).
        if not miss:
            log(f"SKIP_EXECUTE {stage}: volume already complete {done}")
        else:
            rc = run_stage(stage)
            append_prep(f"- Executed `{stage}` rc={rc} (required; not skipped).")
            if rc != 0:
                update_status(
                    f"FAILED at {stage}",
                    done_a=done_a,
                    miss_a=[],
                    partial=partial,
                    tasks=info.get("tasks", "?"),
                    state=info.get("state", "?"),
                    name=info.get("description", ""),
                    extra=f"rc={rc}; see internal/batch3-{stage}-launch.log",
                )
                return 3
        wait_stage_volume(stage, done_a)
        mid = run_fetch()
        log(f"FETCH_AFTER_{stage} count={len(mid)}")
        append_prep(f"- Fetch after `{stage}`: {len(mid)} names (no invented Hessian numbers).")

    info = app_info(APP_A) or {}
    update_status(
        "PIPELINE COMPLETE (a→fetch→b-rescale→fetch→c→fetch)",
        done_a=done_a,
        miss_a=[],
        partial=[],
        tasks=info.get("tasks", 0),
        state=info.get("state", "?"),
        name=info.get("description", ""),
        extra="local results/a3-inside-the-hessian/; unlock P4-a-* / P4-invar-dL / P4-top10-frac / P4-c-loss-rank after JSON read; do not invent",
    )
    log("DONE_PIPELINE")
    try:
        lock_fh.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
