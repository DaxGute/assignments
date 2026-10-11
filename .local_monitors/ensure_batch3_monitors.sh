#!/bin/bash
# Ensure Batch3 durable monitors on Desktop/assignments ONLY.
# Does NOT relaunch Modal stage a.
set -euo pipefail
ROOT=/Users/daxtongutekunst/Desktop/assignments
cd "$ROOT"
case "$PWD" in *assignments-a3-batch1*) echo "FORBIDDEN checkout"; exit 4;; esac
test "$PWD" = "$ROOT" || exit 1

PY="$ROOT/.venv/bin/python"
MON="$ROOT/.local_monitors"
PIPE_PY="$MON/batch3_stage_pipeline.py"
WD_SH="$MON/batch3_watchdog.sh"
PIPE_PIDF="$MON/batch3_stage_pipeline.pid"
WD_PIDF="$MON/batch3_watchdog.pid"
PIPE_LOG="$MON/batch3_stage_pipeline.log"

# Never kill a live pipeline (stale-log restarts caused thrash). Adopt or start only.
# Pipeline
alive=0
if [[ -f "$PIPE_PIDF" ]]; then
  pid=$(cat "$PIPE_PIDF" || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    alive=1
    echo "[ensure] pipeline ok pid=$pid"
  fi
fi
if [[ "$alive" -eq 0 ]]; then
  # prefer pgrep match on this checkout only
  existing=$(pgrep -f "$ROOT/.venv/bin/python -u $PIPE_PY" | head -1 || true)
  if [[ -n "${existing:-}" ]]; then
    echo "$existing" > "$PIPE_PIDF"
    echo "[ensure] adopted pipeline pid=$existing"
  else
    nohup "$PY" -u "$PIPE_PY" >> "$PIPE_LOG" 2>&1 &
    echo $! > "$PIPE_PIDF"
    echo "[ensure] started pipeline pid=$(cat "$PIPE_PIDF")"
  fi
fi

# Watchdog
wd_alive=0
if [[ -f "$WD_PIDF" ]]; then
  wpid=$(cat "$WD_PIDF" || true)
  if [[ -n "${wpid:-}" ]] && kill -0 "$wpid" 2>/dev/null; then
    wd_alive=1
    echo "[ensure] watchdog ok pid=$wpid"
  fi
fi
if [[ "$wd_alive" -eq 0 ]]; then
  nohup bash "$WD_SH" >> "$MON/batch3_watchdog.log" 2>&1 &
  echo $! > "$WD_PIDF"
  echo "[ensure] started watchdog pid=$(cat "$WD_PIDF")"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$MON/batch3_monitor.alive"
{
  echo "pid=$(cat "$PIPE_PIDF")"
  echo "watchdog=$(cat "$WD_PIDF")"
  echo "cwd=$PWD"
  echo "note=no stage-a relaunch; on 13/13 pipeline runs fetch→b-rescale→fetch→c→fetch"
} >> "$MON/batch3_monitor.alive"

echo "PIPELINE_PID=$(cat "$PIPE_PIDF")"
echo "WATCHDOG_PID=$(cat "$WD_PIDF")"
