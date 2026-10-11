#!/bin/bash
set -euo pipefail
ROOT=/Users/daxtongutekunst/Desktop/assignments
cd "$ROOT"
case "$PWD" in *assignments-a3-batch1*) exit 4;; esac
test "$PWD" = "$ROOT" || exit 1
PIDFILE=.local_monitors/batch3_stage_pipeline.pid
LOG=.local_monitors/batch3_stage_pipeline.log
PY=$ROOT/.venv/bin/python
# Never relaunch Modal stage a — only keep the local wait→fetch→b→c pipeline alive.
while true; do
  alive=0
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE" || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  fi
  if [[ "$alive" -eq 0 ]]; then
    existing=$(pgrep -f "$ROOT/.venv/bin/python -u $ROOT/.local_monitors/batch3_stage_pipeline.py" | head -1 || true)
    if [[ -n "${existing:-}" ]]; then
      echo "$existing" > "$PIDFILE"
      echo "[$(date '+%F %T')] watchdog adopted pid=$existing" >> "$LOG"
    else
      echo "[$(date '+%F %T')] watchdog restart" >> "$LOG"
      nohup "$PY" -u .local_monitors/batch3_stage_pipeline.py >> "$LOG" 2>&1 &
      echo $! > "$PIDFILE"
    fi
    date -u +%Y-%m-%dT%H:%M:%SZ > .local_monitors/batch3_monitor.alive
    echo "pid=$(cat $PIDFILE)" >> .local_monitors/batch3_monitor.alive
  fi
  sleep 60
done
