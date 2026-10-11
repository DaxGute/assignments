#!/bin/bash
cd /Users/daxtongutekunst/Desktop/assignments || exit 1
LOG=/Users/daxtongutekunst/Desktop/assignments/.local_monitors/batch3_quiet_watch.log
while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') SUPERVISOR start quiet_watch" >> "$LOG"
  /Users/daxtongutekunst/Desktop/assignments/.venv/bin/python -u /Users/daxtongutekunst/Desktop/assignments/.local_monitors/quiet_watch.py >> "$LOG" 2>&1
  rc=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') SUPERVISOR quiet_watch exited rc=$rc; restart in 15s" >> "$LOG"
  sleep 15
done
