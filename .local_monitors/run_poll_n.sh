#!/bin/bash
cd /Users/daxtongutekunst/Desktop/assignments || exit 1
exec uv run python -u .local_monitors/poll_n_loop.py
