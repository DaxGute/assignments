"""Unified A3 batch orchestrator (A2-style).

    uv run python -m experiments.a3_optimization.launch_batch --batch 1 --dry-run
    uv run python -m experiments.a3_optimization.launch_batch --batch 2 --execute
    uv run python -m experiments.a3_optimization.launch_batch --batch 3 --stage a
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if args.dry_run:
        rest = ["--dry-run", *rest]
    elif args.execute:
        rest = ["--execute", *rest]
    if args.batch == 1:
        from experiments.a3_optimization.batches.batch1 import main as batch_main
    elif args.batch == 2:
        from experiments.a3_optimization.batches.batch2 import main as batch_main
    else:
        from experiments.a3_optimization.batches.batch3 import main as batch_main
    return batch_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
