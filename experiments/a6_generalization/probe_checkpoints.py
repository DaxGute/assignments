"""Directional sharpness on released checkpoints, same probe as the batch sweep.

Uses the first four rows of the cached target split and the starter probe at
radius 0.05. Deletes nothing. The batch-16 reproduction of this probe reads
0.1208 against the released 0.1203.
"""

from pathlib import Path
import json
import sys

import numpy as np
import torch

from experiments.a6_generalization.common import Config
from experiments.a6_generalization.models import build
from experiments.a6_generalization.p3_optimization import directional_sharpness

CACHE = Path('results/a6_generalization/reference_cache')


def probe(path, depth, width):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
    model = build(Config(name='probe', depth=depth, width=width, probe_sharpness=False))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if list(missing) or list(unexpected):
        raise RuntimeError(f'{path}: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}')
    ids = np.array(np.load(CACHE / 'target_30k.npy', mmap_mode='r')[:4])
    result = directional_sharpness(model, ids, radius=0.05)
    result['parameters'] = sum(p.numel() for p in model.parameters())
    result['depth'] = depth
    result['width'] = width
    result['checkpoint'] = str(path)
    return result


def main():
    rows = []
    for depth, width in ((4, 256), (5, 320), (6, 384), (7, 448), (8, 512), (9, 576), (10, 640)):
        path = CACHE / f'd{depth}.pt'
        if not path.is_file():
            print('missing', path, file=sys.stderr)
            continue
        row = probe(path, depth, width)
        rows.append(row)
        print(json.dumps(row), flush=True)
    destination = Path('results/a6_generalization/parameter_sharpness.json')
    destination.write_text(json.dumps(rows, indent=2) + '\n')
    print('wrote', destination)


if __name__ == '__main__':
    main()
