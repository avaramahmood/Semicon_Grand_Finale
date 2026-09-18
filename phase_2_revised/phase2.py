#!/usr/bin/env python3
"""Drift-Sense Phase 2 — register an SEM reference against an SEM search image.

    python phase2.py --input pairs.csv --output predictions.csv

Reads `pair_id, search_path, reference_path` and writes
`pair_id, x, y, theta, scale, found, score`. Paths resolve relative to the directory
holding `pairs.csv`.

Both sides are SEM images, so there is no design file and nothing to infer about
per-layer brightness: the reference is correlated against the search over a grid of
zoom and rotation, the best candidates are re-ranked by a trained cross-encoder, and
the pose is refined to sub-pixel.

Runs offline, CPU only. `weights/model_best.pt` carries its own cfg, Platt calibration
and F1-optimal found threshold, so the checkpoint decides the settings, not this script.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import signal
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))

import torch                                                          # noqa: E402
import dsr_core as core                                               # noqa: E402

WEIGHTS = os.path.join(HERE, 'weights', 'model_best.pt')
PER_PAIR_TIMEOUT_S = 18.0        # the rubric's hard timeout is 20 s; leave margin
DEFAULT_THREADS = 4
KIND = 'sem'


class _Timeout(Exception):
    pass


def _alarm(signum, frame):                                            # noqa: ARG001
    raise _Timeout()


def load_pair(reference_path, search_path):
    """One row -> the item format StageA expects. A 3-channel search is an optical
    capture; it is matched on luminance, and the flag rides along for the re-ranker."""
    ref = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)
    srch = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
    if ref is None or srch is None:
        raise FileNotFoundError(f'{reference_path} / {search_path}')
    gray = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    return dict(search=np.ascontiguousarray(gray(srch)),
                small=core.ref_to_small(gray(ref)), kind=KIND,
                meta=dict(optical=srch.ndim == 3))


def load_engine(weights=WEIGHTS, device='cpu'):
    """(model, stagea, platt, threshold, trained)."""
    if os.path.isfile(weights):
        model, stagea, platt, thr = core.load_bundle(weights, device)
        return model, stagea, platt, thr, True
    raise SystemExit(
        f'no checkpoint at {weights}.\n'
        'Phase 2 is trained: the re-ranker head is zero-initialised, so without weights\n'
        'the ranking falls back to the raw classical score and the presence head is\n'
        'uncalibrated. Put model_best.pt in weights/ (see weights/README.md).')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='pairs.csv')
    ap.add_argument('--output', required=True, help='predictions.csv')
    ap.add_argument('--weights', default=WEIGHTS)
    ap.add_argument('--threshold', type=float, default=None,
                    help='found cutoff, a calibrated probability in [0,1]. Default: the '
                         'F1-optimal value saved in the checkpoint.')
    ap.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    model, stagea, platt, bundle_thr, _ = load_engine(args.weights, args.device)
    thr = args.threshold if args.threshold is not None else bundle_thr

    root = os.path.dirname(os.path.abspath(args.input))
    with open(args.input, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'[phase2] {len(rows)} pairs | {os.path.basename(args.weights)} '
          f'({stagea.cfg.get("variant")}) | threshold {thr:.4f} | {args.threads} threads',
          file=sys.stderr, flush=True)

    have_alarm = hasattr(signal, 'SIGALRM')
    if have_alarm:
        signal.signal(signal.SIGALRM, _alarm)

    out, times = [], []
    for n, r in enumerate(rows):
        pid = r['pair_id']
        t0 = time.perf_counter()
        try:
            if have_alarm:
                signal.setitimer(signal.ITIMER_REAL, PER_PAIR_TIMEOUT_S)
            item = load_pair(os.path.join(root, r['reference_path']),
                             os.path.join(root, r['search_path']))
            p = core.predict(model, stagea, item, platt=platt, threshold=thr)
        except Exception as e:                                        # noqa: BLE001
            # a missing row scores zero, so never propagate; rejecting is the honest
            # fallback when we have no evidence the reference is there at all
            print(f'[phase2] {pid}: {type(e).__name__}: {e} -> reject', file=sys.stderr)
            p = dict(x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=0.0)
        finally:
            if have_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
        times.append(time.perf_counter() - t0)
        p['pair_id'] = pid
        out.append(p)
        if (n + 1) % 10 == 0:
            print(f'[phase2] {n + 1}/{len(rows)}  median {np.median(times):.2f}s',
                  file=sys.stderr, flush=True)
        gc.collect()

    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_id', 'x', 'y', 'theta', 'scale',
                                          'found', 'score'])
        w.writeheader()
        for p in out:
            w.writerow({'pair_id': p['pair_id'],
                        'x': round(float(p['x']), 3), 'y': round(float(p['y']), 3),
                        'theta': round(float(p['theta']), 4),
                        'scale': round(float(p['scale']), 4),
                        'found': int(p['found']),
                        'score': round(float(p['score']), 6)})
    print(f'[phase2] wrote {args.output}: {len(out)} rows, median '
          f'{np.median(times):.2f}s/pair (max {max(times):.2f}s)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
