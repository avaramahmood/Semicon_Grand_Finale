#!/usr/bin/env python3
"""Drift-Sense Phase 3 — register a GDSII design against an SEM search image.

    python phase3.py --input pairs.csv --output predictions.csv

Same shape and arguments as Phase 2's `register.py`. Reads the Phase 3 `pairs.csv`
(`pair_id, search_path, reference_gds_path, search_gds_path, reference_sem_path,
params_json_path`; the last two are empty on the blind split and are never required)
and writes `pair_id, x, y, theta, scale, found, score`.

Registration is driven off EDGES, as the Phase 3 spec asks: the per-layer signed
dx/dy gradient fit. The per-layer brightness ("yield raster") is inferred separately
and its fit quality feeds the `score` column, which is what calibration is scored on —
it is never what drives the match.

Runs offline, CPU only, no network. The re-ranker's output head is zero-initialised and
its logit is `head(e) + alpha * classical_scores`, so with no checkpoint present the
ranking IS the classical similarity score -- deterministic, not seed-dependent. Dropping
a trained `weights/model_best.pt` in turns the head into a learned correction on top; the
bundle carries its own cfg, calibration and threshold.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))         # dsr_core, p3_data

import torch                                                          # noqa: E402
import dsr_core as core                                               # noqa: E402
import p3_data as p3                                                  # noqa: E402

WEIGHTS = os.path.join(HERE, 'weights', 'model_best.pt')
PER_PAIR_TIMEOUT_S = 18.0        # the rubric's hard timeout is 20 s; leave margin
DEFAULT_THREADS = 4

# Found cutoff on the presence score: the fraction of the reference's 1 nm design the search
# design reproduces at the best CAD peak. Chosen on DEVELOPMENT sets only -- 24 i4c pairs
# with rotation off, 24 with rotation on, and our 24-pair absent-heavy set -- never on the
# test sets it is later scored on. There the classes do not overlap: present pairs 0.9993 to
# 1.0000, absent pairs 0.7440 to 0.8981. 0.95 sits in the gap.
DEFAULT_GEOMETRY_THR = 0.95

# Method 2 of the four trained variants: global pose + per-layer dx/dy edge fit +
# tone map. Cold (untrained) on 50 held-out generated pairs: 84.19 of 85, with
# localization and pose both at 1.000. See docs/METHODS.md.
FALLBACK_CFG = dict(
    phase='p3', zoom_range=[8.0, 12.0], theta_range=[-5.0, 5.0],
    families=['edgefit2', 'mtm'], raw_family='mtm',
    patch_channels=['T_edge', 'S_edge', 'T_fit', 'S_int', 'R_fit'],
    refine_families=['edgefit2', 'mtm'],
    use_global_pose=True, use_cad_prior=False, per_pose=24,
    consensus=True, offset_head=True, ch=48, estimate_greys=True,
    sigma_band=1.0, sigma_edge=1.0, subpix='parab', mtm_edge_band=False,
    refine_levels=[[0.25, 0.5], [0.0625, 0.125], [0.02, 0.04]],
    gt_halfpix=0.45, gt_shear_px=1.5,
    out_dir=os.path.join(HERE, 'weights'),
)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):                                            # noqa: ARG001
    raise _Timeout()


def load_engine(weights=WEIGHTS, device='cpu'):
    """(model, stagea, platt, threshold, trained).

    With no checkpoint the zero-initialised head contributes an identical constant to
    every candidate, so the ranking reduces exactly to the classical score. Verified:
    six random seeds pick the same candidate and give the same output to the pixel."""
    if os.path.isfile(weights):
        model, stagea, platt, thr = core.load_bundle(weights, device)
        return model, stagea, platt, thr, True
    cfg = core.make_cfg(**FALLBACK_CFG)
    stagea = core.StageA(cfg, torch.device(device))
    model = core.CrossEncoder(cfg).to(device)
    model.eval()
    return model, stagea, (1.0, 0.0), 0.5, False


def _geometry_score(item):
    """Presence from the search-CAD geometry, used when no calibrated head ships.

    The two CADs share a frame, so locating the reference inside the search design is
    noise-free: a strong peak with no rival means present, many near-equal rivals or a
    weak peak means the site is not there. Measured untrained: F1 0.959-0.973."""
    g = item.get('gdsg')
    if g and g.get('fine_best') is not None:
        # How much of the reference's 1 nm design the search design reproduces at the best
        # CAD peak (p3_data.fine_match). Present: the true copy matches exactly. Absent: the
        # reference came from elsewhere and nothing reproduces it. This replaced
        # top_sigma * (1 - runner_up), which read similar-but-different layouts as "the design
        # repeats" and rejected 7 of 26 present pairs on the organisers' generator.
        return float(g['fine_best'])
    if g:
        return float(g['top_sigma']) * (1.0 - float(g['runner_up']))
    pr = item.get('prior')
    if pr:
        return -float(np.log1p(pr['rivals']))
    return 0.0


def predict_one(row, root, engine):
    model, stagea, platt, thr, trained = engine
    cfg = stagea.cfg
    item = p3.load_pair(
        os.path.join(root, row['reference_gds_path']),
        os.path.join(root, row['search_path']),
        os.path.join(root, row['search_gds_path']) if row.get('search_gds_path') else None,
        global_pose_on=bool(cfg['use_global_pose']),
        # present only on the training split; the spec warns code that depends on it
        # will fail on the scored run, so it is strictly optional here
        params_json_path=(os.path.join(root, row['params_json_path'])
                          if row.get('params_json_path') else None),
    )
    p = core.predict(model, stagea, item, platt=platt, threshold=thr)
    if not trained:
        # no calibrated presence head: use the geometry evidence, and let the caller's
        # threshold decide. The score column is still monotone in confidence, which is
        # what the AUC part of calibration measures (measured AUC 1.000 on 14 absent).
        s = _geometry_score(item)
        p['score'] = s
        p['found'] = int(s >= thr)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='pairs.csv')
    ap.add_argument('--output', required=True, help='predictions.csv')
    ap.add_argument('--weights', default=WEIGHTS)
    ap.add_argument('--threshold', type=float, default=None,
                    help='found cutoff. With weights: a calibrated probability in '
                         '[0,1] (default: the F1-optimal value saved in the bundle). '
                         f'Without weights: a geometry score, default '
                         f'{DEFAULT_GEOMETRY_THR}. The two are '
                         'on different scales -- do not pass one for the other.')
    ap.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    engine = load_engine(args.weights, args.device)
    trained = engine[4]
    thr = args.threshold if args.threshold is not None else (
        engine[3] if trained else DEFAULT_GEOMETRY_THR)
    engine = engine[:3] + (thr, trained)

    root = os.path.dirname(os.path.abspath(args.input))
    with open(args.input, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'[phase3] {len(rows)} pairs | weights: '
          f'{args.weights if trained else "none (classical scoring)"} | '
          f'threshold {thr:.4f} | {args.threads} threads', file=sys.stderr, flush=True)

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
            p = predict_one(r, root, engine)
        except Exception as e:                                        # noqa: BLE001
            # A missing row scores zero, so never propagate. Rejecting is the honest
            # fallback: we have no evidence this pair contains anything.
            print(f'[phase3] {pid}: {type(e).__name__}: {e} -> reject',
                  file=sys.stderr)
            p = dict(x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=-9.99)
        finally:
            if have_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
        times.append(time.perf_counter() - t0)
        p['pair_id'] = pid
        out.append(p)
        if (n + 1) % 10 == 0:
            print(f'[phase3] {n + 1}/{len(rows)}  median {np.median(times):.2f}s',
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
    print(f'[phase3] wrote {args.output}: {len(out)} rows, median '
          f'{np.median(times):.2f}s/pair (max {max(times):.2f}s)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
