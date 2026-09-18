#!/usr/bin/env python3
"""Score a Phase 3 predictions.csv against a ground_truth.csv, on the published rubric.

    python score.py --truth dataset/ground_truth.csv --pred predictions.csv

Reports the 85 points a local run can compute — localization 40, pose 20, rejection 15,
calibration 10 — and leaves the two the jury owns (generator analysis 10, efficiency 5,
which is quartile-ranked against other entrants). Pass `--times` to also print the
efficiency evidence.

`--sweep` prints the rejection F1 across candidate thresholds, which is how the found
cutoff is chosen when no calibrated head ships in the weights bundle.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import dsr_core as core                                               # noqa: E402


def read(path, want_float=('x', 'y', 'theta', 'scale', 'score')):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in want_float or k in ('present', 'found'):
                r[k] = float(v) if v not in ('', None) else 0.0
    return {r['pair_id']: r for r in rows}


def build_records(truth, pred):
    recs, missing = [], []
    for pid, t in truth.items():
        p = pred.get(pid)
        if p is None:
            missing.append(pid)
            # a missing row scores zero: reject at the worst possible score
            p = dict(x=0.0, y=0.0, theta=0.0, scale=0.0, found=0.0, score=-9.99)
        recs.append(dict(present=t['present'], x=t['x'], y=t['y'],
                         theta=t['theta'], scale=t['scale'],
                         px=p['x'], py=p['y'], pth=p['theta'], pz=p['scale'],
                         score=p['score'], found=int(p['found']), gray=True,
                         severity=0, kind='present' if t['present'] > 0.5 else 'absent',
                         pair_id=pid))
    return recs, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--truth', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--worst', type=int, default=5, help='list the N worst present pairs')
    args = ap.parse_args(argv)

    truth, pred = read(args.truth), read(args.pred)
    recs, missing = build_records(truth, pred)
    extra = sorted(set(pred) - set(truth))
    r = core.rubric(recs)

    pres = [c for c in recs if c['present'] > 0.5]
    errs = np.array([np.hypot(c['px'] - c['x'], c['py'] - c['y']) for c in pres])
    tiers = [(1, 1.00), (2, 0.80), (3, 0.60), (5, 0.40)]

    print(f'pairs {len(recs)}  present {len(pres)}  absent {len(recs) - len(pres)}')
    if missing:
        print(f'  !! {len(missing)} MISSING prediction rows (each scores zero): '
              f'{", ".join(missing[:8])}{" ..." if len(missing) > 8 else ""}')
    if extra:
        print(f'  !! {len(extra)} prediction rows with no ground truth: {", ".join(extra[:8])}')
    print()
    print(f'  LOCALIZATION   {r["loc_credit"] * 40:6.2f} / 40   credit {r["loc_credit"]:.3f}   '
          f'median {np.median(errs):.2f} px')
    for t, w in tiers:
        print(f'      <= {t} px ({w:.2f} credit)   {np.mean(errs <= t):.3f}   '
              f'{int((errs <= t).sum())}/{len(errs)}')
    print(f'  POSE           {r["pose_credit"] * 20:6.2f} / 20   scale {r["scale_credit"]:.3f}  '
          f'theta {r["theta_credit"]:.3f}')
    print(f'  REJECTION      {r["rejection_f1"] * 15:6.2f} / 15   F1 {r["rejection_f1"]:.3f}')
    print(f'  CALIBRATION    {r["roc_auc"] * 10:6.2f} / 10   AUC {r["roc_auc"]:.3f}')
    print(f'  ---------------------------------')
    print(f'  TOTAL (of 85)  {r["points"]:6.2f}      + generator analysis 10 + efficiency 5, '
          f'both judged')

    if args.worst and len(pres):
        order = np.argsort(-errs)[:args.worst]
        print(f'\n  worst {len(order)} present pairs:')
        for i in order:
            c = pres[int(i)]
            print(f'    {c["pair_id"]}  err {errs[int(i)]:7.2f} px   '
                  f'truth ({c["x"]:.1f}, {c["y"]:.1f}) scale {c["scale"]:.3f} theta {c["theta"]:+.2f}'
                  f'  ->  pred ({c["px"]:.1f}, {c["py"]:.1f}) scale {c["pz"]:.3f} theta {c["pth"]:+.2f}'
                  f'  found={c["found"]} score {c["score"]:.3f}')

    if args.sweep:
        s = np.array([c['score'] for c in recs])
        y = np.array([c['present'] > 0.5 for c in recs])
        print('\n  found-threshold sweep (F1 on the found flag):')
        best = (0.0, None)
        for t in np.unique(np.round(np.quantile(s, np.linspace(0, 1, 25)), 4)):
            f = (lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1e-9))(
                int(((s >= t) & y).sum()), int(((s >= t) & ~y).sum()), int(((s < t) & y).sum()))
            print(f'    >= {t:9.4f}   F1 {f:.3f}')
            best = max(best, (f, t))
        print(f'  best: threshold {best[1]:.4f} -> F1 {best[0]:.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
