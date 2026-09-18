"""Drift-Sense Phase 2-revised / Phase 3 -- shared training core.

One module for all six methods (3 x Phase 2 revised, 3 x Phase 3). The method
decides WHICH similarity families, patch channels and heads are on; the
candidate search, cross-encoder, losses, streaming, checkpoint/resume and
rubric validation live here, so methods differ only in what is compared.

Spine (the declared Phase 1 approach, unchanged in kind):
  similarity maps over a zoom x theta grid -> K candidates -> early-fusion
  cross-encoder with a classical-score warm start -> residual pose head ->
  coarse-to-fine pose refinement -> calibrated found flag.

Carried forward because it was MEASURED to matter:
  * Stage A runs in torch (FFT). 25 poses of 1000x1000 ZNCC: 0.19 s torch vs
    1.19 s cv2 on 4 CPU cores; torch ZNCC == cv2.TM_CCOEFF_NORMED to 7e-5.
    The same code runs on the GPU in training and on CPU at inference.
  * Presence reads a RAW (not band-matched) low-res pose surface; band matching
    flipped the present/absent gap from +0.143 to -1.013 in Phase 2.
  * Warm start: at init the model's argmax IS the classical argmax.
  * Pose refinement: on the organisers' 25 pairs the untrained pipeline scores
    pose credit 0.85 (A2 heldout had theta<=0.25 deg on only 36.6%).

Added from the training-methods survey (.ml-intern/ledger-training-methods.md):
  * rank-and-sort term on the candidate list (arXiv:2107.11669, 1705.10872)
  * residual pose head reading only the patch pair (2407.11668, 2601.12530)
  * learned boundary net for edge methods, trained on exact synthetic
    boundaries, auto-fallback to classical gradients (2308.06468, 2408.04258,
    2409.02348, 1812.07032)
  * modality dropout for hybrids (2203.15332, 2507.06566, 2310.15261)
  * post-hoc Platt calibration + F1 threshold for the found flag, fitted on
    held-out val (2405.20459, 2301.09044); EMA weights shipped (2411.18704)

Families (similarity maps, per pose):
  int   band-matched ZNCC on intensity
  edge  ZNCC on boundary maps: smoothed gradient magnitude (classical) or the
        learned boundary net's output
  mtm   Phase 3: per-layer tone-mapped match. One grey per CAD layer fitted by
        least squares at every position, score sqrt(R^2) (arXiv:2007.12463).
"""

import glob
import hashlib
import json
import math
import os
import queue
import random
import signal
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEARCH_PX = 1000
REF_PX = 1000
SMALL_PX = 250            # reference is area-downsampled 4x once; templates come from this
SMALL_FACTOR = REF_PX / SMALL_PX
TOL_PX = 5.0

# ==========================================================================
# Rubric -- identical to phase2submission/eval (the scored contract)
# ==========================================================================


def loc_credit(err_px):
    if err_px <= 1.0:
        return 1.00
    if err_px <= 2.0:
        return 0.80
    if err_px <= 3.0:
        return 0.60
    if err_px <= 5.0:
        return 0.40
    return 0.00


def scale_credit(pred, true):
    if true <= 0:
        return 0.0
    r = abs(pred - true) / true
    return 1.00 if r <= 0.01 else 0.60 if r <= 0.02 else 0.30 if r <= 0.05 else 0.0


def theta_credit(pred, true):
    d = abs(pred - true)
    return 1.00 if d <= 0.25 else 0.60 if d <= 0.5 else 0.30 if d <= 1.0 else 0.0


def roc_auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return float('nan')
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum())
                 / (len(p) * len(n)))


def best_f1(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    best = (0.0, float(s.min()) - 1 if len(s) else 0.0)
    for t in np.unique(s):
        pr = s >= t
        tp, fp, fn = int((pr & y).sum()), int((pr & ~y).sum()), int((~pr & y).sum())
        d = 2 * tp + fp + fn
        v = 2 * tp / d if d else 0.0
        if v > best[0]:
            best = (v, float(t))
    return best


def fit_platt(logits, labels, iters=60, l2=1e-3):
    """Post-hoc Platt scaling p = sigmoid(a*s + b), Newton's method. Monotone, so
    it cannot change AUC; it makes the score a probability and the threshold
    portable (arXiv:2405.20459: post-hoc beats train-time calibration)."""
    s, y = np.asarray(logits, float), np.asarray(labels, float)
    if len(s) < 4 or y.min() == y.max():
        return 1.0, 0.0
    mu, sd = s.mean(), s.std() + 1e-9
    z = (s - mu) / sd
    w = np.zeros(2)
    X = np.stack([z, np.ones_like(z)], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0] / sd), float(w[1] - w[0] * mu / sd)


def calibrate(logit, platt):
    a, b = platt
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, a * logit + b))))


# ==========================================================================
# Config
# ==========================================================================

DEFAULTS = dict(
    variant='1_intensity', phase='p2r',
    # --- what the method compares
    families=['int'],            # similarity maps used for candidates
    raw_family='int',            # low-res RAW surface for presence ('int' | 'edge' | 'mtm')
    patch_channels=['T_int', 'S_int'],
    refine_families=['int'],
    consensus=False,             # set-attention over the K candidates (NCNet-style)
    offset_head=True,            # residual (dx, dy, dzoom, dtheta) from the patch pair
    edge_source='classical',     # 'classical' | 'learned' (boundary net, auto-fallback)
    modality_dropout=0.0,        # hybrids: P(blank one family) per training sample
    # --- pose search
    zoom_range=(8.0, 12.0), theta_range=(-5.0, 5.0),
    zooms=(8.0, 9.0, 10.0, 11.0, 12.0), thetas=(-5.0, -2.5, 0.0, 2.5, 5.0),
    k=64, per_pose=6, nms_r=6, patch=80,
    sigma_band=0.5, sigma_edge=1.0,   # measured: see phase-2-3-training-plan.md
    mtm_edge_band=False,         # measured: the free per-window fit already absorbs edge brightening
    estimate_greys=True,         # run the coarse pass to infer ONE yield raster (features + report)
    use_cad_prior=False,         # Phase 3: candidates from search.gds geometry (see p3_data.cad_prior)
    use_global_pose=False,       # Phase 3: pose from the whole search CAD vs SEM (p3_data.gds_geometry)
    # --- ground-truth CONVENTION of the i4c CAD generator (phase3/GROUND-TRUTH-CONVENTION.md).
    # Its label is NOT where the feature is visible, by two measured amounts. Both are
    # applied to the reported x/y, and p3_data emits training labels the same way, so the
    # whole pipeline targets one convention. Set both to 0.0 to report the visible position.
    gt_halfpix=0.0,              # cv2.resize(INTER_AREA) by 10 inverts as (X-4.5)/10, but
                                 # cad_pipeline writes rx/SCALE_FACTOR: the label is +0.45 px
                                 # high in BOTH axes.
    gt_shear_px=0.0,             # gt is fixed BEFORE apply_raster_drift, which shifts row y
                                 # by shear*y/(h-1). Nominal 1.5; load_pair overrides it per
                                 # pair from params.json when the dataset ships one.
    pose_prune_keep=0,           # >0: score all poses at 1/pose_prune_ds first, keep the best N at full res
    pose_prune_ds=4,
    pose_prune_r=0,              # >0: with a CAD prior, read the coarse map only within
                                 # this radius (full-res px) of the prior's mapped
                                 # position. MEASURED on the 50-pair CAD set with the
                                 # prior + edgefit2 + mtm: the plain global max is the
                                 # better of the two (81.40 vs 81.12 at keep=6), so 0.
    mtm_coarse_ds=2,             # 'mtm2': downsample of the coarse locating pass
    prefilter='gauss',           # 'gauss' | 'dog' | 'anscombe' | 'median' (all end in the sigma_band low-pass)
    dog_sigma=12.0,              # 'dog': subtract a wide Gaussian -> band-pass (removes vignette / charging drift)
    subpix='quad2d',             # 'parab' | 'gauss' | 'centroid' | 'quad2d' | 'none'
    refine_levels=((0.25, 0.5), (0.0625, 0.125), (0.02, 0.04)),
    # --- model
    ch=40, d=128, scalar_dropout=0.3, alpha_init=8.0,
    # --- learned boundary net (edge_source='learned')
    edge_ch=16, edge_pretrain_min=45.0, edge_lr=2e-3, edge_batch=8, edge_crop=256, edge_buf=400,
    # --- optimisation
    lr=3e-4, weight_decay=0.01, batch=16, ema_decay=0.999, tau=1.0,
    simans_sigma=0.05, dual_focal=0.3, rank_weight=0.5, warmup_min=5.0, cooldown_frac=0.15,
    replay_ratio=8.0,            # gradient samples per unique pair
    buf_pairs=1200, min_fill=48,
    # --- streaming
    n_workers=3, pairs_per_canvas=8, queue_max=64, seed=0,
    # --- budget / resume
    total_train_hours=10.0,      # LR schedule spans this across ALL sessions
    session_hours=11.5,          # stop + checkpoint before Kaggle's 12 h wall
    ckpt_every_min=10.0, val_every_min=30.0,
    val_pairs=160, val_seed=990_000,
    out_dir=None, resume='auto',
    device=None, log_every_sec=120.0,
)

# keys that change tensor shapes or features -- a checkpoint only resumes into the same set
ARCH_KEYS = ('variant', 'families', 'raw_family', 'patch_channels', 'consensus', 'offset_head',
             'edge_source', 'edge_ch', 'zooms', 'thetas', 'k', 'patch', 'ch', 'd', 'prefilter', 'sigma_band',
             'estimate_greys', 'mtm_edge_band', 'use_cad_prior', 'use_global_pose')


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    for key in ('families', 'patch_channels', 'refine_families'):
        cfg[key] = list(cfg[key])
    for key in ('zooms', 'thetas', 'zoom_range', 'theta_range'):
        cfg[key] = [float(v) for v in cfg[key]]
    cfg['refine_levels'] = [tuple(float(x) for x in lv) for lv in cfg['refine_levels']]
    if cfg['out_dir'] is None:
        base = '/kaggle/working' if os.path.isdir('/kaggle/working') else './runs'
        cfg['out_dir'] = os.path.join(base, cfg['variant'])
    return cfg


def arch_signature(cfg):
    blob = json.dumps({k: cfg[k] for k in ARCH_KEYS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def poses_of(cfg):
    return [(z, t) for z in cfg['zooms'] for t in cfg['thetas']]


# ==========================================================================
# Reference -> pose templates (cv2, CPU; identical at train and inference)
# ==========================================================================

def ref_to_small(ref):
    """(C, 1000, 1000) float -> (C, 250, 250) float32, area average."""
    ref = np.asarray(ref, np.float32)
    if ref.ndim == 2:
        ref = ref[None]
    return np.stack([cv2.resize(c, (SMALL_PX, SMALL_PX), interpolation=cv2.INTER_AREA)
                     for c in ref]).astype(np.float32)


def pose_warp(small, zoom, theta):
    """What the reference looks like inside the search raster at (zoom, theta).

    Mirrors the generators' template convention (rotate +theta about the
    reference centre, scale 1/zoom, box prefilter of width zoom) from the 4x
    area-downsampled reference: area 4 already contributes 16/12 px^2 of box
    variance, the rest comes from a Gaussian so the prefilter is smooth in z.
    """
    C = small.shape[0]
    n = int(round(REF_PX / zoom))
    var = max(zoom * zoom - SMALL_FACTOR ** 2, 0.0) / 12.0
    sig = math.sqrt(var) / SMALL_FACTOR
    c = (SMALL_PX - 1) / 2.0
    M = cv2.getRotationMatrix2D((c, c), theta, SMALL_FACTOR / zoom)
    M[0, 2] += (n - 1) / 2.0 - c
    M[1, 2] += (n - 1) / 2.0 - c
    out = np.empty((C, n, n), np.float32)
    for i in range(C):
        ch = small[i]
        if sig > 0.05:
            ch = cv2.GaussianBlur(ch, (0, 0), sig)
        out[i] = cv2.warpAffine(ch, M, (n, n), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    return out


# ==========================================================================
# Torch image ops
# ==========================================================================

def _gk(sigma, device):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum(), r


def gblur(x, sigma):
    """Separable Gaussian, reflect padding. x: (N, C, H, W)."""
    if sigma <= 0:
        return x
    k, r = _gk(sigma, x.device)
    C = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def grad_mag(x, sigma):
    """|grad(G_sigma * x)|, per channel. x: (N, C, H, W)."""
    b = gblur(x, sigma)
    b = F.pad(b, (1, 1, 1, 1), mode='replicate')
    gx = (b[..., 1:-1, 2:] - b[..., 1:-1, :-2]) * 0.5
    gy = (b[..., 2:, 1:-1] - b[..., :-2, 1:-1]) * 0.5
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def prefilter(x, cfg, values=True):
    """Stage 0 on (N, C, H, W) intensities in [0, 1], applied identically to search
    and templates. Every option ends in the sigma_band Gaussian, the measured
    Phase 1 win (top-1 +0.15 at low noise); the options differ in what comes first:
      gauss     low-pass only
      dog       low-pass minus a wide low-pass: band-pass, removes vignette and
                slow charging drift that ZNCC's zero-mean does not
      anscombe  2*sqrt(x*255 + 3/8): variance-stabilises Poisson shot noise first
      median    3x3 median first (impulse / salt-and-pepper noise)
    `values=False` (CAD masks) skips the value transforms (anscombe) that only
    make sense for detector counts."""
    kind = cfg['prefilter']
    if kind == 'anscombe' and values:
        x = 2.0 * torch.sqrt(x * 255.0 + 0.375) / 32.0
    elif kind == 'median' and values and min(x.shape[-2:]) >= 3:
        u = F.pad(x, (1, 1, 1, 1), mode='replicate').unfold(2, 3, 1).unfold(3, 3, 1)
        x = u.reshape(*u.shape[:4], 9).median(-1).values
    y = gblur(x, cfg['sigma_band'])
    if kind == 'dog':
        y = y - gblur(x, cfg['dog_sigma']) if min(x.shape[-2:]) > 2 * math.ceil(3 * cfg['dog_sigma']) \
            else y - y.mean((2, 3), keepdim=True)
    return y


def _integral(x):
    return F.pad(x.cumsum(0).cumsum(1), (1, 0, 1, 0))


def _box(I, h, w):
    return I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]


def _norm01(x):
    return (x - x.mean()) / (x.std() + 1e-6)


class SearchCtx:
    """FFT + integral images of one search channel, cached across poses."""

    def __init__(self, s):
        s = _norm01(s.float())
        self.s = s
        self.H, self.W = s.shape
        self.Fs = torch.fft.rfft2(s)
        sd = s.double()
        self.I1, self.I2 = _integral(sd), _integral(sd * sd)

    def xcorr(self, T):
        """T: (B, h, w) -> (B, H-h+1, W-w+1), sum_x S[u+x] T[x]."""
        B, h, w = T.shape
        pad = T.new_zeros(B, self.H, self.W)
        pad[:, :h, :w] = T
        c = torch.fft.irfft2(self.Fs[None] * torch.conj(torch.fft.rfft2(pad)), s=(self.H, self.W))
        return c[:, :self.H - h + 1, :self.W - w + 1]

    def window_var(self, h, w):
        n = h * w
        s1, s2 = _box(self.I1, h, w), _box(self.I2, h, w)
        return (s2 - s1 * s1 / n).clamp_min(0.0)


def zncc_maps(ctx, T):
    """ZNCC of templates T (B, h, w) against the search (== cv2.TM_CCOEFF_NORMED)."""
    B, h, w = T.shape
    t0 = T - T.mean((1, 2), keepdim=True)
    tn = t0.flatten(1).norm(dim=1)
    num = ctx.xcorr(t0)
    var = ctx.window_var(h, w)
    den = tn[:, None, None] * var.sqrt().float()[None]
    ok = (var > 1e-6 * h * w)[None] & (tn > 1e-6)[:, None, None]
    return torch.where(ok, num / den.clamp_min(1e-12), torch.zeros_like(num)).clamp(-1, 1)


def mtm_maps(ctx, M):
    """Tone-mapped match. M: (B, L, h, w) soft layer masks.

    At every position fit S ~ c + sum_l g_l M_l by least squares and return
    sqrt(R^2). The centered mask Gram matrix does not depend on position, so the
    masks are whitened once per pose and explained variance = sum_j (w_j * S)^2
    needs only L FFT correlations. With L=1 this is exactly |ZNCC| (9e-8).
    """
    B, L, h, w = M.shape
    m = M - M.mean((2, 3), keepdim=True)
    flat = m.flatten(2).double()
    G = flat @ flat.transpose(1, 2)
    ev, evec = torch.linalg.eigh(G)
    thr = ev.amax(-1, keepdim=True) * 1e-4 + 1e-9 * h * w
    scale = torch.where(ev > thr, ev.clamp_min(1e-30).rsqrt(), torch.zeros_like(ev))
    Wm = evec * scale[:, None, :]
    live = (scale > 0).any(0)
    Wm = Wm[:, :, live]
    r = int(live.sum())
    if r == 0:
        return torch.zeros(B, ctx.H - h + 1, ctx.W - w + 1, device=M.device)
    wm = torch.einsum('blhw,blr->brhw', m.double(), Wm).float()
    num = ctx.xcorr(wm.reshape(B * r, h, w)).reshape(B, r, ctx.H - h + 1, ctx.W - w + 1)
    expl = (num * num).sum(1).double()
    var = ctx.window_var(h, w)
    ok = var > 1e-6 * h * w
    r2 = torch.where(ok[None], expl / var.clamp_min(1e-12)[None], torch.zeros_like(expl))
    return r2.clamp(0, 1).sqrt().float()


def mtm_maps_multi(ctxs, Ts):
    """Tone-map fit sharing ONE weight vector across several observation channels.

    Used by 'edgefit2': fit per-layer weights so that the SEM's dx AND dy derivative
    images are explained together, `[Sx; Sy] ~ sum_l w_l [Mx_l; My_l]`. Gradient
    magnitude discards edge direction; signed derivatives keep it, and direction is
    what separates a boundary from a parallel neighbour one period away.
    """
    B, L, h, w = Ts[0].shape
    n = h * w
    G = None
    ms = []
    for T in Ts:
        m = T - T.mean((2, 3), keepdim=True)
        ms.append(m)
        flat = m.flatten(2).double()
        G = flat @ flat.transpose(1, 2) if G is None else G + flat @ flat.transpose(1, 2)
    ev, evec = torch.linalg.eigh(G)
    thr = ev.amax(-1, keepdim=True) * 1e-4 + 1e-9 * n
    scale = torch.where(ev > thr, ev.clamp_min(1e-30).rsqrt(), torch.zeros_like(ev))
    Wm = evec * scale[:, None, :]
    live = (scale > 0).any(0)
    Wm = Wm[:, :, live]
    r = int(live.sum())
    H, W = ctxs[0].H, ctxs[0].W
    if r == 0:
        return torch.zeros(B, H - h + 1, W - w + 1, device=Ts[0].device)
    expl = torch.zeros(B, H - h + 1, W - w + 1, device=Ts[0].device, dtype=torch.float64)
    var = torch.zeros_like(expl)
    for ctx, m in zip(ctxs, ms):
        wm = torch.einsum('blhw,blr->brhw', m.double(), Wm).float()
        num = ctx.xcorr(wm.reshape(B * r, h, w)).reshape(B, r, H - h + 1, W - w + 1)
        expl = expl + (num * num).sum(1).double()
        var = var + ctx.window_var(h, w)[None]
    ok = var > 1e-6 * n
    r2 = torch.where(ok, expl / var.clamp_min(1e-12), torch.zeros_like(expl))
    return r2.clamp(0, 1).sqrt().float()


def deriv_xy(x, sigma):
    """Signed first derivatives of a smoothed image: (dx, dy), each (N, C, H, W)."""
    b = gblur(x, sigma)
    b = F.pad(b, (1, 1, 1, 1), mode='replicate')
    return ((b[..., 1:-1, 2:] - b[..., 1:-1, :-2]) * 0.5,
            (b[..., 2:, 1:-1] - b[..., :-2, 1:-1]) * 0.5)


def peaks(m, per_pose, r):
    """Top `per_pose` local maxima at least r px apart (Chebyshev).

    A (2r+1)^2 max-pool is exact but costs 0.26 s per 1000^2 map on CPU. Greedy
    suppression over the top 250*per_pose values gives the same peaks whenever a
    peak's neighbourhood holds fewer than 250 of them, for ~5 ms."""
    v = m.flatten()
    vals, idx = v.topk(min(250 * per_pose, v.numel()))
    vals, idx = vals.tolist(), idx.tolist()
    W = m.shape[1]
    keep_v, keep_y, keep_x = [], [], []
    for val, ii in zip(vals, idx):
        yy, xx = divmod(ii, W)
        if any(abs(yy - a) <= r and abs(xx - b) <= r for a, b in zip(keep_y, keep_x)):
            continue
        keep_v.append(val)
        keep_y.append(yy)
        keep_x.append(xx)
        if len(keep_v) >= per_pose:
            break
    return keep_v, keep_y, keep_x


def psr(m, r=6):
    """Peak-to-sidelobe ratio of one correlation map: (peak - mean) / std outside
    a (2r+1)^2 window around the peak (MOSSE-style confidence)."""
    v = m.flatten()
    i = int(v.argmax())
    yy, xx = divmod(i, m.shape[1])
    mask = torch.ones_like(m, dtype=torch.bool)
    mask[max(0, yy - r):yy + r + 1, max(0, xx - r):xx + r + 1] = False
    side = m[mask]
    return float((m[yy, xx] - side.mean()) / (side.std() + 1e-6))


def surface_stats(per_pose_max, nz, nt, best_map=None):
    """[peak, peak - best non-adjacent pose, normalised entropy, std, PSR/100]."""
    S = per_pose_max.reshape(nz, nt)
    pk = float(S.max())
    if S.size == 1:
        return [pk, 0.0, 0.0, 0.0, psr(best_map) / 100.0 if best_map is not None else 0.0]
    i, j = np.unravel_index(int(np.argmax(S)), S.shape)
    m = np.ones_like(S, bool)
    m[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = False
    second = float(S[m].max()) if m.any() else pk
    q = np.exp((S - pk) / 0.05).ravel()
    q /= q.sum()
    ent = float(-(q * np.log(q + 1e-12)).sum() / math.log(max(S.size, 2)))
    return [pk, pk - second, ent, float(S.std()), psr(best_map) / 100.0 if best_map is not None else 0.0]


def raw_pool(family):
    """Downsample factor of the RAW presence surface: 2x for intensity/edge (the
    measured Phase 2 setting), 4x for the ~L-times costlier tone-mapped family."""
    return 4 if family == 'mtm' else 2


# ==========================================================================
# Learned boundary net (edge methods)
# ==========================================================================

class EdgeNet(nn.Module):
    """Tiny boundary detector (~6K params with ch=16), run at half resolution.

    Size class of TEED (arXiv:2308.06468, 58K) and UHNet (2408.04258, 42K).
    Trained from scratch on EXACT boundaries: the data workers push the clean
    scene's layer-boundary map through the same PSF, pose warp, drift and barrel
    as the noisy search, so the target is pixel-aligned and grey-free. Classical
    gradients degrade fastest at low SNR while a light learned detector held up
    (AiM-ED 2409.02348); boundaries, not texture edges, are the right label
    (DexiNed 2112.02250).

    Measured on 4 CPU threads at 1000x1000: a full-resolution dilated version
    took 1.87 s, so the stem strides 2 and the output is upsampled back.
    """

    def __init__(self, ch=16):
        super().__init__()

        def block(dil):
            return nn.Sequential(nn.Conv2d(ch, ch, 3, padding=dil, dilation=dil, groups=ch),
                                 nn.Conv2d(ch, ch, 1), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(1, ch, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
        self.blocks = nn.ModuleList([block(d) for d in (1, 2, 4, 1)])
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-6)
        h = self.stem(x)
        for b in self.blocks:
            h = h + b(h)
        return F.interpolate(self.out(h), size=(H, W), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def boundary(self, x):
        return torch.sigmoid(self.forward(x))


# ==========================================================================
# Stage A: similarity families -> K candidates -> patch tensors
# ==========================================================================

YIELD_BG = 31.0 / 255.0


def yield_greys(L):
    """The organiser yield rule (src/cad/yield_model.py): grey rises linearly
    with layer number. Used only to render a CAD reference for the 'int' family."""
    if L <= 1:
        return np.array([0.85])
    return 0.20 + (0.85 - 0.20) * np.arange(L) / (L - 1)


def boundary_band(M, sigma=1.0):
    """One extra basis image: where any layer boundary runs. A real SEM brightens
    edges, which a per-layer piecewise-constant fit cannot represent at all, so the
    band gets its own fitted level (the i4c generator has no edge brightening; our
    stream adds it in 50% of samples)."""
    return grad_mag(M, sigma).sum(1, keepdim=True)


def fit_greys(masks, patch, edge_band=None, ridge=1e-3):
    """Least-squares per-layer greys from ONE window: patch ~ c + sum_l g_l M_l (+ g_b B).

    This is the yield raster the Phase 3 spec asks you to infer. Estimated once from
    the best coarse candidate and then reused, instead of re-solved at every position:
    fewer free parameters at scoring time means a wrong location can no longer explain
    the window by re-fitting its greys."""
    L = masks.shape[0]
    cols = [masks.reshape(L, -1)]
    if edge_band is not None:
        cols.append(edge_band.reshape(1, -1))
    cols.append(torch.ones(1, masks[0].numel(), device=masks.device))
    X = torch.cat(cols, 0).double()
    y = patch.reshape(-1).double()
    G = X @ X.T + ridge * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    g = torch.linalg.solve(G, X @ y)
    fit = (X.T @ g)
    ss = ((y - y.mean()) ** 2).sum().clamp_min(1e-9)
    r2 = float(1 - ((y - fit) ** 2).sum() / ss)
    return g.float(), r2


class StageA:
    """Everything between (search, reference) and the model's input tensors."""

    def __init__(self, cfg, device, edge_net=None):
        self.cfg, self.dev = cfg, device
        self.edge_net = edge_net
        self.greys = None                 # set per pair by the 'mtm2' coarse pass
        self.poses = poses_of(cfg)
        self.nz, self.nt = len(cfg['zooms']), len(cfg['thetas'])
        self.fams = cfg['families']
        self.need_edgefit = ('edgefit' in self.fams or cfg['raw_family'] == 'edgefit'
                             or 'edgefit' in cfg['refine_families'])
        self.need_edgefit2 = ('edgefit2' in self.fams or 'edgefit2' in cfg['refine_families'])
        self.need_edge = ('edge' in self.fams or cfg['raw_family'] == 'edge'
                          or 'edge' in cfg['refine_families']
                          or any(c.endswith('_edge') for c in cfg['patch_channels']))
        self.fit = any(c in ('T_fit', 'R_fit') for c in cfg['patch_channels'])
        self.two_stage = uses_greys(cfg)

    # ---- edge maps
    def _edge(self, x):
        """x: (N, 1, H, W) intensity in [0, 1] -> boundary map."""
        if self.edge_net is not None:
            return self.edge_net.boundary(x)
        return grad_mag(x, self.cfg['sigma_edge'])

    # ---- search side
    def search_prep(self, search_u8):
        s = torch.from_numpy(np.ascontiguousarray(search_u8)).to(self.dev).float()[None, None] / 255.0
        out = {'raw': s[0, 0], 'int': prefilter(s, self.cfg)[0, 0]}
        if self.need_edge:
            out['edge'] = self._edge(s)[0, 0]
        if self.need_edgefit:
            out['edgefit'] = gblur(grad_mag(s, self.cfg['sigma_edge']), self.cfg['sigma_band'])[0, 0]
        if self.need_edgefit2:
            dx, dy = deriv_xy(s, self.cfg['sigma_edge'])
            out['edgefit2_x'] = gblur(dx, self.cfg['sigma_band'])[0, 0]
            out['edgefit2_y'] = gblur(dy, self.cfg['sigma_band'])[0, 0]
        return out

    # ---- template side
    def templates(self, small, kind, pose_list=None):
        """Raw warped templates grouped by size: {n: (pose idx list, (B, C, n, n))}."""
        pose_list = self.poses if pose_list is None else pose_list
        groups = {}
        for pi, (z, t) in enumerate(pose_list):
            arr = pose_warp(small, z, t)
            groups.setdefault(arr.shape[-1], []).append((pi, arr))
        return {n: ([p for p, _ in lst], torch.from_numpy(np.stack([a for _, a in lst])).to(self.dev))
                for n, lst in groups.items()}

    def fam_templates(self, T, kind):
        """T: (B, C, n, n) raw -> dict family -> template tensor."""
        cfg = self.cfg
        out = {}
        if kind == 'sem':
            ti = T[:, :1] / 255.0
        else:                                   # cad: C soft layer masks
            g = torch.tensor(yield_greys(T.shape[1]), device=T.device, dtype=torch.float32)
            ti = YIELD_BG + ((g - YIELD_BG)[None, :, None, None] * T).sum(1, keepdim=True)
        out['int_raw'] = ti[:, 0]
        out['int'] = prefilter(ti, cfg)[:, 0]
        if self.need_edge:
            if kind == 'sem':
                out['edge'] = self._edge(ti)[:, 0]
            else:
                # grey-free CAD boundary map: every layer change is a boundary
                out['edge'] = grad_mag(T, cfg['sigma_edge']).sum(1)
        if self.need_edgefit2:
            src = T if kind == 'cad' else T[:, :1] / 255.0
            dx, dy = deriv_xy(src, cfg['sigma_edge'])
            out['edgefit2_x'] = gblur(dx, cfg['sigma_band'])
            out['edgefit2_y'] = gblur(dy, cfg['sigma_band'])
        if self.need_edgefit:
            # Per-layer EDGE bases. A real SEM's step at a boundary depends on which two
            # layers meet there, so a flat "all boundaries equal" map correlates badly
            # against |grad SEM|. Give every layer its own edge image and let least
            # squares solve its edge strength -- the tone-map trick in the gradient
            # domain. Grey-free: no brightness assumption, immune to gamma/contrast.
            src = T if kind == 'cad' else T[:, :1] / 255.0
            out['edgefit'] = gblur(grad_mag(src, cfg['sigma_edge']), cfg['sigma_band'])
        if kind == 'cad':
            Mb = prefilter(T, cfg, values=False)
            if cfg['mtm_edge_band']:
                Mb = torch.cat([Mb, boundary_band(Mb, cfg['sigma_edge'])], 1)
            out['mtm'] = Mb
            out['mtm_raw'] = T
            if self.greys is not None and 'mtm2' in self.fams:   # fixed render from the estimated yield raster
                Mb = out['mtm']
                r = (Mb * self.greys[:Mb.shape[1], None, None]).sum(1, keepdim=True)
                if cfg['mtm_edge_band'] and len(self.greys) > Mb.shape[1]:
                    r = r + self.greys[Mb.shape[1]] * boundary_band(Mb, cfg['sigma_edge'])
                out['mtm2'] = r[:, 0]
        return out

    def family_maps(self, sctx, tmpl_groups, kind, fams, raw=False, pool=1):
        """-> {family: [map per pose]}. `pool` downsamples for the coarse pass."""
        P = max((max(ix) for ix, _ in tmpl_groups.values() if len(ix)), default=-1) + 1
        maps = {f: [None] * P for f in fams}
        for n, (idx, T) in tmpl_groups.items():
            if raw:
                T = F.avg_pool2d(T, raw_pool(fams[0]))
            elif pool > 1:
                T = F.avg_pool2d(T, pool)
                if min(T.shape[-2:]) < 8:
                    continue
            ft = self.fam_templates(T, kind)
            for f in fams:
                if f == 'edgefit2':
                    mm = mtm_maps_multi([sctx['edgefit2_x'], sctx['edgefit2_y']],
                                        [ft['edgefit2_x'], ft['edgefit2_y']])
                elif f == 'edgefit':
                    mm = mtm_maps(sctx['edgefit_raw' if raw else 'edgefit'], ft['edgefit'])
                elif f == 'mtm2':
                    mm = zncc_maps(sctx['mtm2_raw' if raw else 'int'], ft['mtm2'])
                elif f == 'mtm':
                    mm = mtm_maps(sctx['mtm_raw' if raw else 'int'], ft['mtm_raw' if raw else 'mtm'])
                elif f == 'int':
                    mm = zncc_maps(sctx['int_raw' if raw else 'int'], ft['int_raw' if raw else 'int'])
                else:
                    mm = zncc_maps(sctx['edge_raw' if raw else 'edge'], ft['edge'])
                for j, pi in enumerate(idx):
                    maps[f][pi] = mm[j]
        return maps

    def pooled_ctx(self, sp, fams, pool):
        """Contexts on the prepared channels downsampled by `pool` -- for the coarse
        pose-pruning pass. Blur-then-pool and pool-then-blur differ only slightly, and
        the coarse pass only has to RANK poses."""
        ctx = {}
        pl = lambda k: F.avg_pool2d(sp[k][None, None], pool)[0, 0]
        if {'int', 'mtm', 'mtm2'} & set(fams):
            ctx['int'] = SearchCtx(pl('int'))
        if 'edge' in fams:
            ctx['edge'] = SearchCtx(pl('edge'))
        if 'edgefit' in fams:
            ctx['edgefit'] = SearchCtx(pl('edgefit'))
        if 'edgefit2' in fams:
            ctx['edgefit2_x'] = SearchCtx(pl('edgefit2_x'))
            ctx['edgefit2_y'] = SearchCtx(pl('edgefit2_y'))
        return ctx

    def contexts(self, sp, fams, raw=False):
        ctx = {}
        if raw:
            for f in fams:
                half = F.avg_pool2d(sp['raw'][None, None], raw_pool(f))
                if f == 'edge':
                    half = self._edge(half) if self.edge_net is not None else grad_mag(half, 0.8)
                elif f in ('edgefit', 'edgefit2'):
                    half = grad_mag(half, 0.8)
                ctx[f + '_raw'] = SearchCtx(half[0, 0])
            return ctx
        if {'int', 'mtm', 'mtm2'} & set(fams):
            ctx['int'] = SearchCtx(sp['int'])
        if 'edge' in fams:
            ctx['edge'] = SearchCtx(sp['edge'])
        if 'edgefit' in fams:
            ctx['edgefit'] = SearchCtx(sp['edgefit'])
        if 'edgefit2' in fams:
            ctx['edgefit2_x'] = SearchCtx(sp['edgefit2_x'])
            ctx['edgefit2_y'] = SearchCtx(sp['edgefit2_y'])
        return ctx

    def prior_xy(self, prior):
        """The search-CAD hit, mapped into SEM pixels for every pose on the grid.

        The two CADs share frame, scale and orientation, so the geometry match gives
        the reference's canvas-frame origin to a few nm; only the pose is unknown, and
        the grid already enumerates it. One prior candidate per pose."""
        x0, y0, cs = float(prior['x0']), float(prior['y0']), float(prior['canvas_px'])
        # the canvas size is not stored in the GDS, so a bounding box that stops short of
        # the canvas edge offsets this mapping; p3_data measures the residual translation
        sx, sy = prior.get('shift', (0.0, 0.0))
        out = []
        for (z, t) in self.poses:
            rad = math.radians(t)
            c_, s_ = math.cos(rad), math.sin(rad)
            cc, sc = (cs - 1) / 2.0, (SEARCH_PX - 1) / 2.0
            px = (c_ * (x0 + (REF_PX - 1) / 2.0 - cc) + s_ * (y0 + (REF_PX - 1) / 2.0 - cc)) / z + sc + sx
            py = (-s_ * (x0 + (REF_PX - 1) / 2.0 - cc) + c_ * (y0 + (REF_PX - 1) / 2.0 - cc)) / z + sc + sy
            out.append((px, py))
        return out

    @torch.no_grad()
    def __call__(self, search_u8, small, kind, gt=None, prior=None, gdsg=None):
        cfg = self.cfg
        K = cfg['k']
        sp = self.search_prep(search_u8)
        gmode = cfg['use_global_pose'] and gdsg is not None
        full_poses = self.poses
        if gmode:
            # the whole-field fit has already fixed the pose; search translation only
            self.poses = [(float(gdsg['zoom']), float(gdsg['theta']))]
        nz_, nt_ = (1, 1) if gmode else (self.nz, self.nt)
        groups = self.templates(small, kind)
        sizes = {pi: n for n, (idx, _) in groups.items() for pi in idx}
        self.greys, yield_stats = None, [0.0, 0.0, 0.0]
        if self.two_stage:
            self.greys, yield_stats = self._estimate_greys(sp, groups, sizes)
        all_groups = groups
        keep_poses, coarse_ppm = None, {}
        if cfg['pose_prune_keep'] and cfg['pose_prune_keep'] < len(self.poses):
            ds = cfg['pose_prune_ds']
            cmaps = self.family_maps(self.pooled_ctx(sp, self.fams, ds), groups, kind, self.fams, pool=ds)
            # where the search-CAD says the reference is, per pose, in coarse-map pixels
            wins = None
            if cfg['use_cad_prior'] and prior is not None and cfg['pose_prune_r']:
                r = max(int(round(cfg['pose_prune_r'] / ds)), 1)
                wins = []
                for pi, (px, py) in enumerate(self.prior_xy(prior)):
                    n = sizes.get(pi, cfg['patch'])
                    cy, cx = (py - (n - 1) / 2.0) / ds, (px - (n - 1) / 2.0) / ds
                    wins.append((int(round(cy)), int(round(cx)), r))

            def pose_peak(m, pi):
                if m is None:
                    return -9.0
                if wins is None:
                    return float(m.max())
                cy, cx, r = wins[pi]
                sub = m[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
                return float(sub.max()) if sub.numel() else float(m.max())

            score = np.full(len(self.poses), -9.0, np.float32)
            for f in self.fams:
                ppm = np.array([pose_peak(m, pi) for pi, m in enumerate(cmaps[f])], np.float32)
                coarse_ppm[f] = np.array([float(m.max()) if m is not None else -9.0
                                          for m in cmaps[f]], np.float32)   # presence: whole field
                score = np.maximum(score, ppm)
            keep_poses = sorted(np.argsort(-score)[:cfg['pose_prune_keep']].tolist())
            groups = {n: ([p for p in idx if p in keep_poses],
                          T[[i for i, p in enumerate(idx) if p in keep_poses]])
                      for n, (idx, T) in groups.items()}
            groups = {n: g for n, g in groups.items() if g[0]}
        maps = self.family_maps(self.contexts(sp, self.fams), groups, kind, self.fams)

        # ---- global pose-surface statistics (presence evidence)
        glob_feats = []
        for f in self.fams:
            if keep_poses is not None:
                ppm = coarse_ppm[f]                       # all poses, from the coarse pass
                best_map = next(m for m in maps[f] if m is not None)
            else:
                ppm = np.array([float(m.max()) for m in maps[f]], np.float32)
                best_map = maps[f][int(ppm.argmax())]
            glob_feats += surface_stats(ppm, nz_, nt_, best_map)
        rf = cfg['raw_family']
        rmaps = self.family_maps(self.contexts(sp, [rf], raw=True), all_groups, kind, [rf], raw=True)
        ppm = np.array([float(m.max()) for m in rmaps[rf] if m is not None], np.float32)
        rvalid = [m for m in rmaps[rf] if m is not None]
        glob_feats += surface_stats(ppm, nz_, nt_, rvalid[int(ppm.argmax())])
        if self.two_stage:
            glob_feats += yield_stats          # how well the inferred yield raster fits

        # ---- candidates: equal quota per family, dedupe on (pose, iy, ix)
        per_fam = {}
        for f in self.fams:
            rows = []
            for pi, m in enumerate(maps[f]):
                if m is None:
                    continue
                v, iy, ix = peaks(m, cfg['per_pose'], cfg['nms_r'])
                rows += [(a, pi, b, c) for a, b, c in zip(v, iy, ix)]
            rows.sort(key=lambda r: -r[0])
            per_fam[f] = rows
        quota = K // len(self.fams)
        chosen, seen = [], set()
        for fi, f in enumerate(self.fams):
            take = quota if fi < len(self.fams) - 1 else K - len(chosen)
            got = 0
            for r in per_fam[f]:
                if r[1:] in seen:
                    continue
                seen.add(r[1:])
                chosen.append(r[1:])
                got += 1
                if got >= take:
                    break
        for f in self.fams:                      # top up if a family ran short
            for r in per_fam[f]:
                if len(chosen) >= K:
                    break
                if r[1:] not in seen:
                    seen.add(r[1:])
                    chosen.append(r[1:])
        # candidates from the search-CAD geometry: one per pose, inserted before padding
        if self.cfg['use_cad_prior'] and prior is not None:
            for pi, (px, py) in enumerate(self.prior_xy(prior)):
                if keep_poses is not None and pi not in keep_poses:
                    continue
                n = sizes.get(pi, cfg["patch"])
                iy_, ix_ = int(round(py - (n - 1) / 2.0)), int(round(px - (n - 1) / 2.0))
                lim = SEARCH_PX - n
                if not (0 <= ix_ <= lim and 0 <= iy_ <= lim) or (pi, iy_, ix_) in seen:
                    continue
                seen.add((pi, iy_, ix_))
                chosen.insert(0, (pi, iy_, ix_))
            chosen = chosen[:K]
        cad_idx = {}
        if gmode:
            z0, t0 = self.poses[0]
            cs = float(gdsg['canvas_px'])
            rad = math.radians(t0)
            c_, s_ = math.cos(rad), math.sin(rad)
            cc, sc = (cs - 1) / 2.0, (SEARCH_PX - 1) / 2.0
            sx, sy = gdsg.get('shift', (0.0, 0.0))
            n = sizes.get(0, cfg['patch'])
            for (x0, y0, sig) in gdsg['peaks']:
                X, Y = x0 + (REF_PX - 1) / 2.0 - cc, y0 + (REF_PX - 1) / 2.0 - cc
                px, py = (c_ * X + s_ * Y) / z0 + sc + sx, (-s_ * X + c_ * Y) / z0 + sc + sy
                iy_, ix_ = int(round(py - (n - 1) / 2.0)), int(round(px - (n - 1) / 2.0))
                if not (0 <= ix_ <= SEARCH_PX - n and 0 <= iy_ <= SEARCH_PX - n):
                    continue
                key = (0, iy_, ix_)
                if key not in seen:
                    seen.add(key)
                    chosen.insert(0, key)
                cad_idx[key] = sig / max(float(gdsg['top_sigma']), 1e-9)
            chosen = chosen[:K]
        valid = np.zeros(K, np.float32)
        valid[:len(chosen)] = 1.0
        while len(chosen) < K:
            chosen.append(chosen[-1] if chosen else (0, 0, 0))

        pidx = np.array([c[0] for c in chosen])
        iy = np.array([c[1] for c in chosen])
        ix = np.array([c[2] for c in chosen])
        n_side = np.array([sizes[p] for p in pidx], np.float32)
        cx = ix + (n_side - 1) / 2.0
        cy = iy + (n_side - 1) / 2.0
        zc = np.array([self.poses[p][0] for p in pidx], np.float32)
        tc = np.array([self.poses[p][1] for p in pidx], np.float32)
        scores = np.zeros((K, len(self.fams)), np.float32)
        for fi, f in enumerate(self.fams):
            for j in range(K):
                scores[j, fi] = float(maps[f][pidx[j]][iy[j], ix[j]])

        # ---- patches
        chans = self._patches(sp, groups, kind, pidx, cx, cy, n_side)
        fitsc = None
        if self.fit:
            chans, fitsc = self._fit(chans)
        stack = torch.stack([chans[c] for c in cfg['patch_channels']], 1)   # K, C, P, P
        for ci, c in enumerate(cfg['patch_channels']):
            if c not in ('T_fit', 'R_fit'):
                x = stack[:, ci]
                stack[:, ci] = (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-6)

        scal = self._scalars(scores, pidx, cx, cy, zc, tc, n_side, fitsc)
        if cfg['use_global_pose']:
            rel = np.array([cad_idx.get((int(p), int(a), int(b)), 0.0) for p, a, b in zip(pidx, iy, ix)], np.float32)
            scal = np.concatenate([scal, np.stack([(rel > 0).astype(np.float32), rel], 1)], 1)
            if gmode:
                # field R^2 is NOT used: training simulates the global pose cheaply (truth +
                # measured error) and has no R^2, so feeding it at inference would be a
                # train/test feature mismatch. The geometry terms are computed identically.
                glob_feats += [float(gdsg['top_sigma']) / 10.0, float(gdsg['runner_up']),
                               float(gdsg['ties']) / 5.0, 0.0]
            else:
                glob_feats += [0.0, 1.0, 1.0, 0.0]
        if self.cfg['use_cad_prior']:
            pxy = np.array(self.prior_xy(prior), np.float32) if prior is not None else None
            dist = (np.hypot(cx - pxy[pidx, 0], cy - pxy[pidx, 1]) if pxy is not None
                    else np.full(len(cx), 999.0, np.float32))
            sig = float(prior['sigma']) if prior is not None else 0.0
            riv = float(prior['rivals']) if prior is not None else 1e6
            scal = np.concatenate([scal, np.stack([
                np.exp(-dist / 5.0), np.full(len(cx), sig / 10.0, np.float32),
                np.full(len(cx), math.log1p(riv) / 10.0, np.float32)], 1).astype(np.float32)], 1)
            glob_feats += [sig / 10.0, math.log1p(riv) / 10.0,
                           float(np.exp(-dist.min() / 5.0)) if len(dist) else 0.0]
        geom = np.stack([cx / SEARCH_PX, cy / SEARCH_PX, (zc - 10.0) / 2.0, tc / 5.0], 1).astype(np.float32)
        out = dict(patch=stack.half().cpu(), scalars=torch.from_numpy(scal),
                   scores=torch.from_numpy(scores), geom=torch.from_numpy(geom),
                   valid=torch.from_numpy(valid), selfcorr=self._selfcorr(sp['int']).cpu(),
                   glob=torch.tensor(glob_feats, dtype=torch.float32),
                   cand=np.stack([cx, cy, zc, tc], 1).astype(np.float32),
                   greys=(self.greys.cpu() if self.greys is not None else None))
        out.update(self._labels(gt, out['cand'], valid))
        out['gmode'] = gmode
        self.poses = full_poses
        return out

    def _estimate_greys(self, sp, groups, sizes):
        """Stage A1 for 'mtm2': locate coarsely with the free per-window tone-map fit at
        1/ds resolution, then estimate ONE set of per-layer greys (the yield raster) from
        the best window at full resolution. Returns (greys, [R2, order_corr, resid])."""
        cfg, ds = self.cfg, self.cfg['mtm_coarse_ds']
        half = F.avg_pool2d(sp['int'][None, None], ds)
        ctx = SearchCtx(half[0, 0])
        best = (-1.0, None, 0, 0)
        for n, (idx, T) in groups.items():
            Md = F.avg_pool2d(prefilter(T, cfg, values=False), ds)
            if min(Md.shape[-2:]) < 8 or Md.shape[-1] >= ctx.W:
                continue
            mm = mtm_maps(ctx, Md)
            for j, pi in enumerate(idx):
                v = float(mm[j].max())
                if v > best[0]:
                    iy, ix = np.unravel_index(int(mm[j].argmax()), mm[j].shape)
                    best = (v, pi, int(iy) * ds, int(ix) * ds)
        if best[1] is None:
            return None, [0.0, 0.0, 0.0]
        _, pi, y0, x0 = best
        n = sizes[pi]
        for nn, (idx, T) in groups.items():
            if pi in idx:
                Mb = prefilter(T[idx.index(pi):idx.index(pi) + 1], cfg, values=False)[0]
                break
        y0 = int(np.clip(y0, 0, SEARCH_PX - n))
        x0 = int(np.clip(x0, 0, SEARCH_PX - n))
        patch = sp['int'][y0:y0 + n, x0:x0 + n]
        band = boundary_band(Mb[None], cfg['sigma_edge'])[0, 0] if cfg['mtm_edge_band'] else None
        g, r2 = fit_greys(Mb, patch, band)
        L = Mb.shape[0]
        gl = g[:L].cpu().numpy()
        cover = Mb.mean((1, 2)).cpu().numpy()
        seen = np.where(cover > 0.01)[0]
        order = 0.0
        if len(seen) >= 2:
            rk = np.argsort(np.argsort(gl[seen])).astype(np.float64)
            order = float(np.corrcoef(rk, np.arange(len(seen)))[0, 1]) if rk.std() > 0 else 0.0
        return g, [r2, order, float(np.std(gl))]

    def _grid(self, cx, cy, n_side, H, W):
        P = self.cfg['patch']
        u = (torch.arange(P, device=self.dev, dtype=torch.float32) + 0.5) / P
        n = torch.from_numpy(n_side).to(self.dev)
        x0 = torch.from_numpy(cx.astype(np.float32)).to(self.dev) - n / 2.0
        y0 = torch.from_numpy(cy.astype(np.float32)).to(self.dev) - n / 2.0
        gx = (x0[:, None] + u[None] * n[:, None]) / (W - 1) * 2 - 1
        gy = (y0[:, None] + u[None] * n[:, None]) / (H - 1) * 2 - 1
        return torch.stack([gx[:, None, :].expand(-1, P, -1), gy[:, :, None].expand(-1, -1, P)], -1)

    def _patches(self, sp, groups, kind, pidx, cx, cy, n_side):
        cfg, P, K = self.cfg, self.cfg['patch'], len(pidx)
        want = set(cfg['patch_channels'])
        if self.fit:
            want |= {'S_int', 'M'}
        chans = {}
        H, W = sp['int'].shape
        g1 = self._grid(cx, cy, n_side, H, W).reshape(1, K * P, P, 2)
        for name, key in (('S_int', 'int'), ('S_edge', 'edge')):
            if name in want:
                o = F.grid_sample(sp[key][None, None], g1, mode='bilinear', padding_mode='border',
                                  align_corners=True)
                chans[name] = o.reshape(K, P, P)
        tnames = [c for c in ('T_int', 'T_edge', 'M') if c in want]
        if tnames:
            store = {t: torch.zeros(K, P, P, device=self.dev) for t in tnames if t != 'M'}
            masks = None
            for n, (idx, T) in groups.items():
                sel = [j for j in range(K) if pidx[j] in idx]
                if not sel:
                    continue
                ft = self.fam_templates(T, kind)
                pos = {p: i for i, p in enumerate(idx)}
                rows = torch.tensor([pos[pidx[j]] for j in sel], device=self.dev)
                seli = torch.tensor(sel, device=self.dev)
                if 'T_int' in tnames:
                    r = F.interpolate(ft['int'][:, None], size=(P, P), mode='bilinear', align_corners=False)
                    store['T_int'][seli] = r[rows, 0]
                if 'T_edge' in tnames:
                    r = F.interpolate(ft['edge'][:, None], size=(P, P), mode='bilinear', align_corners=False)
                    store['T_edge'][seli] = r[rows, 0]
                if 'M' in tnames:
                    r = F.interpolate(ft['mtm'], size=(P, P), mode='bilinear', align_corners=False)
                    if masks is None:
                        masks = torch.zeros(K, r.shape[1], P, P, device=self.dev)
                    masks[seli] = r[rows]
            chans.update(store)
            if masks is not None:
                chans['M'] = masks
        return chans

    def _fit(self, chans):
        """Per-candidate yield fit: S_patch ~ c + sum_l g_l M_l (ridge LS)."""
        M, y = chans['M'], chans['S_int']
        K, L, P, _ = M.shape
        X = torch.cat([M.flatten(2), torch.ones(K, 1, P * P, device=self.dev)], 1)
        G = X @ X.transpose(1, 2) + 1e-3 * torch.eye(L + 1, device=self.dev)
        b = X @ y.flatten(1)[:, :, None]
        g = torch.linalg.solve(G, b)[:, :, 0]
        fit = (X.transpose(1, 2) @ g[:, :, None])[:, :, 0].view(K, P, P)
        res = y - fit
        ysd = y.flatten(1).std(1).clamp_min(1e-6)
        sst = ((y - y.mean((1, 2), keepdim=True)) ** 2).flatten(1).sum(1).clamp_min(1e-9)
        r2 = 1 - (res ** 2).flatten(1).sum(1) / sst
        chans['T_fit'] = (fit - fit.mean((1, 2), keepdim=True)) / ysd[:, None, None]
        chans['R_fit'] = res / ysd[:, None, None]
        cover = M.mean((2, 3)).cpu().numpy()
        gl = g[:, :L].cpu().numpy()
        sc = np.zeros((K, 4), np.float32)
        for j in range(K):
            present = np.where(cover[j] > 0.01)[0]
            if len(present) >= 2:
                gg = gl[j, present]
                rk = np.argsort(np.argsort(gg)).astype(np.float32)
                li = np.arange(len(present), dtype=np.float32)
                sc[j, 1] = float(np.corrcoef(rk, li)[0, 1]) if rk.std() > 0 else 0.0
                sc[j, 2] = float(np.mean(np.diff(gg) < 0))
        sc[:, 0] = r2.cpu().numpy()
        sc[:, 3] = (res.flatten(1).std(1) / ysd).cpu().numpy()
        return chans, sc

    def _selfcorr(self, s_int):
        s = F.adaptive_avg_pool2d(s_int[None, None], 256)[0, 0]
        s = s - s.mean()
        Fk = torch.fft.rfft2(s)
        ac = torch.fft.fftshift(torch.fft.irfft2(Fk * torch.conj(Fk), s=s.shape))
        ac = ac / (ac.abs().max() + 1e-9)
        return F.adaptive_avg_pool2d(ac[None, None], 64)[0, 0].float()

    def _scalars(self, scores, pidx, cx, cy, zc, tc, n_side, fitsc):
        K = len(pidx)
        cfg = self.cfg
        p = scores[:, 0]
        zmid = 0.5 * (cfg['zoom_range'][0] + cfg['zoom_range'][1])
        zspan = max(1e-6, 0.5 * (cfg['zoom_range'][1] - cfg['zoom_range'][0]))
        tspan = max(1e-6, max(abs(cfg['theta_range'][0]), abs(cfg['theta_range'][1])))
        npx = n_side ** 2
        fisher = np.arctanh(np.clip(p, -0.999, 0.999)) * np.sqrt(np.maximum(npx - 3, 1))
        cols = [p, p / (abs(p.max()) + 1e-6), np.arange(K) / K,
                np.hypot(cx - SEARCH_PX / 2, cy - SEARCH_PX / 2) / (SEARCH_PX / 2),
                (zc - zmid) / zspan, tc / tspan, (p - p.mean()) / (p.std() + 1e-6),
                np.log(npx) / 10.0, fisher / 200.0, np.full(K, p.std())]
        for fi in range(1, scores.shape[1]):
            s = scores[:, fi]
            cols += [s, (s - s.mean()) / (s.std() + 1e-6)]
        out = np.stack(cols, 1).astype(np.float32)
        if fitsc is not None:
            out = np.concatenate([out, fitsc], 1)
        return out

    def _labels(self, gt, cand, valid):
        cfg = self.cfg
        K = len(cand)
        w = np.zeros(K, np.float32)
        off = np.zeros((K, 4), np.float32)
        if gt is None:
            gt = [0, 0, 0, 0, 0]
        present, gx, gy, gth, gz = [float(v) for v in gt]
        zs = cfg['zooms'][1] - cfg['zooms'][0] if len(cfg['zooms']) > 1 else 1.0
        ts = cfg['thetas'][1] - cfg['thetas'][0] if len(cfg['thetas']) > 1 else 1.0
        if present > 0.5:
            d = np.hypot(cand[:, 0] - gx, cand[:, 1] - gy)
            dz, dt = (gz - cand[:, 2]) / zs, (gth - cand[:, 3]) / ts
            pos = (d <= TOL_PX) & (valid > 0)
            w = np.where(pos, np.exp(-d * d / 8.0 - 2.0 * (dz * dz + dt * dt)), 0.0).astype(np.float32)
            off = np.stack([(gx - cand[:, 0]) / TOL_PX, (gy - cand[:, 1]) / TOL_PX, dz, dt], 1).astype(np.float32)
        return dict(target=torch.from_numpy(w), offset=torch.from_numpy(off),
                    y=torch.tensor(present, dtype=torch.float32),
                    gt=torch.tensor([present, gx, gy, gth, gz], dtype=torch.float32))


def n_scalars(cfg):
    n = 10 + 2 * (len(cfg['families']) - 1) + (3 if cfg['use_cad_prior'] else 0) + \
        (2 if cfg['use_global_pose'] else 0)
    if any(c in ('T_fit', 'R_fit') for c in cfg['patch_channels']):
        n += 4
    return n


def uses_greys(cfg):
    """Is the coarse yield-raster pass on? It adds 3 glob features (R^2 of the inferred
    yield fit, agreement of the fitted greys with layer order, their spread)."""
    return 'mtm2' in cfg['families'] or (cfg['estimate_greys'] and cfg['phase'] == 'p3')


def n_glob(cfg):
    return 5 * (len(cfg['families']) + 1) + (3 if uses_greys(cfg) else 0) + (3 if cfg['use_cad_prior'] else 0) + \
        (4 if cfg['use_global_pose'] else 0)


# ==========================================================================
# Model
# ==========================================================================

class Consensus(nn.Module):
    """One set-attention layer over the K candidates with a learned relative
    geometry bias -- "same site, one lattice period apart" is expressible here and
    not in per-candidate scoring (NCNet arXiv:1810.10510, Sparse-NCNet 2004.10566).
    Zero-initialised residual, so it starts as the identity."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.geo = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, heads))
        self.proj = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        for m in (self.proj, self.ff[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, e, geom, valid):
        B, K, d = e.shape
        q, k, v = self.qkv(self.n1(e)).chunk(3, -1)
        sh = lambda t: t.view(B, K, self.h, d // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        rel = geom[:, :, None, :] - geom[:, None, :, :]
        dist = torch.sqrt((rel[..., :2] ** 2).sum(-1, keepdim=True) + 1e-9)
        same = ((rel[..., 2:].abs().sum(-1, keepdim=True)) < 1e-4).float()
        bias = self.geo(torch.cat([rel, dist, same], -1)).permute(0, 3, 1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h) + bias
        att = att.masked_fill(valid[:, None, None, :] < 0.5, -1e4)
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, K, d)
        e = e + self.proj(y)
        return e + self.ff(self.n2(e))


class CrossEncoder(nn.Module):
    """Early fusion (template + search channels stacked at layer 1), VALID convs,
    classical-score warm start. Early fusion beat late fusion by 0.27 hard-regime
    accuracy in Phase 1 (DeepCompare arXiv:1504.03641 found the same); keeping the
    template's full spatial layout beats pooling it (TMR 2508.17636)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ch, d = cfg['ch'], cfg['d']
        cin = len(cfg['patch_channels']) + 3            # + coord x, coord y, self-correlation
        self.body = nn.Sequential(
            nn.Conv2d(cin, ch, 5, stride=2), nn.GroupNorm(4, ch), nn.GELU(),
            nn.Conv2d(ch, 2 * ch, 3, stride=2), nn.GroupNorm(4, 2 * ch), nn.GELU(),
            nn.Conv2d(2 * ch, 2 * ch, 3), nn.GroupNorm(4, 2 * ch), nn.GELU(),
            nn.Conv2d(2 * ch, 4 * ch, 3, stride=2), nn.GroupNorm(4, 4 * ch), nn.GELU(),
            nn.Conv2d(4 * ch, 4 * ch, 3), nn.GroupNorm(4, 4 * ch), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        ns, ng, nf = n_scalars(cfg), n_glob(cfg), len(cfg['families'])
        self.embed = nn.Sequential(nn.Linear(4 * ch + ns + 4, d), nn.GELU(), nn.Linear(d, d))
        self.consensus = Consensus(d) if cfg['consensus'] else None
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.alpha = nn.Parameter(torch.full((nf,), cfg['alpha_init'] / nf))
        self.presence = nn.Sequential(nn.Linear(2 * d + 3 + ng, d), nn.GELU(), nn.Linear(d, 1))
        # residual pose from the PATCH PAIR ONLY -- refiners that see only the two
        # patches transfer across candidate generators (XRefine arXiv:2601.12530)
        self.offset = (nn.Sequential(nn.Linear(4 * ch, d), nn.GELU(), nn.Linear(d, 4))
                       if cfg['offset_head'] else None)
        self.log_var = nn.Parameter(torch.zeros(3))

    def forward(self, b):
        patch, scal, geom, valid = b['patch'].float(), b['scalars'], b['geom'], b['valid']
        B, K, C, P, _ = patch.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, P, device=patch.device),
                                torch.linspace(-1, 1, P, device=patch.device), indexing='ij')
        sc = F.interpolate(b['selfcorr'][:, None], size=(P, P), mode='bilinear', align_corners=False)
        extra = torch.cat([xs[None, None].expand(B, 1, P, P), ys[None, None].expand(B, 1, P, P), sc], 1)
        x = torch.cat([patch, extra[:, None].expand(B, K, 3, P, P)], 2).view(B * K, C + 3, P, P)
        f = self.body(x).view(B, K, -1)
        s = scal
        if self.training and self.cfg['scalar_dropout'] > 0:
            # without this the head echoes the warm-start score and ignores pixels
            s = s * (torch.rand(B, K, 1, device=s.device) > self.cfg['scalar_dropout']).float()
        e = self.embed(torch.cat([f, s, geom], -1))
        if self.consensus is not None:
            e = self.consensus(e, geom, valid)
        logit = self.head(e)[..., 0] + (self.alpha[None, None] * b['scores']).sum(-1)
        logit = logit.masked_fill(valid < 0.5, -1e4)
        vm = valid[..., None]
        mean_e = (e * vm).sum(1) / vm.sum(1).clamp_min(1)
        max_e = e.masked_fill(vm < 0.5, -1e4).max(1).values
        top2 = logit.topk(2, -1).values
        ent = -(logit.softmax(-1) * logit.log_softmax(-1)).sum(-1, keepdim=True)
        pres = self.presence(torch.cat([mean_e, max_e, torch.logsumexp(logit, -1, keepdim=True),
                                        top2[:, :1] - top2[:, 1:2], ent, b['glob']], -1))[..., 0]
        out = {'logit': logit, 'presence': pres}
        if self.offset is not None:
            out['offset'] = self.offset(f)
        return out


# ==========================================================================
# Losses
# ==========================================================================


# ==========================================================================
# Batching + modality dropout
# ==========================================================================

BATCH_KEYS = ('patch', 'scalars', 'scores', 'geom', 'valid', 'selfcorr', 'glob',
              'target', 'offset', 'y')


# ==========================================================================
# Inference: pick -> residual pose -> refine -> calibrated found flag
# ==========================================================================

@torch.no_grad()
def refine_translate(stagea, item, cand, families, R=6):
    """Translation-only sub-pixel refinement at a FIXED pose (the whole-field pose is
    ~10x better constrained than anything a 100 px window can give -- measured: local
    re-estimation raised the median scale error from 0.01% to 0.19%)."""
    cfg = stagea.cfg
    cx, cy, z, t = [float(v) for v in cand]
    groups = stagea.templates(item['small'], item['kind'], [(z, t)])
    n, (idx, T) = next(iter(groups.items()))
    sp = stagea.search_prep(item['search'])
    x0 = int(np.clip(int(round(cx - (n - 1) / 2.0)) - R, 0, SEARCH_PX - n - 2 * R))
    y0 = int(np.clip(int(round(cy - (n - 1) / 2.0)) - R, 0, SEARCH_PX - n - 2 * R))
    sub = {k: v[y0:y0 + n + 2 * R, x0:x0 + n + 2 * R] for k, v in sp.items()}
    ctx = stagea.contexts(sub, families)
    ft = stagea.fam_templates(T, item['kind'])
    acc = None
    for f in families:
        if f == 'edgefit2':
            mm = mtm_maps_multi([ctx['edgefit2_x'], ctx['edgefit2_y']], [ft['edgefit2_x'], ft['edgefit2_y']])
        elif f == 'edgefit':
            mm = mtm_maps(ctx['edgefit'], ft['edgefit'])
        elif f == 'mtm':
            mm = mtm_maps(ctx['int'], ft['mtm'])
        elif f == 'int':
            mm = zncc_maps(ctx['int'], ft['int'])
        else:
            mm = zncc_maps(ctx['edge'], ft['edge'])
        acc = mm if acc is None else acc + mm
    m = (acc / len(families))[0].cpu().numpy()
    iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
    fx, fy = subpixel_xy(m, iy, ix, cfg['subpix'])
    return (x0 + ix + fx + (n - 1) / 2.0, y0 + iy + fy + (n - 1) / 2.0, z, t, float(m[iy, ix]))


@torch.no_grad()
def refine(stagea, item, cand, families):
    """Coarse-to-fine local pose search around the pick (5x5 at 0.25 zoom / 0.5
    deg, then 5x5 at 0.0625 / 0.125) with parabolic sub-grid fits in x, y, zoom,
    theta. Same torch family functions as Stage A, on a small search crop."""
    cfg = stagea.cfg
    kind = item['kind']
    sp = stagea.search_prep(item['search'])
    cx, cy, z0, t0 = [float(v) for v in cand]
    zlo, zhi = cfg['zoom_range'][0] - 0.3, cfg['zoom_range'][1] + 0.3
    tlo, thi = cfg['theta_range'][0] - 0.5, cfg['theta_range'][1] + 0.5
    R = 3
    best = None
    for zs, ts in cfg['refine_levels']:
        zc_, tc_ = (z0, t0) if best is None else (best[2], best[3])
        zl = [float(np.clip(zc_ + zs * i, zlo, zhi)) for i in (-2, -1, 0, 1, 2)]
        tl = [float(np.clip(tc_ + ts * i, tlo, thi)) for i in (-2, -1, 0, 1, 2)]
        groups = stagea.templates(item['small'], kind, [(z, t) for z in zl for t in tl])
        surf = np.full((5, 5), -2.0, np.float32)
        locs = {}
        for n, (idx, T) in groups.items():
            x0 = int(np.clip(int(round(cx - (n - 1) / 2.0)) - R, 0, SEARCH_PX - n - 2 * R))
            y0 = int(np.clip(int(round(cy - (n - 1) / 2.0)) - R, 0, SEARCH_PX - n - 2 * R))
            sub = {k: v[y0:y0 + n + 2 * R, x0:x0 + n + 2 * R] for k, v in sp.items()}
            ctx = stagea.contexts(sub, families)
            ft = stagea.fam_templates(T, kind)
            acc = None
            for f in families:
                if f == 'edgefit2':
                    mm = mtm_maps_multi([ctx['edgefit2_x'], ctx['edgefit2_y']],
                                        [ft['edgefit2_x'], ft['edgefit2_y']])
                elif f == 'edgefit':
                    mm = mtm_maps(ctx['edgefit'], ft['edgefit'])
                elif f == 'mtm2':
                    mm = zncc_maps(ctx['int'], ft['mtm2'])
                elif f == 'mtm':
                    mm = mtm_maps(ctx['int'], ft['mtm'])
                elif f == 'int':
                    mm = zncc_maps(ctx['int'], ft['int'])
                else:
                    mm = zncc_maps(ctx['edge'], ft['edge'])
                acc = mm if acc is None else acc + mm
            acc = acc / len(families)
            for j, pi in enumerate(idx):
                m = acc[j].cpu().numpy()
                iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
                surf[pi // 5, pi % 5] = m[iy, ix]
                locs[pi] = (m, iy, ix, x0, y0, n)
        i, j = np.unravel_index(int(np.argmax(surf)), surf.shape)
        m, iy, ix, x0, y0, n = locs[i * 5 + j]
        fx, fy = subpixel_xy(m, iy, ix, cfg['subpix'])
        di, dj = subpixel_xy(surf, i, j, cfg['subpix'])
        di, dj = dj, di                                  # subpixel_xy returns (d along axis 1, d along axis 0)
        cx = x0 + ix + fx + (n - 1) / 2.0
        cy = y0 + iy + fy + (n - 1) / 2.0
        best = (cx, cy, zl[i] + di * zs, tl[j] + dj * ts, float(surf[i, j]))
    return best


def _parab(a, b, c):
    d = a - 2 * b + c
    return 0.0 if abs(d) < 1e-12 else float(np.clip((a - c) / (2 * d), -0.5, 0.5))


def subpixel_xy(m, iy, ix, method):
    """Sub-grid offset (dx along columns, dy along rows) of the peak of a 2-D surface.
      parab    separable 3-point parabola
      gauss    separable 3-point Gaussian (parabola on log values; ZNCC shifted to >0)
      centroid weighted centroid of the 3x3 neighbourhood above its minimum
      quad2d   least-squares 2-D quadratic over the 3x3 neighbourhood
      none     integer peak"""
    H, W = m.shape
    if method == 'none' or not (0 < ix < W - 1 and 0 < iy < H - 1):
        fx = _parab(m[iy, ix - 1], m[iy, ix], m[iy, ix + 1]) if method != 'none' and 0 < ix < W - 1 else 0.0
        fy = _parab(m[iy - 1, ix], m[iy, ix], m[iy + 1, ix]) if method != 'none' and 0 < iy < H - 1 else 0.0
        return fx, fy
    n = m[iy - 1:iy + 2, ix - 1:ix + 2].astype(np.float64)
    if method == 'parab':
        return _parab(n[1, 0], n[1, 1], n[1, 2]), _parab(n[0, 1], n[1, 1], n[2, 1])
    if method == 'gauss':
        L = np.log(np.clip(n + 1.001, 1e-6, None))
        return _parab(L[1, 0], L[1, 1], L[1, 2]), _parab(L[0, 1], L[1, 1], L[2, 1])
    if method == 'centroid':
        w = n - n.min()
        s = w.sum()
        if s <= 1e-12:
            return 0.0, 0.0
        return (float(np.clip((w.sum(0) * np.array([-1, 0, 1])).sum() / s, -0.5, 0.5)),
                float(np.clip((w.sum(1) * np.array([-1, 0, 1])).sum() / s, -0.5, 0.5)))
    # quad2d: f = a + b x + c y + d x^2 + e x y + g y^2 on the 3x3 grid
    ys, xs = np.mgrid[-1:2, -1:2]
    A = np.stack([np.ones(9), xs.ravel(), ys.ravel(), xs.ravel() ** 2, (xs * ys).ravel(), ys.ravel() ** 2], 1)
    a, b, c, d, e, g = np.linalg.lstsq(A, n.ravel(), rcond=None)[0]
    H2 = np.array([[2 * d, e], [e, 2 * g]])
    if np.linalg.det(H2) <= 1e-12 or d >= 0 or g >= 0:
        return _parab(n[1, 0], n[1, 1], n[1, 2]), _parab(n[0, 1], n[1, 1], n[2, 1])
    off = -np.linalg.solve(H2, np.array([b, c]))
    return float(np.clip(off[0], -0.5, 0.5)), float(np.clip(off[1], -0.5, 0.5))


@torch.no_grad()
def predict(model, stagea, item, prepared=None, do_refine=True, platt=(1.0, 0.0), threshold=0.5):
    """One pair -> dict(x, y, theta, scale, found, score). `score` is the
    calibrated probability that the reference is present; `found` = score >= threshold."""
    cfg, dev = stagea.cfg, stagea.dev
    if prepared is None:
        prepared = stagea(item['search'], item['small'], item['kind'], prior=item.get('prior'), gdsg=item.get('gdsg'))
    b = {k: prepared[k][None].to(dev) for k in ('patch', 'scalars', 'scores', 'geom', 'valid',
                                                  'selfcorr', 'glob')}
    was = model.training
    model.eval()
    out = model(b)
    model.train(was)
    j = int(out['logit'][0].argmax())
    cand = prepared['cand'][j].copy()
    if 'offset' in out and not prepared.get('gmode'):
        zs = cfg['zooms'][1] - cfg['zooms'][0] if len(cfg['zooms']) > 1 else 1.0
        ts = cfg['thetas'][1] - cfg['thetas'][0] if len(cfg['thetas']) > 1 else 1.0
        o = out['offset'][0, j].float().cpu().numpy().clip(-1.5, 1.5)
        cand = cand + np.array([o[0] * TOL_PX, o[1] * TOL_PX, o[2] * zs, o[3] * ts], np.float32)
    logit = float(out['presence'][0])
    score = calibrate(logit, platt)
    res = dict(x=float(cand[0]), y=float(cand[1]), scale=float(cand[2]), theta=float(cand[3]),
               logit=logit, score=score, found=int(score >= threshold), pick=j)
    if do_refine:
        if 'greys' in prepared:
            stagea.greys = (prepared['greys'].to(dev) if prepared['greys'] is not None else None)
        if prepared.get('gmode'):
            r = refine_translate(stagea, item, cand, cfg['refine_families'])
        else:
            r = refine(stagea, item, cand, cfg['refine_families'])
        res.update(x=r[0], y=r[1], scale=r[2], theta=r[3], refine_score=r[4])
    res['x'], res['y'] = gt_convention(cfg, res['x'], res['y'], item)
    return res


def gt_convention(cfg, x, y, item=None):
    """Visible position -> the position the generator's LABEL calls it.

    Measured on 12 pairs from the i4c generator: median error 1.31 px -> 0.21 px and the
    <=1 px rate 0.17 -> 0.92, every tier collapsing to 0.92 (11 of 12 sub-pixel, the
    twelfth a genuine miss). Both terms are conventions, not matcher error."""
    hp = cfg.get('gt_halfpix', 0.0)
    sh = cfg.get('gt_shear_px', 0.0)
    if item is not None and item.get('shear_px') is not None:
        sh = float(item['shear_px'])
    if hp == 0.0 and sh == 0.0:
        return x, y
    return x + hp + sh * (y / float(SEARCH_PX - 1)), y + hp


def rubric(records):
    """records: dicts with present, x, y, theta, scale (truth), px, py, pth, pz, score, gray."""
    pres = [r for r in records if r['present'] > 0.5]
    locs, scs, ths, errs = [], [], [], []
    for r in pres:
        e = math.hypot(r['px'] - r['x'], r['py'] - r['y'])
        lc = loc_credit(e)
        errs.append(e)
        locs.append(lc)
        scs.append(scale_credit(r['pz'], r['scale']) if lc > 0 else 0.0)
        ths.append(theta_credit(r['pth'], r['theta']) if lc > 0 else 0.0)
    gray = [r for r in records if r.get('gray', True)] or records
    s = np.array([r['score'] for r in gray])
    y = np.array([r['present'] > 0.5 for r in gray])
    f1, thr = best_f1(s, y)
    auc = roc_auc(s, y)
    nz = lambda v: 0.0 if (v is None or v != v) else v
    out = dict(n=len(records), loc_credit=float(np.mean(locs)) if locs else float('nan'),
               le_1px=float(np.mean([e <= 1 for e in errs])) if errs else float('nan'),
               median_err=float(np.median(errs)) if errs else float('nan'),
               scale_credit=float(np.mean(scs)) if scs else float('nan'),
               theta_credit=float(np.mean(ths)) if ths else float('nan'),
               rejection_f1=f1, threshold=thr, roc_auc=auc)
    out['pose_credit'] = float(np.nanmean([out['scale_credit'], out['theta_credit']])) if pres else float('nan')
    out['points'] = round(40 * nz(out['loc_credit']) + 20 * nz(out['pose_credit']) + 15 * f1 + 10 * nz(auc), 3)
    # presence broken down: the pooled AUC hides which absent/present kinds fail
    absent_kinds = {}
    for r in gray:
        if r['present'] < 0.5:
            absent_kinds.setdefault(r.get('kind', 'absent'), []).append(r['score'] < thr)
    hard = [r['score'] >= thr for r in gray if r['present'] > 0.5 and r.get('severity', 0) >= 3]
    out['presence'] = dict(
        tpr=float(np.mean([r['score'] >= thr for r in gray if r['present'] > 0.5])) if pres else float('nan'),
        tnr=float(np.mean([r['score'] < thr for r in gray if r['present'] < 0.5]))
        if any(r['present'] < 0.5 for r in gray) else float('nan'),
        tpr_hard=float(np.mean(hard)) if hard else float('nan'),
        tnr_by_kind={k: round(float(np.mean(v)), 3) for k, v in absent_kinds.items()})
    return out


# ==========================================================================
# Streaming producers
# ==========================================================================


# ==========================================================================
# Checkpointing
# ==========================================================================


def load_bundle(path, device='cpu'):
    """model_best.pt / model_final.pt -> (model, stagea, platt, threshold) for inference."""
    bd = torch.load(path, map_location=device, weights_only=False)
    # A checkpoint carries the cfg it was TRAINED with, which may predate keys added
    # since. Backfill from DEFAULTS so an older bundle still loads; the bundle's own
    # values always win, and every key added after these were trained defaults to the
    # behaviour they had (no CAD prior, no global pose, no pruning, no label offsets).
    cfg = dict(DEFAULTS)
    cfg.update(bd['cfg'])
    model = CrossEncoder(cfg).to(device)
    model.load_state_dict(bd['state_dict'])
    model.eval()
    edge_net = None
    if bd.get('edge_state') is not None:
        edge_net = EdgeNet(cfg['edge_ch']).to(device)
        edge_net.load_state_dict(bd['edge_state'])
        edge_net.eval()
    return model, StageA(cfg, torch.device(device), edge_net), tuple(bd.get('platt', (1.0, 0.0))), \
        float(bd.get('threshold', 0.5))
