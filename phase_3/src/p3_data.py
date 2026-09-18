"""Phase 3 inference — CAD (GDS, 8 layers, no brightness) against an SEM search image.

`load_pair()` turns one row of `pairs.csv` into everything the matcher needs:

  * the reference GDS as 8 per-layer masks;
  * the search GDS as one per-layer occupancy raster, built once at 2 nm and area-pooled
    to whatever resolution each stage wants;
  * the global pose, fitted by least squares over the whole 1000x1000 field;
  * the reference's origin inside the search design, found by noise-free CAD-to-CAD
    correlation, with runner-up peaks as the presence evidence.

The canvas size is deliberately NOT taken from the GDS bounding box -- see global_pose().
"""
from __future__ import annotations

import json
import math

import numpy as np
import cv2

import dsr_core as core

KIND = 'cad'
N_LAYERS = 8

# The generator paints layers bottom-to-top at 1 nm/px over a background of 31
# (`yield_model.background_intensity()`), which never collides with the layer labels
# 1..N. This is that one function, inlined, so the whole CAD generator does not have to
# ship just to read a reference file.
BACKGROUND_LABEL = 31


def rasterize_label(cell, size_px, num_layers=N_LAYERS):
    """A gdstk cell -> a label raster: layer index + 1, painter's order, 1 nm/px."""
    canvas = np.full((size_px, size_px), BACKGROUND_LABEL, np.uint8)
    for layer in range(num_layers):
        polys = cell.get_polygons(layer=layer, datatype=0)
        if not polys:
            continue
        cv2.fillPoly(canvas, [np.round(p.points).astype(np.int32) for p in polys],
                     color=int(layer + 1))
    return canvas


def canvas_size_for(zoom, theta):
    t = np.deg2rad(abs(theta))
    return int(np.ceil(core.SEARCH_PX * zoom * (np.cos(t) + np.sin(t)))) + 8

def canvas_size_i4c(zoom, theta):
    """The organisers' i4c convention: the canvas is always 1000*zoom nm and the
    rotation happens INSIDE it (a padded canvas is rotated and the original footprint is
    cropped back out). Ours, canvas_size_for(), grows with rotation. Measured on i4c data
    with rotation on: our convention scored pose 12.67/20, this one 20.00/20; on our data
    the reverse (20.00 vs 17.25). global_pose() tries both and keeps the better fit."""
    return int(np.ceil(core.SEARCH_PX * zoom)) + 8


def canvas_to_search_affine(cs, zoom, theta):
    t = np.deg2rad(theta)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, s], [-s, c]], np.float64) / zoom
    cc, sc = (cs - 1) / 2.0, (core.SEARCH_PX - 1) / 2.0
    return np.hstack([R, (np.array([sc, sc]) - R @ np.array([cc, cc])).reshape(2, 1)])
CAD_DS = 8                      # nm per pixel for the CAD-vs-CAD search

def cad_onehot(lab, ds=CAD_DS, L=N_LAYERS):
    """label raster -> (L, h/ds, w/ds) per-layer soft occupancy."""
    h = (lab.shape[0] // ds) * ds
    return np.stack([cv2.resize((lab[:h, :h] == i + 1).astype(np.float32), (h // ds, h // ds),
                                interpolation=cv2.INTER_AREA) for i in range(L)])
STACK_DS = 8        # nm/px of the shared per-layer occupancy stack

def cad_stack_from_polys(poly_iter, canvas_px, ds=STACK_DS, L=N_LAYERS):
    """Per-layer soft occupancy of the search CAD at `ds` nm/px, built ONCE.

    Rasterises at ds/4 nm/px (not 1 nm: nothing downstream uses finer than ds) and
    area-pools for fractional coverage. Measured: the old path rasterised a
    12,000 px canvas at 1 nm and rebuilt per-layer masks four times (~6 s per pair)."""
    # Rasterise at 2 nm and area-pool 4x. Measured: 4 nm rasterisation aliased thin
    # features (~14 nm fins) enough that 2 of 16 sites picked a wrong CAD peak.
    fine = max(1, ds // 4)
    size = int(math.ceil(canvas_px / fine))
    lab = np.zeros((size, size), np.uint8)
    by_layer, keep = {}, []
    for layer, pts in poly_iter:
        pts = np.asarray(pts, np.float64)
        keep.append((int(layer), pts))
        by_layer.setdefault(layer, []).append(np.round(pts / fine).astype(np.int32))
    for layer in sorted(by_layer):                    # painter's order: higher layers on top
        if layer < L:
            cv2.fillPoly(lab, by_layer[layer], color=int(layer + 1))
    k = max(1, ds // fine)
    h = (size // k) * k
    stack = np.stack([cv2.resize((lab[:h, :h] == i + 1).astype(np.float32), (h // k, h // k),
                                 interpolation=cv2.INTER_AREA) for i in range(L)])
    bbox = (np.array([[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()] for _, q in keep])
            if keep else np.zeros((0, 4)))
    return dict(stack=stack, ds=fine * k, canvas_px=float(canvas_px), polys=keep, bbox=bbox)

def cad_stack_from_gds(path):
    import gdstk
    cell = gdstk.read_gds(path).top_level()[0]
    (x0, y0), (x1, y1) = cell.bounding_box()
    return cad_stack_from_polys(((p.layer, p.points) for p in cell.get_polygons()),
                                int(math.ceil(max(x1, y1))))

def pool_stack(cad, ds):
    """Coarser level of the shared stack (ds must be a multiple of cad['ds'])."""
    k = max(1, int(round(ds / cad['ds'])))
    st = cad['stack']
    if k == 1:
        return st
    h = (st.shape[1] // k) * k
    return np.stack([cv2.resize(x[:h, :h], (h // k, h // k), interpolation=cv2.INTER_AREA) for x in st])

def onehot_ds(lab, ds, L=N_LAYERS):
    h = (lab.shape[0] // ds) * ds
    return np.stack([cv2.resize((lab[:h, :h] == i + 1).astype(np.float32), (h // ds, h // ds),
                                interpolation=cv2.INTER_AREA) for i in range(L)])

def _field_fit(stack, ds, sem_r, cs, z, th, return_greys=False, shift=(0.0, 0.0), return_render=False):
    """Per-layer grey fit over the whole field. Solved through the (L+1)x(L+1)
    normal equations instead of an SVD on ~160k rows (measured 1.1 s -> see bench)."""
    res = sem_r.shape[0]
    M = canvas_to_search_affine(cs, z, th).copy()
    M[:, 2] += np.asarray(shift, np.float64)
    M[:, :2] *= ds
    M *= res / float(core.SEARCH_PX)
    X = np.stack([cv2.warpAffine(s_, M, (res, res), flags=cv2.INTER_LINEAR) for s_ in stack]
                 + [np.ones((res, res), np.float32)], -1).reshape(-1, stack.shape[0] + 1)
    y = sem_r.reshape(-1)
    G = X.T.astype(np.float64) @ X
    b = X.T.astype(np.float64) @ y
    g = np.linalg.solve(G + 1e-6 * np.eye(G.shape[0]), b)
    yy = float(y.astype(np.float64) @ y)
    n, ym = y.size, float(y.mean())
    sse = yy - float(b @ g)                       # ||y - Xg||^2 at the LS solution
    sst = yy - n * ym * ym
    r2 = 1.0 - sse / max(sst, 1e-9)
    if return_render:
        return r2, g.astype(np.float32), (X @ g.astype(np.float32)).reshape(res, res)
    return (r2, g.astype(np.float32)) if return_greys else r2
COARSE_STARTS = 5       # descents run from the best N coarse cells; see coarse()
SHIFT_DEADZONE = 3.0    # search px; below this the canvas size is right and the

def _shift_estimate(stack, ds, sem_r, cs, z, t, shift, r2=None):
    """Translation between the rendered search CAD and the SEM (search px), by phase
    correlation of the full-field render with fitted greys. Needed because the canvas
    size is NOT in the GDS: search.gds holds mat polygons only, so its bounding box can
    stop a strip short of the canvas edge and the centre-to-centre mapping is offset.

    Accepted only if it raises the full-field R^2 (a periodic design can hand phase
    correlation a wrong peak: measured once in 10, off by 11 px) and is larger than the
    dead zone (measured: a correct canvas leaves ~0.5-1.3 px of x drift, and chasing
    that costs 0.34 px of localization because the label does not include it)."""
    res = sem_r.shape[0]
    if r2 is None:
        r2 = _field_fit(stack, ds, sem_r, cs, z, t, shift=shift)
    _, _, rend = _field_fit(stack, ds, sem_r, cs, z, t, shift=shift, return_render=True)
    win = cv2.createHanningWindow((res, res), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(rend.astype(np.float32), sem_r.astype(np.float32), win)
    k = float(core.SEARCH_PX) / res
    cand = (shift[0] + dx * k, shift[1] + dy * k)
    if math.hypot(*cand) < SHIFT_DEADZONE:
        cand = (0.0, 0.0)
    if cand == shift:
        return shift, r2
    r2c = _field_fit(stack, ds, sem_r, cs, z, t, shift=cand)
    return (cand, r2c) if r2c > r2 + 1e-4 else (shift, r2)

def global_pose(search_lab, search_img, cs=None, cad=None, estimate_shift=True):
    """(zoom, theta, R^2, per-layer greys, shift, canvas_px). Coarse grid at low
    resolution, a translation estimate, then coordinate descent with parabolic steps at
    two finer resolutions (the full-field R^2 surface is smooth: ~25 evaluations replace
    a 242-point dense grid).

    `cs=None` DERIVES the canvas size from each candidate pose via canvas_size_for(),
    which is what inference must do: the canvas size is not in the GDS, and taking it
    from the polygon bounding box is not merely imprecise but wrong by up to 2278 nm
    (measured), i.e. 132 px of translation in BOTH axes. That wrecks the fit before the
    shift estimator can run -- 11 of 48 pairs picked a zoom 20-48% off, every one of them
    a catastrophic miss. Deriving it instead reproduces our generator's canvas exactly
    and i4c's to within 8 nm (0.4 px), inside the shift dead zone.
    """
    sem = search_img.astype(np.float32)
    level = (lambda ds: pool_stack(cad, ds)) if cad is not None else (lambda ds: onehot_ds(search_lab, ds))
    stack = level(48)
    sem_r = cv2.resize(sem, (128, 128), interpolation=cv2.INTER_AREA)
    # The grid is ALIGNED (it contains 10.0 and 0.0) and finer than the R^2 peak.
    # Measured: the peak half-width in zoom is +-0.05 on the i4c generator and +-0.14 to
    # +-0.22 on ours, against a 0.5-step grid whose worst-case distance to a sample is
    # 0.25. The old grid -- np.arange(7.75, 12.26, 0.5) and np.arange(-5.5, 5.51, 1.0) --
    # was offset by half a step and so could never sample zoom 10.0 or theta 0.0, which is
    # EXACTLY what the i4c CAD generator emits. It landed on a flat aliasing ridge from the
    # periodic mat/strip pitch instead, scoring R^2 0.23 against 0.85 at the truth, and
    # coordinate descent could not cross back. Result: theta wrong on 12 of 12 real pairs,
    # always by the same +-0.415 deg, costing 4.0 of the 20 pose points.
    Z, T = np.arange(8.0, 12.01, 0.25), np.arange(-5.0, 5.01, 0.5)
    shift = (0.0, 0.0)

    def coarse(sh, csf, Zg=None, Tg=None, starts=None):
        """Grid, then a short descent from each of the best COARSE_STARTS cells.

        One start is not enough: a periodic layout admits several (zoom, rotation) pairs
        that correlate similarly, and the broad false ridge outscores the narrow true peak
        at grid resolution. Restarting from the top few cells and keeping the best final
        R^2 fixed theta on 23 of 23 of our pairs (was 17) and 12 of 12 i4c pairs (was 0)."""
        f0 = lambda z_, t_: _field_fit(stack, 48, sem_r, csf(z_, t_), z_, t_, shift=sh)  # noqa: E731
        Zg = Z if Zg is None else Zg
        Tg = T if Tg is None else Tg
        surf = np.array([[f0(z_, t_) for t_ in Tg] for z_ in Zg])
        best = None
        for flat in np.argsort(surf.ravel())[::-1][:starts or COARSE_STARTS]:
            i, j = np.unravel_index(int(flat), surf.shape)
            z_, t_, cur = float(Zg[i]), float(Tg[j]), float(surf[i, j])
            zs_, ts_ = 0.25, 0.5
            for _ in range(4):
                for axis in (0, 1):
                    st = zs_ if axis == 0 else ts_
                    a = f0(z_ - st, t_) if axis == 0 else f0(z_, t_ - st)
                    b = f0(z_ + st, t_) if axis == 0 else f0(z_, t_ + st)
                    off = core._parab(a, cur, b) * st if max(a, b) <= cur else (st if b > a else -st)
                    if axis == 0:
                        z_ += off
                    else:
                        t_ += off
                    cur = f0(z_, t_)
                zs_ *= 0.5
                ts_ *= 0.5
            if best is None or cur > best[2]:
                best = (z_, t_, cur)
        return best

    # The canvas size is not in the GDS, and the two generators we know disagree on how it
    # relates to the pose (see canvas_size_i4c). select_convention() picks one from a cheap
    # search; the full search below then runs under that convention only.
    def select_convention():
        """Which canvas convention this data uses, from a CHEAP search: the aligned
        0.5-zoom / 1-degree grid with 3 descents, no shift estimate. The wrong convention
        adds a pose-dependent centre offset that sharply lowers the best fit, so a coarse
        look is enough to tell them apart; the full search then runs once. (Running the
        full search under both was right 36/36 but cost 2.4-4.0 s, over the 5 s budget.)"""
        Zc, Tc = np.arange(8.0, 12.01, 0.5), np.arange(-5.0, 5.01, 1.0)
        return max(((coarse(shift, f, Zc, Tc, 3)[2], i, f)
                    for i, f in enumerate((canvas_size_for, canvas_size_i4c))))[2]

    choice = None
    for csf in ([lambda z_, t_: float(cs)] if cs else [select_convention()]):
        z, t, r2c = coarse(shift, csf)
        sh_ = shift
        # the grid above assumed the CAD and the SEM share a centre. On the organisers'
        # own files they do not (their search.gds bounding box stopped ~68 search px short
        # of the canvas), and a wrong offset picks a wrong zoom, so the grid is re-run once
        # the offset is known. 120 fits at 128x128 = ~0.1 s.
        if estimate_shift:
            sh2, _ = _shift_estimate(stack, 48, sem_r, csf(z, t), z, t, shift, r2=r2c)
            if sh2 != shift:
                z2, t2, r2b = coarse(sh2, csf)
                if r2b > r2c:
                    z, t, sh_, r2c = z2, t2, sh2, r2b
        if choice is None or r2c > choice[3]:
            choice = (csf, z, t, r2c, sh_)
    csf, z, t, _, shift = choice
    fine_ds = 16 if cad is not None else 12
    for res, ds, zs, ts in ((200, 24, 0.25, 0.5), (400, fine_ds, 0.06, 0.12)):
        stack = level(ds)
        sem_r = cv2.resize(sem, (res, res), interpolation=cv2.INTER_AREA)
        f = lambda zz, tt: _field_fit(stack, ds, sem_r, csf(zz, tt), zz, tt, shift=shift)
        best = f(z, t)
        if estimate_shift:
            shift, best = _shift_estimate(stack, ds, sem_r, csf(z, t), z, t, shift, r2=best)
        for _ in range(3):
            for axis in (0, 1):
                st = zs if axis == 0 else ts
                a = f(z - st, t) if axis == 0 else f(z, t - st)
                b = f(z + st, t) if axis == 0 else f(z, t + st)
                off = core._parab(a, best, b) * st if max(a, b) <= best else (st if b > a else -st)
                if axis == 0:
                    z += off
                else:
                    t += off
                best = f(z, t)
            zs *= 0.5
            ts *= 0.5
    if estimate_shift:
        shift, _ = _shift_estimate(stack, fine_ds, sem_r, csf(z, t), z, t, shift)
    r2, g = _field_fit(stack, fine_ds, sem_r, csf(z, t), z, t, return_greys=True, shift=shift)
    return (float(z), float(t), float(r2), g[:N_LAYERS].astype(np.float32),
            (float(shift[0]), float(shift[1])), float(csf(z, t)))

def cad_peaks(search_lab, ref_lab, ds=CAD_DS, n=5, cad=None):
    """Top-n CAD<->CAD translation peaks with NMS (multi-peak). Geometry is noise-free,
    so the strongest peak is trusted; a runner-up within 8% is a genuine design repeat
    that only the SEM can (sometimes) break, and it lowers the confidence."""
    if cad is not None:
        ds = cad['ds']
        S = cad['stack']
    else:
        S = cad_onehot(search_lab, ds)
    R = cad_onehot(ref_lab, ds)
    num = None
    for l in range(S.shape[0]):
        t = R[l] - R[l].mean()
        if t.std() < 1e-6 or t.shape[0] >= S[l].shape[0]:
            continue
        r = cv2.matchTemplate(S[l], t.astype(np.float32), cv2.TM_CCORR)
        num = r if num is None else num + r
    if num is None:
        return [(0.0, 0.0, 0.0)]
    sd = max(float(num.std()), 1e-9)
    m, out, rx = num.copy(), [], max(R.shape[1] // 2, 6)
    for _ in range(n):
        _, pk, _, loc = cv2.minMaxLoc(m)
        out.append((float(loc[0] * ds), float(loc[1] * ds), float(pk / sd)))
        m[max(loc[1] - rx, 0):loc[1] + rx, max(loc[0] - rx, 0):loc[0] + rx] = -1e30
    return out

FINE_PEAKS = 3          # CAD peaks re-checked at 1 nm
FINE_SEARCH_NM = 8      # the 8 nm peak is within +-4 nm of the true origin
FINE_LOCK = 0.99        # measured: the true copy matches the reference at 1.0000 ...
FINE_MARGIN = 0.01      # ... and the nearest near-copy at 0.8751-0.9729
FINE_BACKGROUND = 31    # same background label the reference raster uses


def _window_labels(cad, x0, y0, size):
    """The search design around (x0, y0) as a 1 nm label raster -- layer + 1, painter's
    order, from only the polygons that touch the window."""
    img = np.full((size, size), FINE_BACKGROUND, np.uint8)
    bb = cad['bbox']
    if len(bb) == 0:
        return img
    hit = np.nonzero((bb[:, 2] >= x0) & (bb[:, 0] <= x0 + size) &
                     (bb[:, 3] >= y0) & (bb[:, 1] <= y0 + size))[0]
    by_layer = {}
    for i in hit:
        layer, pts = cad['polys'][i]
        by_layer.setdefault(layer, []).append(np.round(pts - (x0, y0)).astype(np.int32))
    for layer in sorted(by_layer):
        if layer < N_LAYERS:
            cv2.fillPoly(img, by_layer[layer], int(layer + 1))
    return img


def fine_match(cad, ref_lab, x0, y0, r=FINE_SEARCH_NM):
    """(agreement, x0, y0): the fraction of the reference's 1 nm label raster that the
    search design reproduces exactly at this origin, after a +-r nm alignment search.

    Why it exists: at the 8 nm resolution of the CAD search, similar-but-different layouts
    look like near-ties (0.94-0.99), and the SEM image cannot separate them either. At 1 nm
    the true copy matches EXACTLY (1.0000 on every pair checked) while the near-copies
    reach only 0.8751-0.9729 -- so the design file alone can decide which copy it is."""
    n = ref_lab.shape[0]
    x0, y0 = int(round(x0)), int(round(y0))
    big = _window_labels(cad, x0 - r, y0 - r, n + 2 * r)
    sub_ = ref_lab[::4, ::4]
    best = (-1.0, 0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            a = float((big[r + dy:r + dy + n:4, r + dx:r + dx + n:4] == sub_).mean())
            if a > best[0]:
                best = (a, dx, dy)
    full = (-1.0, 0, 0)
    for dy in range(max(-r, best[2] - 2), min(r, best[2] + 2) + 1):
        for dx in range(max(-r, best[1] - 2), min(r, best[1] + 2) + 1):
            a = float((big[r + dy:r + dy + n, r + dx:r + dx + n] == ref_lab).mean())
            if a > full[0]:
                full = (a, dx, dy)
    return full[0], x0 + full[1], y0 + full[2]


def gds_geometry(search_lab, ref_lab, search_img, cs=None, cad=None):
    """Everything the GDS route gives, in one dict (training workers and phase3.py
    call this same function): global pose, fitted yield raster, multi-peak origins."""
    z, t, r2, greys, shift, cs_fit = global_pose(search_lab, search_img, cs, cad=cad)
    peaks = cad_peaks(search_lab, ref_lab, cad=cad)
    top = peaks[0][2]
    ratio = peaks[1][2] / max(top, 1e-9) if len(peaks) > 1 else 0.0
    ties = int(sum(p[2] >= 0.92 * top for p in peaks))
    fine, lock = [], None
    if cad is not None and cad.get('polys'):
        for (x0, y0, sg) in peaks[:FINE_PEAKS]:
            a, fx, fy = fine_match(cad, ref_lab, x0, y0)
            fine.append((a, fx, fy, sg))
        fine.sort(key=lambda f: -f[0])
        # the copy that reproduces the design exactly goes first, snapped to 1 nm
        peaks = [(float(fx), float(fy), sg) for (_, fx, fy, sg) in fine] + list(peaks[FINE_PEAKS:])
        second = fine[1][0] if len(fine) > 1 else 0.0
        if fine[0][0] >= FINE_LOCK and fine[0][0] - second >= FINE_MARGIN:
            lock = 0
    return dict(zoom=z, theta=t, field_r2=r2, greys=greys, peaks=peaks, top_sigma=top, shift=shift,
                runner_up=ratio, ties=ties, canvas_px=float(cs_fit),
                fine=[f[0] for f in fine], fine_best=(fine[0][0] if fine else None),
                fine_second=(fine[1][0] if len(fine) > 1 else None), lock=lock)

def cad_prior_stack(cad, ref_lab):
    """cad_prior() on the shared stack: (x0_nm, y0_nm, peak_sigma, rivals)."""
    pk = cad_peaks(None, ref_lab, cad=cad, n=64)
    top = pk[0][2]
    rivals = int(sum(p[2] >= 0.92 * top for p in pk[1:]))
    return pk[0][0], pk[0][1], top, rivals

def gds_label(path, size_px=None, num_layers=N_LAYERS):
    """A .gds file -> label raster (layer number + 1), plus the size it spans in nm."""
    import gdstk
    cell = gdstk.read_gds(path).top_level()[0]
    polys = cell.get_polygons()
    if size_px is None:
        (x0, y0), (x1, y1) = cell.bounding_box()
        size_px = int(math.ceil(max(x1, y1)))
    lab = rasterize_label(cell, size_px, num_layers)
    return lab, size_px, (max([p.layer for p in polys] + [num_layers - 1]) + 1)

def load_pair(reference_gds_path, search_path, search_gds_path=None, global_pose_on=True,
              params_json_path=None):
    """Inference-side loader for phase3.py.

    reference GDS -> per-layer soft masks; search PNG -> gray. When the search GDS is
    given (Phase 3 ships it in BOTH splits) the geometry prior is computed here with the
    SAME function the training stream uses: the reference's canvas-frame origin, the
    peak's significance and the rival count (the presence signal)."""
    srch = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
    if srch.ndim == 3:
        srch = cv2.cvtColor(srch, cv2.COLOR_BGR2GRAY)
    ref_lab, _, nl = gds_label(reference_gds_path, core.REF_PX)
    masks = np.stack([(ref_lab == i + 1).astype(np.float32) for i in range(N_LAYERS)])
    item = dict(search=np.ascontiguousarray(srch), small=core.ref_to_small(masks), kind=KIND,
                prior=None, meta=dict(optical=False))
    if params_json_path:
        # the label ignores scan drift, so the correction needs that pair's real shear
        # amplitude rather than the 1.5 px nominal (core.gt_convention)
        try:
            with open(params_json_path) as fh:
                item['shear_px'] = float(json.load(fh)['shear_amplitude_px'])
        except Exception:                                            # noqa: BLE001
            pass
    if search_gds_path:
        cad = cad_stack_from_gds(search_gds_path)
        x0, y0, sigma, rivals = cad_prior_stack(cad, ref_lab)
        # cad['canvas_px'] is the POLYGON BOUNDING BOX, which is not the canvas: the
        # design overflows the rendered frame (measured: by up to 2278 nm). Only used as
        # a last resort, when no global fit ran to derive the real one.
        cs, shift = cad['canvas_px'], (0.0, 0.0)
        if global_pose_on:
            item['gdsg'] = gds_geometry(None, ref_lab, item['search'], None, cad=cad)
            shift = item['gdsg']['shift']
            cs = item['gdsg']['canvas_px']
        item['prior'] = dict(x0=x0, y0=y0, sigma=sigma, rivals=rivals, canvas_px=float(cs),
                             shift=shift)
    return item
