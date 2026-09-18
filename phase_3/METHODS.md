# Phase 3 — method

Register a GDSII design (8 layers, **no brightness anywhere in the file**) against a
noisy SEM capture, and say whether the site is in the image at all.

The spec's instruction is followed literally: **registration runs on edges, brightness
runs the score.** The per-layer grey fit is computed, but it is evidence for the
calibration column, not the thing that drives the match.

---

## 1. Pose, from the whole field

`search.gds` ships in both splits and covers the entire search image, not just the
reference site. That makes the pose a **whole-field least-squares problem** rather than a
patch-matching one: render the search CAD at a candidate (zoom, rotation), solve the 8
per-layer greys plus a background offset by least squares over all 1000×1000 pixels, and
score the fit by R².

- Coarse 10×12 grid at 128×128, then coordinate descent with parabolic steps at 200² and
  400². About 25 evaluations replace a 242-point dense grid.
- The greys are solved through the 9×9 normal equations (`G = XᵀX`, `b = Xᵀy`,
  `sse = yᵀy − b·g`), not `lstsq` on 160,000 rows — same answer, 1.1 s → negligible.
- Measured on 48 present pairs: **scale credit 1.000**, median scale error 0.02%.

### The canvas size is not in the GDS — and the bounding box is not a substitute

`canvas_to_search_affine(cs, z, θ)` needs the canvas size, because it maps the canvas
*centre* to the image centre. The GDS does not record it: the mats' design polygons
**overflow** the rendered canvas (measured bounding boxes of 11,360–14,308 nm against
true canvases of 9,088–11,769 nm) and are not centred in it either.

Taking `cs` from the bounding box is not merely imprecise, it is catastrophic: the error
reached 2,278 nm, which is **132 px of translation in both axes**. The field fit then has
~75% overlap, R² collapses to 0.01, and the coarse grid picks a wrong zoom. Measured on
the 50-pair set: **11 of 48 pairs missed by hundreds of pixels, every one of them with a
scale error of 20–48%** (median 25.85%, against 0.072% on the pairs that worked). Total
68.39 of 85.

The fix is to stop guessing and derive it: the search image is 1000 px at `zoom` nm/px, so
`cs = canvas_size_for(z, θ)` is a function of the candidate pose, evaluated at every grid
point. It reproduces our generator's canvas exactly and i4c's to within 8 nm (0.4 px).
Same 50 pairs afterwards: **48 of 48 within 1 px, 84.12 of 85.**

A residual translation estimator (`_shift_estimate`, phase correlation of the fitted
render against the SEM) stays as the safety net for whatever the blind set's framing
turns out to be, gated by a 3 px dead zone and an R² acceptance test.

## 2. Location, from geometry

Both CADs are in the same frame, scale and orientation, so finding the reference design
inside the search design is a **noise-free translation search** — no SEM involved. Per-layer
one-hot correlation (layer identity is the signal; binary occupancy alone peaks at 1.000
against ~10⁶ rivals), rasterised at 2 nm and area-pooled to 8 nm. 4 nm aliased thin fins
and picked the wrong peak on 2 of 16 sites.

Multi-peak with NMS: the strongest peak is trusted, and a near-equal runner-up marks a
genuine design repeat that lowers confidence.

## 3. The local cue — per-layer signed edge fitting

A real SEM's brightness step at a boundary depends on **which two layers meet there**, so a
flat "all boundaries equal" CAD edge map is the wrong template. This was the single
biggest gain in the project. Measured on 50 pairs, one change at a time:

| variant | pts | why |
|---|---|---|
| flat boundary map | 49.0 | the step depends on which two layers meet |
| per-layer **magnitude** fit | 70.7 | +21.7: each layer gets its own edge strength, solved by least squares |
| per-layer **signed dx/dy** fit | 77.5 | +6.8: magnitude discards edge direction |
| + search-CAD geometry prior | 80.1 | +2.6: F1 0.886 → 0.986, AUC 0.907 → 0.997 |
| + tone map as a second opinion | 81.2 | +1.0: localization 0.928 → 0.950 |

No prior art was found for per-layer gradient basis fitting, and no measured comparison of
signed dx/dy against magnitude.

## 4. Tone mapping — for the score, not the match

At each position the SEM window `S` is fitted as `c + Σ_l g_l · M_l` over the 8 CAD layer
masks. **No grey is ever assumed**: the 8 levels and the offset are unknowns solved from
the image, so an arbitrary per-layer brightness, a gamma or a contrast change is fitted
exactly. Score is √R².

The mask Gram matrix does not depend on position, so the masks are whitened once per pose
and explained variance is `Σ_j (w_j ⋆ S)²` — L FFT correlations, exact. With one layer it
equals |ZNCC| (checked to 9e-8). With deliberately wrong greys, plain ZNCC peaked at 0.37
and tone mapping at 0.95, both at the true location.

**The fitted greys are the yield raster.** Their R², their agreement with layer order and
the residual level are what feed the presence head and the score column — which is exactly
what the spec says calibration is asking for.

## 5. Presence

Pose-surface statistics (peak, margin, entropy, std, PSR) per cue, plus the CAD geometry
evidence, into a presence head; Platt-calibrated with an F1-optimal threshold saved in the
weights bundle.

Before training the peak value alone is at chance on Phase 3 (AUC 0.41–0.53) while **PSR
reaches AUC 0.925**, so an untrained head would be worthless. The shipped fallback
therefore uses the geometry score directly — `top_sigma · (1 − runner_up)` — which
measured **F1 1.000 / AUC 1.000** on the 24-pair absent-heavy set and 0.990 / 0.969 on the
50-pair set.

## 6. Sub-pixel and refinement

2-D quadratic / parabolic peak interpolation, then three refinement levels. In global-pose
mode the refinement is **translation-only at the fixed global pose** — re-estimating the
pose locally degraded it from 0.01% to 0.19% scale error, because a ~100 px window
constrains scale and rotation far less than the whole field.

## 7. Ground-truth convention

The generator's label is **not** where the feature is visible, by two measured amounts.
Both are corrected, and the training labels are emitted the same way so that training,
validation and inference share one target. Worth ~8 of the 40 localization points.
See `GROUND-TRUTH-CONVENTION.md`.

---

## What was tried and rejected, on measurement

| rejected | cost |
|---|---|
| two-stage fixed-grey render | −1.3 pts (presence AUC 0.925 → 0.859) |
| boundary-band basis | −0.2 pts |
| 15-pose fast grid | −2.2 pts |
| quarter-res pose pruning on the coarse **global** max, no prior | −9.5 pts |
| prior-window pose pruning (the "fix" for the above) | −0.3 vs the plain global max — the hypothesis was wrong |
| band-pass (DoG) prefilter | pose 0.870 vs 0.935 plain |
| Anscombe VST, median 3×3 | no gain at the chosen σ |
| local pose re-estimation in global mode | scale error 0.01% → 0.19% |

## Efficiency

2.9–3.4 s per pair end to end on 4 CPU threads, against a 5 s median budget and a 20 s
hard timeout. Three changes got there from ~9 s: one shared 2 nm CAD raster pooled to
every level (was rebuilding an 11,000 px canvas four times — 4.6 s of 6.3 s), normal
equations instead of `lstsq`, and pose pruning for the 25-pose variant.
