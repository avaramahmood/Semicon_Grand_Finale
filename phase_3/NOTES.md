# Ground-truth convention — what `drift-sense-i4c` actually labels

Measured 2026-09-18 on 12 pairs written by the i4c generator itself
(`build_cad_geometry` → `render_cad_sample`, real `reference.gds` / `search.gds` /
`search.png`), scored against its own `gt_x`/`gt_y`.

**Its label is not where the feature is visible.** Two fixed offsets, both in the
generator, both now corrected in our pipeline (`dsr_core.gt_convention`).

| | median error | ≤1 px | ≤2 px | ≤3 px | ≤5 px |
|---|---|---|---|---|---|
| against `gt_x`/`gt_y` as written | 1.31 px | 0.17 | — | — | 0.92 |
| with the correction applied | **0.21 px** | **0.92** | 0.92 | 0.92 | 0.92 |

The residual was a *constant* — median dx −1.01 to −1.26 px, dy −0.45 to −0.50 px, on
four independent methods. After correction, dx −0.03 to −0.09 and dy −0.00 to −0.03. Every
tier collapses to the same 0.92: 11 of 12 pairs sub-pixel, the twelfth a genuine miss. So
this was a labelling convention, not matcher accuracy — and it was worth **~8 of the 40
localization points**, because a ~1 px systematic error moves every pair from the 1.00
credit tier to the 0.80 one.

## The two offsets

**1 — half a pixel, both axes (+0.45).** `sem_imaging.downsample_area_average` is
`cv2.resize(..., INTER_AREA)` with factor 10. Output pixel *j* covers input
`[10j, 10j+10)`, centre `10j + 4.5`, so the inverse map is `(X − 4.5)/10`. But
`cad_pipeline.render_cad_sample` writes `gt_x, gt_y = rx / SCALE_FACTOR`. Unconditional,
independent of drift or noise.

**2 — scan drift the label never sees (x only, ~+0.75 px at the middle row).**
`gt_x/gt_y` are computed from the rotated canvas *before* `sem_imaging.image_search` runs.
`image_search` then calls `apply_raster_drift`, which shifts row *y* by
`shear_amplitude_px · y/(h−1) + N(0, drift_jitter_px)` — defaults 1.5 px and 0.5 px.
Nothing feeds back into the label.

Our Phase 2 generator takes the opposite choice deliberately: `apply_scan_distortion`
applies *only* the per-row shift to the search image (rotation and barrel are attributed to
the reference) precisely so the label can be corrected exactly, and
`drift_sense_generate.py:1382` does `gt_x = gt_x + dx[row]` at the crop's centre row.

## Why a convention is needed at all

Scan drift is a **per-row** warp, not a rigid translation — the deliberate structural
choice in both generators, and what makes the problem hard. So a reference crop spanning
~100 rows **has no single true x**. Three defensible answers:

| convention | who uses it | property |
|---|---|---|
| **A** no correction — centre before imaging | **i4c CAD generator (our target)** | ~0.75 px from the visible feature at default shear |
| **B** correct at the crop's centre row | our Phase 2 generator | exact for that row; inherits its jitter draw, ±0.5 px (±2.3 px at 'severe') |
| **C** correct by the mean over the crop's rows | neither | what a window matcher actually estimates |

## What we implemented

`dsr_core` config keys `gt_halfpix` (0.45) and `gt_shear_px` (1.5 nominal), set in
`phase3/build.py` for all four methods. Set both to 0.0 to report the visible position.

- **`predict()`** adds `gt_halfpix` to x and y, and `shear · y/(SEARCH_PX−1)` to x.
- **`load_pair(..., params_json_path=...)`** overrides the nominal shear with that pair's
  real `shear_amplitude_px`. The Phase 3 spec gives every pair a `params_json_path`, so
  the correction is exact per pair rather than assuming the 1.5 default.
- **`p3_data.stream`** emits *training* labels in the same convention — drift left in,
  half-pixel added — so training, validation and inference all aim at one target instead
  of the model having to learn the offset as a bias.

Convention A puts the row's random jitter into the label, and a window matcher averages
~100 rows, so that jitter is irreducible: **under A the ≤1 px tier is not fully reachable
even by a perfect matcher.** Our own generator's jitter range (0.1–3.5 px) is far wider
than i4c's 0.5 default, so validation scores on our stream understate what the real set
would give.

## Still open — worth asking

1. **Scale.** The i4c code cannot vary it: `SCALE_FACTOR = PIXEL_SIZE_SEARCH_NM //
   PIXEL_SIZE_REF_NM = 10`, no parameter, no CLI flag. Every pair it can produce is scale
   10.0. But the Phase 3 statement promises the blind set has *"a wider pose range than
   Phase 2"*. Both cannot be true. We are hedging at zoom 8–12.
2. **Rotation.** `search_rotation_deg` defaults to **0.0** (cap 10.0) and
   `generate_cad_dataset.py` does not expose it, so the CLI can only ever emit 0°. We
   assume ±5°.
3. **`search.gds`.** Verified by repo-wide grep for `write_gds`: the CLI
   (`generate_cad_dataset.py:98`) writes `reference/<id>.gds` only, and its manifest
   columns are `id, reference_gds_path, reference_preview_path, search_path, match_found,
   gt_x, gt_y, gt_box_*, architecture_kind, num_layers, <params>, seed` — **no
   `search_gds_path`, no `theta`, no `scale`, no `present`** (the five `scale` matches in
   that file are all `polygon_scale_*`, i.e. fab distortion). The only code that writes a
   search GDS is Streamlit: `app.py:528` (bulk bookmark zip) and `app.py:862` (single
   sample), both via `_full_canvas_cell`.

   Neither path matches the manifest in the Phase 3 statement, so **some generator we do
   not have produces the blind set.** Three of our four methods need `search.gds`, and the
   +68 px canvas fix depends on how it is written. That fix was validated against a
   reimplementation of `_full_canvas_cell`, since `app.py` is a Streamlit script that
   cannot be imported; the two were then compared directly and are identical polygon for
   polygon, with the same bounding box (11360-11391 nm against a 10000 nm canvas) on both
   architectures. So the overflow is real in the organisers' own writer -- but whether the
   blind set's writer behaves the same is unverified.
4. Is the half-pixel `INTER_AREA` offset intended, or an artefact the real scoring
   harness does not have?
