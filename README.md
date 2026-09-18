# Drift-Sense — Grand Finale

Two submissions, one matching core.

**Phase 2 revised** registers an SEM reference against an SEM search image.
**Phase 3** registers a GDSII design against one. Both report the same thing: where the
reference is, at what zoom and rotation, whether it is there at all, and how confident we
are.

![Overview](overview.png)

| | scored on | result | per pair | weights |
|---|---|---|---|---|
| [`phase_2_revised/`](phase_2_revised/) | the organisers' 25-pair set | **83.00 / 85** | 0.90 s | trained, shipped |
| [`phase_3/`](phase_3/) | 50-pair blind split | **83.62 / 85** | 3.12 s | none — classical |

Both run on CPU with no network, well inside the 5 s median budget and the 20 s hard
timeout. The 85 is what a local run can compute: localization 40, pose 20, rejection 15,
calibration 10. The other 15 — generator analysis 10, efficiency 5 — belong to the jury,
and efficiency is ranked against other entrants rather than scored absolutely.

---

## Run either one

```bash
cd phase_2_revised   &&  python -m pip install -r requirements.txt
python phase2.py --input <dataset>/pairs.csv --output predictions.csv
python score.py  --truth <dataset>/ground_truth.csv --pred predictions.csv
```

```bash
cd phase_3   &&  python -m pip install -r requirements.txt
python phase3.py --input <dataset>/pairs.csv --output predictions.csv
python score.py  --truth <dataset>/ground_truth.csv --pred predictions.csv
```

Paths inside `pairs.csv` resolve relative to the directory holding it. Phase 3 needs
`gdstk` to read the design files; Phase 2 does not.

## What the two problems share

`src/dsr_core.py` is the same file in both packages, byte for byte. It owns everything
after the phase-specific front end:

- **K = 64 candidates** — the top peaks per pose, non-maximum suppressed
- **a cross-encoder** that scores all 64 jointly, with `logit = head(e) + α·classical_score`
  and the head zero-initialised, so a trained model is a *correction* to the classical
  ranking rather than a replacement for it
- **refinement** — sub-pixel peak interpolation, then three levels of pose refinement
- **a presence head** over pose-surface statistics (peak, margin, entropy, std, PSR),
  Platt-calibrated with an F1-optimal threshold stored in the checkpoint

## Where they differ

**Phase 2** has two SEM images and nothing to infer: correlate the reference against the
search over a 5×5 zoom/rotation grid, as band-matched ZNCC in the Fourier domain.

**Phase 3** has a design file with no brightness anywhere in it, and that changes the shape
of the problem. `search.gds` covers the whole search image, so the pose becomes a
least-squares fit over all 1,000,000 pixels rather than a patch match, and the location
becomes a noise-free search between two designs. The per-layer greys are solved from the
image at every candidate pose — and then used for the **confidence score**, never for the
match, because a brightness fit you had to guess at is exactly the wrong thing to register
on. Registration runs on per-layer signed edges instead.

Each package has a `method.png` walking through its pipeline on a real pair, with every
panel a genuine intermediate rather than an illustration.

## Two bugs worth reading about

Both were found only by running against real files through a real entry point, and either
alone would have scored near zero on localization while looking perfect in development.

- **The canvas size is not in the GDS.** The design polygons overflow the rendered canvas,
  so deriving the canvas from the polygon bounding box put the CAD→SEM mapping out by up to
  132 px in both axes. 11 of 48 pairs then picked a zoom 20–48% wrong. Deriving it from the
  candidate pose instead took Phase 3 from **68.39 to 84.12**. → `phase_3/METHODS.md`
- **The generator's label is not where the feature is visible**, by a half-pixel downsample
  convention in both axes and by scan drift it never corrects for. Worth ~8 of the 40
  localization points. → `phase_3/NOTES.md`

## Layout

```
overview.png              the figure above
phase_2_revised/          phase2.py, score.py, src/, weights/model_best.pt, results/
phase_3/                  phase3.py, score.py, src/, results/
```

Each package is self-contained: nothing outside its own directory, no build step, and
pinned dependencies. Together they are about 10 MB, most of it the Phase 2 checkpoint and
the figures.

## Honest gaps

- **Phase 3 is untrained.** Its numbers are the classical pipeline, which is deterministic
  by construction — the zero-init head makes the ranking exactly the classical score, and
  six random seeds give the same answer to the pixel.
- **`pairs.csv` is assumed to sit at the dataset root** in both entry points.
- **Rejection rests on few absent pairs** — 5 in the Phase 2 real set, 16 across the Phase 3
  sets. Perfect F1 there is encouraging, not established.
- **Three questions for the organisers** about the Phase 3 blind set, none answerable from
  the code they published. → `phase_3/NOTES.md`
