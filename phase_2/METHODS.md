# Phase 2 — method

Register an SEM reference against a noisy SEM search image. Both sides are images, so
there is no design file and no per-layer brightness to infer — the whole problem is
finding the reference under an unknown zoom, rotation, scan drift and heavy noise, and
knowing when it is not there at all.

---

## 1. Candidate search

Band-matched ZNCC between the reference and the search, over a 5×5 grid of zoom and
rotation, computed as FFT correlation. K=64 candidates survive non-maximum suppression.

Settings were swept on the organisers' 25-pair set and a held-out generated set:

| knob | chosen | evidence |
|---|---|---|
| prefilter σ | **0.5** | pose credit 0.960 at σ0.5 against **0.848** at σ2, which was the Phase 1 default. σ3 collapses to 0.754 — Phase 1's value does not transfer to 8 layers |
| band-pass (DoG) | **off** | pose 0.870 (ratio 3) / 0.874 (ratio 6) against 0.935 plain |
| Anscombe VST | **off** | +0.017 pose at σ2 only, nothing at the chosen σ |
| median 3×3 | **off** | 0.824 against 0.828 |
| sub-pixel | **2-D quadratic** | no sub-pixel ≤1 px 65%, centroid 77.5% (pixel-locking bias), quadratic 82.5% |
| refinement levels | **3** | pose 0.853 against 0.828, at no measurable cost |
| K | **64** | 32 / 64 / 128 identical once recall@K reached 1.000 |
| pose grid | **25** | 81 poses bought +0.016 pose for +30% time |

## 2. Re-ranking

A cross-encoder sees all 64 candidates jointly — both patches, their similarity
statistics and the pose-surface context — and scores them. Its logit is

```
logit = head(e) + alpha · classical_score
```

with `head`'s output layer zero-initialised, so at the start of training the ranking is
exactly the classical score and the network learns a *correction* rather than having to
rediscover matching from scratch.

## 3. Refinement

Sub-pixel peak interpolation followed by three refinement levels over translation, zoom
and rotation.

## 4. Presence

A head over the raw pose-surface statistics — peak, margin, entropy, standard deviation
and PSR — rather than the peak value alone, because no single statistic is reliable across
conditions. Platt-calibrated after training, with an F1-optimal threshold chosen on
held-out validation. Both are saved in the checkpoint, so `register.py --threshold` defaults
to the bundle's value.

PSR is the strongest single statistic (AUC 0.939 on the edge variant), which is why it is
in the feature set at all.

## 5. What training actually bought

Untrained, the same pipeline scored **loc 0.970, pose 0.960** on the organisers' 25 pairs.
Trained, it scores **loc 0.970, pose 0.960** — identical.

Training moved **rejection and calibration**: those are what an untrained presence head
cannot do, and they are 25 of the 85 local points. On the 120-pair held-out set the
shipped model reaches F1 0.976 / AUC 0.994.

This is worth stating plainly because it sets expectations: the geometry was already right
before training, and the network's job was to decide *whether the reference is there*.

## 6. The three variants

| variant | cue | organisers' 25 | held-out 120 | s/pair |
|---|---|---|---|---|
| **intensity** | band-matched ZNCC | **83.00** | **80.32** | **0.86** |
| hybrid | intensity ∪ edge, modality dropout | 82.65 | 80.13 | 1.61 |
| edge | learned boundary detector → edge correlation | 82.10 | 77.67 | 0.95 |

The hybrid used modality dropout (15% per cue) so the strong cue could not starve the weak
one, and fused both inside the network rather than averaging scores. It still did not beat
intensity alone on either set: identical localization and pose, slightly worse rejection,
1.9× the cost. On SEM-vs-SEM the second cue has nothing to add — both sides are the same
modality, which is exactly the case where edges stop being the common ground they are in
Phase 3.

**The learned boundary detector lost to classical gradients** on validation in all three
runs (every checkpoint carries `edge_choice='classical'`). The pipeline measures recall@K
for learned against classical on validation and keeps whichever is higher, so the learned
net could never be worse than the classical baseline — it simply was not better.

## 7. Why the model selection was redone here

Each checkpoint's own validation record ranked them hybrid 80.90 > intensity 80.58 > edge
80.15. Those are three *different* validation draws and are not comparable. Re-running all
three over one common set reversed the order, and the same winner held on the organisers'
real pairs. Model selection was therefore made on the two shared sets, not on the training
logs.
