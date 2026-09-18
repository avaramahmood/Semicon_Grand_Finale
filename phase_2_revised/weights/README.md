# weights/

`model_best.pt` — the shipped model. `phase2.py` loads it automatically; it carries its
own cfg, Platt calibration and F1-optimal found threshold, so the checkpoint decides the
settings and nothing in the script needs editing.

It is the **intensity** variant, chosen by running all three trained variants over the
same two test sets rather than trusting their individual training logs. See `../METHODS.md`
section 6 and 7.

| | organisers' 25 | held-out 120 | s/pair |
|---|---|---|---|
| intensity (this file) | 83.00 | 80.32 | 0.86 |
| hybrid | 82.65 | 80.13 | 1.61 |
| edge | 82.10 | 77.67 | 0.95 |

Bundle contents: `state_dict`, the `cfg` it was trained under, `edge_state` for the
boundary detector, `edge_choice` (`classical` — the learned detector lost to classical
gradients on validation), `platt`, `threshold`, `steps` and the best `val` record.
