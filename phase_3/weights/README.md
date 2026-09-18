# weights/

Empty, and the submission does not need anything here.

`phase3.py` checks for `model_best.pt` and uses it if present — the bundle carries its own
cfg, Platt calibration and found threshold, so nothing in the script needs editing. With
no file here, the re-ranker's zero-initialised head contributes an identical constant to
every candidate and the ranking reduces exactly to the classical similarity score, which
is what every number in `../README.md` was measured with.
