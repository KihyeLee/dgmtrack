# Judgement criteria and what happened

`PREREGISTRATION.md` is the working file, reproduced here **verbatim and unedited**.
Entries are appended, never revised, which is the point: several of them record criteria
that were **not** met, and one records a recommendation that a later entry withdraws.
Editing it to tidy it up would remove the only thing it is good for.

Because it is the working file, it refers to paths on the machine the experiments ran on,
and to internal working notes that are not part of this repository:

| Referenced | Why it is not here |
|---|---|
| `docs/KITTI_PREREG_DRAFT_20260923.md`, `docs/V9_REVIEW_AND_PLAN_20260923.md` | superseded drafts of entries that appear in the file itself |
| `docs/KITTI_RESULTS_20260923.md` | generated table, regenerate with `tools/summarize_kitti.py` |
| `docs/SCI_READINESS_20260928.md` | internal notes on where to submit; not a result |
| absolute paths under `/media/...` | the drive layout of the machine that ran the experiments |

## What is here

| | |
|---|---|
| `PREREGISTRATION.md` | every experiment: the criterion written for it, and the outcome |
| `kitti/SUMMARY_MODERNDET.json` | three KITTI detectors, velocity removed vs. the proposed policy |
| `kitti/SAMOTA_THRESHOLD_DIAG.json` | per-threshold sMOTA, showing where the sAMOTA difference comes from |
| `early_track_velocity_moderndet/summary.json` | velocity error against time since track birth, per detector |
| `grid_velocity_calibration*.json` | grid velocity against ground truth, overall and per class |
| `official40/bootstrap_*.json` | paired scene bootstrap of one run against another |

Detections, tracker outputs and checkpoints are not in the repository; the commands that
produce them are in the top-level README.
