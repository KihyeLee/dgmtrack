# DGMTrack — a local dynamic occupancy grid inside detection boxes, for 3D multi-object tracking

When a 3D detector reports no velocity, an object's velocity has to come from the
tracker's own history — which a newly created track does not have. This code builds
a small particle-based dynamic occupancy grid **inside each detection box**,
aggregates the cell velocities to the object level, and gives that velocity to the
tracker **only where the tracker has nothing better**: as a new track's prior, and
in the updates of tracks with at most two previous hits.

The boundary is not a tuned constant. Matching 72,794 object-frames against ground
truth shows the grid is more accurate than the tracker for the first three
keyframes of a track and worse after that; the policy uses the grid exactly on the
near side of that crossover.

## Results

The claim is a **within-row difference**: the same detections, boxes and scores, with only
the velocity information changed. It is not a ranking of trackers — "CV + greedy" is a plain
constant-velocity Kalman filter with greedy matching, and Poly-MOT is stronger for reasons
that have nothing to do with this work. What the policy adds, on the nuScenes validation
split under the official 40-recall-threshold protocol:

| Tracker | Detector | ΔAMOTA ↑ | ΔIDS ↓ | Δ 1 s prediction error ↓ |
|---|---|---|---|---|
| CV + greedy | VoxelNeXt | **+0.0126** [+0.0077, +0.0177] | **−732** [−1023, −491] | −27.3% |
| CV + greedy | TransFusion-L | **+0.0129** [+0.0104, +0.0154] | **−1058** [−1450, −720] | −33.7% |
| Poly-MOT | VoxelNeXt | **+0.0057** [+0.0032, +0.0084] | −26 [−128, +57] | — |
| Poly-MOT | TransFusion-L | **+0.0044** [+0.0021, +0.0075] | −153 [−268, −59] | — |

Intervals are a paired scene bootstrap, 4,000 resamples. The AMOTA interval excludes zero in
all four detector–tracker combinations. The identity-switch interval does not, on one of
them, and that is reported rather than hidden. The last column is the one-second displacement
error of a constant-velocity predictor run on the tracker output, for objects faster than
3 m/s — the quantity a downstream planner actually consumes, and where the effect is largest.

The absolute numbers behind those differences:

| Detector | Tracker | Velocity | AMOTA ↑ | MOTA ↑ | IDS ↓ |
|---|---|---|---:|---:|---:|
| VoxelNeXt | CV + greedy | detector velocity | 0.6450 | 0.5524 | 694 |
| | | velocity removed | 0.6106 | 0.5194 | 1634 |
| | | **removed + proposed** | **0.6232** | **0.5388** | **902** |
| | | oracle (upper bound) | 0.6548 | 0.5637 | 472 |
| | Poly-MOT | velocity removed | 0.6969 | 0.5927 | 472 |
| | | **removed + proposed** | **0.7026** | **0.6052** | **446** |
| TransFusion-L | CV + greedy | velocity removed | 0.6469 | 0.5928 | 2053 |
| | | **removed + proposed** | **0.6598** | **0.6029** | **995** |
| | Poly-MOT | velocity removed | 0.6816 | 0.6150 | 568 |
| | | **removed + proposed** | **0.6860** | **0.6233** | **415** |

The oracle injects the ground-truth velocity through the same measurement path and bounds
what any velocity source can give this pipeline; the policy recovers 28–30% of that headroom.

The validation split was also used to design the policy, so the same two Poly-MOT
configurations were submitted once to the official nuScenes **test** server, a split this work
never looked at, with every parameter left at its validation value:

| nuScenes test split, official server | AMOTA ↑ | AMOTP ↓ | MOTA ↑ | RECALL ↑ |
|---|---:|---:|---:|---:|
| velocity removed | 0.6524 | 0.5766 | 0.5449 | 0.6935 |
| **velocity removed + proposed** | **0.6614** | 0.5798 | **0.5532** | **0.6970** |
| difference | **+0.0090** | +0.0032 | +0.0083 | +0.0035 |

The difference is the same direction as validation (+0.0057 for that pair) and does not shrink.
The server returns totals only, so no interval can be computed on this split; it confirms the
validation control rather than replacing it.

On KITTI, whose detectors report no velocity at all, the detections were regenerated from
three released OpenPCDet checkpoints spanning Car AP@R11 78.70 to 84.54, with every tracker
parameter left at its nuScenes value:

| Detector | Car AP@R11 | Car MOTA, no velocity → + proposed | IDS, no velocity → + proposed |
|---|---:|---|---|
| PointRCNN | 78.70 | 0.6739 → **0.7275** | 97 → **33** |
| PV-RCNN | 83.61 | 0.7693 → **0.8122** | 34 → **14** |
| Voxel R-CNN (Car) | 84.54 | 0.7828 → **0.8334** | 53 → **29** |

The identity and accuracy gains do not depend on the detector generation. The
recall-averaged sAMOTA does, and `results/kitti/SAMOTA_THRESHOLD_DIAG.json` shows why: its
lowest recall point contributes zero when a single operating point has no true positive.

The AMOTA gain is small, and the paper says so. Recall-averaged metrics add misses, false
positives and identity switches with equal weight, and identity switches are a small part of
the total, so a method that only reduces them cannot move such a metric much. That is why the
downstream prediction error is reported alongside.

What the method does **not** do: it cannot improve box localisation (the tracker submits the
matched detection box verbatim), it costs about 20 ms per nuScenes frame on one CPU thread,
and at a low detection-confidence floor it makes pedestrian identities worse on KITTI.
`docs/WITHDRAWN_CLAIMS.md` lists every claim an earlier draft made that the evidence did not
support.

## Judgement criteria and their outcomes

`results/PREREGISTRATION.md` holds the criterion for every experiment together
with what happened, including the ones that were **not** met — the class-wise
velocity rules were rejected on nuScenes, and the KITTI detector-generation test
came out "partly reproduced" on the published sAMOTA. Nothing in that file has
been edited after the fact; entries are appended.

## Building

```bash
g++ -std=c++17 -O2 src/apps/dogm_track.cpp -o build/dogm_track \
    -Isrc -Isrc/common -Isrc/tracker -Isrc/dogm -I/usr/include/eigen3
```
Needs Eigen 3 and a C++17 compiler. Analysis scripts need Python 3.8 with
`nuscenes-devkit==1.1.11`, `motmetrics==1.1.3`, numpy, scipy and pandas.
The devkit's tracking accumulator is not compatible with motmetrics 1.4.

## Reproducing the KITTI result

KITTI tracking (velodyne, calib, label_02, oxts) and an AB3DMOT checkout are the
only external inputs; both are public.

```bash
# 1. detections from a released OpenPCDet checkpoint, written in AB3DMOT's format
python tools/kitti_tracking_infer.py \
    --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt pv_rcnn_8369.pth \
    --kitti_root <KITTI>/tracking --seqmap <AB3DMOT>/scripts/KITTI/evaluate_tracking.seqmap.val \
    --out <DET> --det_name pvrcnn --split val

# 2. KITTI expressed in the conventions the tracker reads
python tools/kitti_to_dgm.py --kitti-root <KITTI>/tracking --ab3dmot <AB3DMOT> \
    --det-root <DET> --det-name pvrcnn --key-every 5 --out <DATA>

# 3. the two runs the comparison needs
build/dogm_track --pose_cache <DATA>/poses.json --det_dir <DATA>/detections/pvrcnn \
    --dataroot <KITTI>/tracking --out_dir <RUN>/none \
    --vel_source none --fusion none --feedback 0 --score_thr 0.1 --seed 12345 --box_output det
build/dogm_track ... --out_dir <RUN>/policy \
    --vel_source dogm --fusion kf --vel_max_hits 2 --feedback 0 --score_thr 0.1 --seed 12345 --box_output det

# 4. AB3DMOT's own 3D MOT evaluation (3D IoU 0.25)
python tools/evaluate_kitti.py --run <RUN>/policy --data <DATA> --det-name pvrcnn --label policy
```

`tools/kitti_tracking_infer.py` runs inside an OpenPCDet checkout (copy it into
`tools/` there). Note that KITTI has no velodyne for frames 177–180 of sequence
0001, so the split has 3,904 scans rather than 3,908.

Copy `configs/paths.example.json` to `configs/paths.json` and fill it in;
`tools/evaluate_kitti.py` reads it to locate the AB3DMOT checkout when
`--ab3dmot` is not given.

## What is here

| Path | |
|---|---|
| `src/` | the tracker and the in-box DOGM (C++17, header-only dependencies) |
| `tools/kitti_*.py` | KITTI conversion, inference, evaluation, per-threshold diagnostic |
| `tools/evaluate_saved_official_fast.py` | nuScenes official 40-threshold evaluation, parallel over thresholds |
| `tools/paired_bootstrap_official40.py` | paired scene bootstrap for any two runs |
| `tools/early_track_velocity*.py` | velocity error against time since track birth |
| `tools/validate_nusc_submission.py` | checks a submission the way the server will |
| `results/PREREGISTRATION.md` | criteria and outcomes for every experiment |
| `docs/` | withdrawn claims, the KITTI detector-generation table |

Detections, tracker outputs, caches and checkpoints are not in the repository.
