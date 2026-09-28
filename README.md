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

nuScenes validation split, 150 scenes, official 40-recall-threshold protocol
(devkit 1.1.11). The same detections, boxes and scores throughout — **only the
velocity information changes**.

| Detector | Tracker | Velocity | AMOTA ↑ | MOTA ↑ | IDS ↓ |
|---|---|---|---:|---:|---:|
| VoxelNeXt | this work | detector velocity | 0.6450 | 0.5524 | 694 |
| VoxelNeXt | this work | removed | 0.6106 | 0.5194 | 1634 |
| VoxelNeXt | this work | **removed + policy** | **0.6232** | **0.5388** | **902** |
| VoxelNeXt | this work | oracle (upper bound) | 0.6548 | 0.5637 | 472 |
| VoxelNeXt | Poly-MOT | removed | 0.6969 | 0.5927 | 472 |
| VoxelNeXt | Poly-MOT | **removed + policy** | **0.7026** | **0.6052** | **446** |
| TransFusion-L | this work | removed | 0.6469 | 0.5928 | 2053 |
| TransFusion-L | this work | **removed + policy** | **0.6598** | **0.6029** | **995** |

Paired scene bootstrap (4,000 resamples), AMOTA difference of the policy against
the velocity-removed run of the same row: +0.0126 [+0.0077, +0.0177],
+0.0057 [+0.0032, +0.0084] and +0.0129 [+0.0104, +0.0154] — none of the four
detector-tracker combinations has an interval containing zero.

KITTI tracking validation split (11 sequences, 3,904 scans), 2 Hz, Car, detections
regenerated from three released OpenPCDet checkpoints. Every tracker parameter is
the nuScenes value, unchanged.

| Detector | Car AP@R11 | MOTA, no velocity → + policy | IDS, no velocity → + policy |
|---|---:|---|---|
| PointRCNN | 78.70 | 0.6739 → 0.7275 | 97 → 33 |
| PV-RCNN | 83.61 | 0.7693 → 0.8122 | 34 → 14 |
| Voxel R-CNN (Car) | 84.54 | 0.7828 → 0.8334 | 53 → 29 |

Downstream: a constant-velocity predictor run on the tracker output lowers its
one-second displacement error by 23–34% for objects faster than 3 m/s, across four
detectors.

What the method does **not** do: it cannot improve box localisation (the tracker
submits the matched detection box verbatim), it costs about 20 ms per nuScenes
frame on one CPU thread, and at a low detection-confidence floor it makes
pedestrian identities worse on KITTI. `docs/WITHDRAWN_CLAIMS.md` lists every claim
an earlier draft made that the evidence did not support.

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
