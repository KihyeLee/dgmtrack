#!/usr/bin/env python3
"""nuScenes-style (centre distance) tracking metrics on KITTI, with per-sequence counts.

The KITTI verdict uses AB3DMOT's evaluator (tools/evaluate_kitti.py), which
only keeps IDS as a split-wide total, so it cannot be resampled by sequence.
This tool supplies the interval: it runs the unmodified devkit
TrackingEvaluation (centre distance 2 m, 40 recall thresholds) on the KITTI
ground truth, captures per-sequence counts at every threshold exactly as
tools/evaluate_saved_official_fast.py does, and writes them to
results/scene_counts/<label>/ so tools/paired_bootstrap_official40.py
resamples KITTI sequences with no change.

Ground truth: Car / Pedestrian / Cyclist of the keyframes only (the frames
the tracker submits); Van and DontCare are dropped. num_lidar_pts is the count
written by tools/kitti_to_dgm.py and, as in the devkit filter, boxes with zero
points are removed. No class-range filter is applied (KITTI labels only exist
inside the camera field of view, < 80 m).
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import evaluate_saved_official_fast as fast  # noqa: E402
from nuscenes.eval.common.config import config_factory  # noqa: E402
from nuscenes.eval.tracking.algo import TrackingEvaluation  # noqa: E402
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig, TrackingMetricData  # noqa: E402
from nuscenes.eval.tracking.loaders import interpolate_tracks  # noqa: E402
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom  # noqa: E402

CLASSES = ['car', 'pedestrian', 'bicycle']


class _SerialPool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def imap_unordered(self, func, it):
        return map(func, it)


def kitti_config():
    base = config_factory('tracking_nips_2019').serialize()
    base['tracking_names'] = CLASSES
    base['class_range'] = {c: base['class_range'][c] for c in CLASSES}
    return TrackingConfig.deserialize(base)


def build_tracks(data, results):
    keyframes = defaultdict(list)          # scene -> [(ts, token)]
    for path in sorted((data / 'detections').glob('*/*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                fr = json.loads(line)
                if fr['is_key_frame']:
                    keyframes[path.stem].append((fr['timestamp'], fr['sample_token']))
    gt = defaultdict(lambda: defaultdict(list))
    pred = defaultdict(lambda: defaultdict(list))
    for scene, frames in keyframes.items():
        cache = json.loads((data / 'gt' / ('%s.json' % scene)).read_text())['frames']
        for ts, token in sorted(frames):
            gt[scene][ts] = []
            pred[scene][ts] = []
            for o in cache[token]['objects']:
                if o['cls'] not in CLASSES or o['kitti_type'] not in ('Car', 'Pedestrian', 'Cyclist'):
                    continue
                if o['num_lidar_pts'] == 0:
                    continue
                gt[scene][ts].append(TrackingBox(
                    sample_token=token, translation=tuple(o['translation']), size=tuple(o['size']),
                    rotation=tuple(o['rotation']), velocity=tuple(o['velocity'] or (0.0, 0.0)),
                    num_pts=o['num_lidar_pts'], tracking_id=o['instance_token'], tracking_name=o['cls']))
            for b in results.get(token, []):
                pred[scene][ts].append(TrackingBox(
                    sample_token=token, translation=tuple(b['translation']), size=tuple(b['size']),
                    rotation=tuple(b['rotation']), velocity=tuple(b['velocity']),
                    tracking_id=b['tracking_id'], tracking_name=b['tracking_name'],
                    tracking_score=b['tracking_score']))
    # as devkit create_tracks: average score per track, then interpolate both
    for scene, tracks in pred.items():
        scores = defaultdict(list)
        for boxes in tracks.values():
            for b in boxes:
                scores[b.tracking_id].append(b.tracking_score)
        avg = {k: float(np.mean(v)) for k, v in scores.items()}
        for boxes in tracks.values():
            for b in boxes:
                b.tracking_score = avg[b.tracking_id]
    gt = {s: interpolate_tracks(t) for s, t in gt.items()}
    pred = {s: defaultdict(list, sorted(interpolate_tracks(t).items(), key=lambda kv: kv[0]))
            for s, t in pred.items()}
    return gt, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--label', required=True, help='written under results/scene_counts/<label>/')
    args = ap.parse_args()
    data = Path(args.data)
    submission = Path(args.run) / 'tracking_results.json'
    results = json.loads(submission.read_text())['results']
    cfg = kitti_config()
    gt, pred = build_tracks(data, results)

    TrackingMetricData.nelem = cfg.num_thresholds
    evaluations = {}
    for c in CLASSES:
        evaluations[c] = TrackingEvaluation(gt, pred, c, cfg.dist_fcn_callable, cfg.dist_th_tp, cfg.min_recall,
                                            num_thresholds=cfg.num_thresholds, metric_worst=cfg.metric_worst,
                                            verbose=False, output_dir=str(ROOT / 'results/kitti'),
                                            render_classes=[])
    fast._EVALUATIONS.clear()
    fast._EVALUATIONS.update(evaluations)
    fast._CAPTURE['enabled'] = True
    MOTAccumulatorCustom.merge_event_dataframes = staticmethod(fast._merge_capturing)
    fast.multiprocessing.Pool = _SerialPool
    md_list, prepared, counts_by_job = fast.accumulate_parallel(evaluations, 1)

    scenes = list(gt.keys())
    counts = np.zeros((len(CLASSES), cfg.num_thresholds, len(scenes), len(fast.COUNT_NAMES)))
    thresholds = np.full((len(CLASSES), cfg.num_thresholds), np.nan)
    recalls = np.full((len(CLASSES), cfg.num_thresholds), np.nan)
    for ci, c in enumerate(CLASSES):
        if prepared[c] is None:
            continue
        th, rc = prepared[c][0], prepared[c][1]
        thresholds[ci], recalls[ci] = th, rc
        for i, t in enumerate(th):
            if not np.isnan(t):
                counts[ci, i] = counts_by_job[(c, t)]
    out = ROOT / 'results/scene_counts' / args.label
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'scene_counts.npz', counts=counts, thresholds=thresholds, recalls=recalls,
                        classes=np.array(CLASSES), scenes=np.array(scenes), count_names=np.array(fast.COUNT_NAMES))
    metrics = fast.aggregate(cfg, md_list).serialize()
    summary = {k: metrics[k] for k in ('amota', 'amotp', 'mota', 'ids', 'frag', 'fp', 'fn', 'recall')}
    summary['label_metrics'] = {k: metrics['label_metrics'][k] for k in ('amota', 'mota', 'ids')}
    (out / 'provenance.json').write_text(json.dumps({
        'run': args.label, 'submission': str(submission), 'submission_sha256': fast.sha256(submission),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'tools/evaluate_kitti_center.py (devkit TrackingEvaluation, centre distance, KITTI GT)',
        'count_names': fast.COUNT_NAMES, 'classes': CLASSES, 'summary': summary}, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
