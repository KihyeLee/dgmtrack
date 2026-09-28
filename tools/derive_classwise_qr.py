#!/usr/bin/env python3
"""Derive per-class Kalman noise values from ground truth, once, for the Q/R baseline.

  r_pos[c]  RMS horizontal centre error of matched detections of class c
            (same-class one-to-one match within 2 m, score >= 0.1)
  q_vel[c]  var(delta v) / dt of annotated GT velocity between consecutive
            keyframes (dt ~ 0.5 s), averaged over the two axes

These are computed from the validation split that the grid policy was also
designed on, so the baseline gets the same information advantage. No search
over values is performed; the numbers are used as derived.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('diagnostic', ROOT / 'tools/prediction_diagnostic.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', default='det150_megvii')
    parser.add_argument('--score-min', type=float, default=0.1)
    args = parser.parse_args()
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    det_dir = Path(paths['original']) / 'cache' / args.detections
    scenes = [s for s in (ROOT / 'configs/val150_scenes.txt').read_text().split() if s]

    pos_err = defaultdict(list)
    dv = defaultdict(list)
    for scene in scenes:
        cached = json.loads((Path(paths['gt_cache']) / f'{scene}.json').read_text())['frames']
        detections = {}
        with (det_dir / f'{scene}.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('is_key_frame'):
                    detections[row['sample_token']] = [
                        {'tracking_name': b['detection_name'], 'translation': b['translation']}
                        for b in row['boxes'] if b['detection_score'] >= args.score_min]
        series = defaultdict(list)
        for token in sorted(cached, key=lambda t: cached[t]['timestamp_s']):
            frame = cached[token]
            gt = [g for g in frame['objects'] if g['num_lidar_pts'] > 0]
            for gi, bi in diagnostic.match_one_to_one(
                    [{'cls': g['cls'], 'translation': g['translation']} for g in gt],
                    detections.get(token, [])):
                d = detections[token][bi]
                pos_err[gt[gi]['cls']].append(np.hypot(gt[gi]['translation'][0] - d['translation'][0],
                                                       gt[gi]['translation'][1] - d['translation'][1]))
            for g in frame['objects']:
                if g['velocity'] is not None:
                    series[g['instance_token']].append((frame['timestamp_s'], g['cls'], np.asarray(g['velocity'][:2])))
        for values in series.values():
            values.sort(key=lambda v: v[0])
            for (t0, cls, v0), (t1, _, v1) in zip(values, values[1:]):
                dt = t1 - t0
                if 0.4 <= dt <= 0.6:
                    dv[cls].append(((v1 - v0) ** 2) / dt)

    report = {'detections': str(det_dir), 'score_min': args.score_min, 'classes': {}}
    for cls in sorted(set(pos_err) | set(dv)):
        e = np.asarray(pos_err[cls]); q = np.asarray(dv[cls])
        report['classes'][cls] = {
            'n_matched_detections': int(e.size), 'r_pos_rms_m': float(np.sqrt(np.mean(e ** 2))) if e.size else None,
            'r_pos_median_m': float(np.median(e)) if e.size else None,
            'n_velocity_pairs': int(len(q)), 'q_vel_m2s3': float(np.mean(q)) if len(q) else None}
    out = ROOT / 'results/classwise_qr' / f'{args.detections}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    spec_r = ','.join(f"{c}:{v['r_pos_rms_m']:.3f}" for c, v in report['classes'].items() if v['r_pos_rms_m'])
    spec_q = ','.join(f"{c}:{v['q_vel_m2s3']:.3f}" for c, v in report['classes'].items() if v['q_vel_m2s3'])
    print('r_pos:', spec_r); print('q_vel:', spec_q)
    for c, v in report['classes'].items():
        print(f"  {c:<12} r_pos rms {v['r_pos_rms_m']:.3f} (median {v['r_pos_median_m']:.3f}, n={v['n_matched_detections']})  q_vel {v['q_vel_m2s3']:.3f} (n={v['n_velocity_pairs']})")


if __name__ == '__main__':
    main()
