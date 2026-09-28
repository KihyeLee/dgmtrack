#!/usr/bin/env python3
"""Calibration scale kappa for the grid velocity measurement noise (PREREGISTRATION 2026-09-24, experiment 1).

The tracker weighs a grid velocity z_b with the noise it actually applies,
R_v = (S_b + 0.5^2 I)(1 + 2(1 - c_b)) (src/tracker/track_kf.hpp, update_velocity).
If R_v were calibrated, NEES = e' R_v^-1 e with e = z_b - v_gt would follow a chi-square
with two degrees of freedom, whose median is 2 ln 2 = 1.386. The preregistered scale is

    kappa = median(NEES) / 1.386

over every matched keyframe detection with a valid grid velocity in the calibration data
that also set the track-age boundary (MEGVII grid report on the nuScenes validation split).
Matching follows tools/calibrate_grid_velocity.py (one-to-one, same class, within 2 m).
Also reported, not used for kappa: the same statistic by track age and by motion.
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

FLOOR = 0.5          # tracker sigma_v_floor
MED_CHI2_2 = 2.0 * np.log(2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid-run', default='mg_grid_report')
    ap.add_argument('--baseline', default='n150_mg_none')
    ap.add_argument('--out', default='results/grid_r_scale_calibration.json')
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    baseline = json.loads((Path(paths['original']) / 'work/results' / args.baseline
                           / 'tracking_results.json').read_text())['results']
    scenes = [s for s in (ROOT / 'configs/val150_scenes.txt').read_text().split() if s]

    rec = []   # (nees_Rv, nees_S, age, gt_speed)
    for scene in scenes:
        cached = json.loads((Path(paths['gt_cache']) / f'{scene}.json').read_text())['frames']
        path = ROOT / 'results/tracker_runs' / args.grid_run / 'scenes' / scene / f'{scene}.jsonl'
        seen = defaultdict(int)
        for line in path.open():
            row = json.loads(line)
            if not row.get('is_key_frame') or row['sample_token'] not in cached:
                continue
            gt = [g for g in cached[row['sample_token']]['objects']
                  if g['num_lidar_pts'] > 0 and g['velocity'] is not None]
            boxes = [b for b in row['boxes'] if b['dogm_valid']]
            tracker_boxes = {diagnostic.geometry_key(b): b for b in baseline[row['sample_token']]}
            pairs = diagnostic.match_one_to_one(
                [{'cls': g['cls'], 'translation': g['translation']} for g in gt],
                [{'tracking_name': b['detection_name'], 'translation': b['translation']} for b in boxes])
            for gi, bi in pairs:
                g, b = gt[gi], boxes[bi]
                v_gt = np.asarray(g['velocity'][:2])
                e = np.asarray(b['velocity']) - v_gt
                S = np.array([[b['dogm_cov'][0], b['dogm_cov'][1]], [b['dogm_cov'][1], b['dogm_cov'][2]]])
                c = min(1.0, max(1e-3, b['dogm_conf']))
                Rv = (S + FLOOR ** 2 * np.eye(2)) * (1.0 + 2.0 * (1.0 - c))
                try:
                    n_rv = float(e @ np.linalg.solve(Rv, e))
                    n_s = float(e @ np.linalg.solve(S, e))
                except np.linalg.LinAlgError:
                    continue
                key = diagnostic.geometry_key({'tracking_name': b['detection_name'], 'translation': b['translation'],
                                               'size': b['size'], 'rotation': b['rotation']})
                tr = tracker_boxes.get(key)
                age = seen[tr['tracking_id']] if tr is not None else -1
                rec.append((n_rv, n_s, age, float(np.linalg.norm(v_gt))))
            for b in baseline[row['sample_token']]:
                seen[b['tracking_id']] += 1

    def st(sel):
        a = np.array([r[0] for r in sel])
        s = np.array([r[1] for r in sel])
        return {'n': int(a.size), 'median_nees_Rv': float(np.median(a)) if a.size else None,
                'mean_nees_Rv': float(np.mean(a)) if a.size else None,
                'kappa_median': float(np.median(a) / MED_CHI2_2) if a.size else None,
                'median_nees_S': float(np.median(s)) if s.size else None}
    out = {'grid_run': args.grid_run, 'floor': FLOOR, 'estimator': 'kappa = median(NEES_Rv) / (2 ln 2)',
           'overall': st(rec)}
    out['kappa'] = out['overall']['kappa_median']
    out['report_only'] = {
        'age_0_2': st([r for r in rec if 0 <= r[2] <= 2]), 'age_ge3': st([r for r in rec if r[2] >= 3]),
        'moving_ge1': st([r for r in rec if r[3] >= 1.0]), 'static_lt1': st([r for r in rec if r[3] < 1.0])}
    (ROOT / args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'kappa': out['kappa'], 'overall': out['overall'], **out['report_only']}, indent=1))


if __name__ == '__main__':
    main()
