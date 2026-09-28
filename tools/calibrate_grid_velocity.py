#!/usr/bin/env python3
"""Measure how reliable the per-detection grid velocity is, and whether its own
confidence and covariance know it.

For every keyframe detection that the grid marked valid, the detection is
matched one-to-one to a visible ground-truth object of the same class within
2 m (the same rule as the prediction diagnostic) and the grid velocity is
compared with the annotated velocity. The same boxes are also scored for the
baseline tracker's own published velocity (constant-velocity estimate from
positions) so the grid is judged against what the tracker already had.

Reports, per confidence decile, per support bin and per class:
  n, median |e|, P(|e| < 1 m/s), mean NEES = e' S^-1 e (2 for a calibrated S)
Ground truth is used for scoring only.
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


def summarise(errors, nees, speeds):
    errors = np.asarray(errors)
    return {'n': int(errors.size),
            'median_abs_error_mps': float(np.median(errors)) if errors.size else None,
            'p_within_1mps': float((errors < 1.0).mean()) if errors.size else None,
            'mean_abs_error_mps': float(errors.mean()) if errors.size else None,
            'mean_nees': float(np.mean(nees)) if nees else None,
            'median_gt_speed_mps': float(np.median(speeds)) if speeds else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-run', default='mg_grid_report')
    parser.add_argument('--baseline', default='n150_mg_none')
    parser.add_argument('--out', default='results/grid_velocity_calibration.json')
    args = parser.parse_args()
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    baseline = json.loads((Path(paths['original']) / 'work/results' / args.baseline
                           / 'tracking_results.json').read_text())['results']
    scenes = [s for s in (ROOT / 'configs/val150_scenes.txt').read_text().split() if s]
    poses = json.loads(Path(paths['pose_cache']).read_text())['frames']

    records = []   # (conf, cells, class, gt_speed, grid_err, nees, tracker_err, cold, age)
    unmatched = 0
    for scene in scenes:
        cached = json.loads((Path(paths['gt_cache']) / f'{scene}.json').read_text())['frames']
        path = ROOT / 'results/tracker_runs' / args.grid_run / 'scenes' / scene / f'{scene}.jsonl'
        seen = defaultdict(int)   # baseline tracking_id -> keyframes already observed in this scene
        for line in path.open():
            row = json.loads(line)
            if not row.get('is_key_frame') or row['sample_token'] not in cached:
                continue
            gt = [g for g in cached[row['sample_token']]['objects']
                  if g['num_lidar_pts'] > 0 and g['velocity'] is not None]
            boxes = [b for b in row['boxes'] if b['dogm_valid']]
            gt_for_match = [{'cls': g['cls'], 'translation': g['translation']} for g in gt]
            box_for_match = [{'tracking_name': b['detection_name'], 'translation': b['translation']}
                             for b in boxes]
            tracker_boxes = {diagnostic.geometry_key(b): b for b in baseline[row['sample_token']]}
            for gi, bi in diagnostic.match_one_to_one(gt_for_match, box_for_match):
                g, b = gt[gi], boxes[bi]
                v_gt = np.asarray(g['velocity'][:2])
                e = np.asarray(b['velocity']) - v_gt
                S = np.array([[b['dogm_cov'][0], b['dogm_cov'][1]], [b['dogm_cov'][1], b['dogm_cov'][2]]])
                try:
                    nees = float(e @ np.linalg.solve(S, e))
                except np.linalg.LinAlgError:
                    nees = float('nan')
                key = diagnostic.geometry_key({'tracking_name': b['detection_name'],
                                               'translation': b['translation'], 'size': b['size'],
                                               'rotation': b['rotation']})
                tracker = tracker_boxes.get(key)
                tracker_err = (float(np.linalg.norm(np.asarray(tracker['velocity']) - v_gt))
                               if tracker is not None else None)
                if tracker is None:
                    unmatched += 1
                cold = tracker is not None and float(np.linalg.norm(tracker['velocity'])) == 0.0
                age = seen[tracker['tracking_id']] if tracker is not None else -1
                ego = poses.get(row['sd_token'], {}).get('ep_t')
                rng = float(np.hypot(b['translation'][0] - ego[0], b['translation'][1] - ego[1])) if ego else float('nan')
                records.append((b['dogm_conf'], b['dogm_cells'], b['detection_name'],
                                float(np.linalg.norm(v_gt)), float(np.linalg.norm(e)), nees, tracker_err, cold, age, rng))
            for b in baseline[row['sample_token']]:
                seen[b['tracking_id']] += 1

    conf = np.array([r[0] for r in records])
    edges = np.quantile(conf, np.linspace(0, 1, 11))
    report = {'grid_run': args.grid_run, 'baseline': args.baseline, 'matched_detections': len(records),
              'matched_without_tracker_box': unmatched,
              'overall': {}, 'by_confidence_decile': [], 'by_support_cells': [], 'by_class': [],
              'by_gt_speed': []}

    def block(subset):
        grid = summarise([r[4] for r in subset], [r[5] for r in subset if np.isfinite(r[5])],
                         [r[3] for r in subset])
        tr = [r[6] for r in subset if r[6] is not None]
        grid['tracker_cv_median_abs_error_mps'] = float(np.median(tr)) if tr else None
        grid['tracker_cv_p_within_1mps'] = float((np.asarray(tr) < 1.0).mean()) if tr else None
        grid['zero_velocity_median_abs_error_mps'] = float(np.median([r[3] for r in subset])) if subset else None
        return grid

    report['overall'] = block(records)
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        subset = [r for r in records if (lo <= r[0] < hi) or (i == 9 and r[0] == hi)]
        report['by_confidence_decile'].append({'decile': i + 1, 'conf_range': [float(lo), float(hi)], **block(subset)})
    for lo, hi in ((0, 10), (10, 25), (25, 50), (50, 100), (100, 10 ** 9)):
        subset = [r for r in records if lo <= r[1] < hi]
        report['by_support_cells'].append({'cells_range': [lo, hi], **block(subset)})
    for name in sorted(set(r[2] for r in records)):
        report['by_class'].append({'class': name, **block([r for r in records if r[2] == name])})
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 3), (3, 8), (8, 100)):
        subset = [r for r in records if lo <= r[3] < hi]
        report['by_gt_speed'].append({'gt_speed_range': [lo, hi], **block(subset)})
    # Cold start: the baseline tracker published exactly zero velocity for this
    # box, i.e. the track had no history to derive a velocity from. This is
    # where an external velocity has no competitor.
    # Range from the ego vehicle, split by GT motion so composition does not hide the trend.
    report['by_range'] = []
    for lo, hi in ((0, 20), (20, 40), (40, 1000)):
        for slo, shi in ((0, 1), (1, 100)):
            subset = [r for r in records if lo <= r[9] < hi and slo <= r[3] < shi]
            report['by_range'].append({'range_m': [lo, hi], 'gt_speed_range': [slo, shi], **block(subset)})
    # Confidence deciles among moving objects only (GT >= 1 m/s): does confidence
    # predict error once the static/moving composition is removed?
    moving = [r for r in records if r[3] >= 1.0]
    medges = np.quantile(np.array([r[0] for r in moving]), np.linspace(0, 1, 6))
    report['by_confidence_quintile_moving'] = []
    for i in range(5):
        lo, hi = medges[i], medges[i + 1]
        subset = [r for r in moving if (lo <= r[0] < hi) or (i == 4 and r[0] == hi)]
        report['by_confidence_quintile_moving'].append({'quintile': i + 1, 'conf_range': [float(lo), float(hi)], **block(subset)})
    # Track age: keyframes the baseline track had already been observed before this one.
    report['by_track_age'] = []
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 6), (6, 10 ** 6)):
        for slo, shi in ((0, 1), (1, 3), (3, 100)):
            subset = [r for r in records if lo <= r[8] < hi and slo <= r[3] < shi]
            report['by_track_age'].append({'age_keyframes': [lo, hi], 'gt_speed_range': [slo, shi], **block(subset)})
    # Class x track age over ALL speeds (the policy is applied to every new track whether or not it
    # moves): mean |error| of the grid and of the tracker's CV estimate, and their difference.
    # Preregistered rule (PREREGISTRATION.md, 2026-09-24): a class with mean(tracker - grid) <= 0
    # over ages 0-2 does not use the grid velocity.
    report['by_class_track_age'] = []
    for name in sorted(set(r[2] for r in records)):
        for lo, hi in ((0, 1), (1, 2), (2, 3), (0, 3), (3, 10 ** 6)):
            subset = [r for r in records if r[2] == name and lo <= r[8] < hi and r[6] is not None]
            g = [r[4] for r in subset]
            tr = [r[6] for r in subset]
            report['by_class_track_age'].append({
                'class': name, 'age_keyframes': [lo, hi], 'n': len(subset),
                'grid_mean_abs': float(np.mean(g)) if g else None, 'tracker_mean_abs': float(np.mean(tr)) if tr else None,
                'grid_median_abs': float(np.median(g)) if g else None, 'tracker_median_abs': float(np.median(tr)) if tr else None,
                'mean_tracker_minus_grid': float(np.mean(tr) - np.mean(g)) if g else None,
                'median_gt_speed': float(np.median([r[3] for r in subset])) if subset else None})
    report['by_tracker_history'] = []
    for label, pick in (('cold_start', lambda r: r[7]), ('warm', lambda r: not r[7])):
        for lo, hi in ((0, 1), (1, 3), (3, 100)):
            subset = [r for r in records if pick(r) and lo <= r[3] < hi]
            report['by_tracker_history'].append({'tracker_state': label, 'gt_speed_range': [lo, hi], **block(subset)})
    (ROOT / args.out).write_text(json.dumps(report, indent=2) + '\n')
    o = report['overall']
    print(f"matched {report['matched_detections']}: grid median |e| {o['median_abs_error_mps']:.3f} m/s, "
          f"P(<1) {o['p_within_1mps']:.3f}, NEES {o['mean_nees']:.2f}; tracker CV median |e| "
          f"{o['tracker_cv_median_abs_error_mps']:.3f}, P(<1) {o['tracker_cv_p_within_1mps']:.3f}; "
          f"zero-velocity median |e| {o['zero_velocity_median_abs_error_mps']:.3f}")
    for row in report['by_confidence_decile']:
        print(f"  conf decile {row['decile']:>2} [{row['conf_range'][0]:.3f}, {row['conf_range'][1]:.3f}] n={row['n']:>6} "
              f"grid median|e| {row['median_abs_error_mps']:.3f} P(<1) {row['p_within_1mps']:.3f} NEES {row['mean_nees']:.2f} | "
              f"tracker CV P(<1) {row['tracker_cv_p_within_1mps']:.3f}")
    for row in report['by_range']:
        print(f"  range {str(row['range_m']):<12} gt speed {str(row['gt_speed_range']):<9} n={row['n']:>6} grid median|e| {row['median_abs_error_mps']:.3f} "
              f"P(<1) {row['p_within_1mps']:.3f} NEES {row['mean_nees']:.2f} | CV P(<1) {row['tracker_cv_p_within_1mps']:.3f}")
    for row in report['by_confidence_quintile_moving']:
        print(f"  moving conf quintile {row['quintile']} [{row['conf_range'][0]:.3f}, {row['conf_range'][1]:.3f}] n={row['n']:>5} grid median|e| {row['median_abs_error_mps']:.3f} "
              f"P(<1) {row['p_within_1mps']:.3f} NEES {row['mean_nees']:.2f} | CV P(<1) {row['tracker_cv_p_within_1mps']:.3f}")
    for row in report['by_track_age']:
        if row['n'] == 0: continue
        print(f"  age {str(row['age_keyframes']):<12} gt speed {str(row['gt_speed_range']):<9} n={row['n']:>6} grid median|e| {row['median_abs_error_mps']:.3f} "
              f"P(<1) {row['p_within_1mps']:.3f} | tracker CV median|e| {row['tracker_cv_median_abs_error_mps']:.3f} P(<1) {row['tracker_cv_p_within_1mps']:.3f}")
    for row in report['by_tracker_history']:
        print(f"  {row['tracker_state']:<10} gt speed {row['gt_speed_range']} n={row['n']:>6} grid median|e| {row['median_abs_error_mps']:.3f} "
              f"P(<1) {row['p_within_1mps']:.3f} | tracker CV median|e| {row['tracker_cv_median_abs_error_mps']:.3f} P(<1) {row['tracker_cv_p_within_1mps']:.3f} "
              f"| zero-velocity median|e| {row['zero_velocity_median_abs_error_mps']:.3f}")
    for row in report['by_gt_speed']:
        print(f"  gt speed {row['gt_speed_range']} n={row['n']:>6} grid median|e| {row['median_abs_error_mps']:.3f} "
              f"P(<1) {row['p_within_1mps']:.3f} | tracker CV P(<1) {row['tracker_cv_p_within_1mps']:.3f}")


if __name__ == '__main__':
    main()
