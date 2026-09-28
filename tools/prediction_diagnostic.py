#!/usr/bin/env python3
"""Independent causal CV diagnostic; NOT a reproduction of the missing paper evaluator.

Uses a common one-to-one geometric match, visible cached GT, all submitted boxes,
and exact future horizons through GT interpolation. No official tracking score
threshold, class-range or bicycle-rack filtering is applied. Predictions use only
current/past submission data; GT identities and future positions are evaluation-only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
RUNS = ('n150_mg_none', 'n150_mg_flow')
MODES = ('history_cv', 'published_velocity_cv', 'history_cv_velocity_fallback')
BANDS = (0, 1, 3)
HORIZONS = (1, 3)


def geometry_key(box):
    return (box['tracking_name'], *[round(float(v), 8)
            for field in ('translation', 'size', 'rotation') for v in box[field]])


def common_boxes(a, b):
    # Reject ambiguous coincident duplicate boxes instead of selecting using IDs.
    a = sorted(a, key=geometry_key)
    b = sorted(b, key=geometry_key)
    ka, kb = [geometry_key(x) for x in a], [geometry_key(x) for x in b]
    if ka != kb:
        raise ValueError('The two submissions do not have identical geometry.')
    if len(set(ka)) != len(ka):
        raise ValueError('Duplicate geometry: an unambiguous shared match is required.')
    return a, b


def match_one_to_one(gt, boxes, limit=2.0):
    pairs = []
    for cls in sorted({o['cls'] for o in gt}):
        gi = [i for i, o in enumerate(gt) if o['cls'] == cls]
        bi = [i for i, o in enumerate(boxes) if o['tracking_name'] == cls]
        if not gi or not bi:
            continue
        gxy = np.array([gt[i]['translation'][:2] for i in gi])
        bxy = np.array([boxes[i]['translation'][:2] for i in bi])
        distances = np.linalg.norm(gxy[:, None] - bxy[None, :], axis=2)
        cost = np.where(distances <= limit, distances, 1e9)
        rows, cols = linear_sum_assignment(cost)
        pairs.extend((gi[r], bi[c]) for r, c in zip(rows, cols) if distances[r, c] <= limit)
    return pairs


class CausalCV:
    def __init__(self, window=1.0, max_gap=0.75):
        self.window, self.max_gap = window, max_gap
        self.history = {}
        self.last_time = -math.inf

    def observe(self, timestamp, boxes):
        if timestamp <= self.last_time:
            raise ValueError('Frames must be strictly chronological.')
        self.last_time = timestamp
        velocities = np.zeros((len(boxes), len(MODES), 2))
        cold = np.ones(len(boxes), dtype=bool)
        ids = [box['tracking_id'] for box in boxes]
        if len(ids) != len(set(ids)):
            raise ValueError('A submission uses the same ID twice in a frame.')
        for i, box in enumerate(boxes):
            xy = np.asarray(box['translation'][:2], dtype=float)
            published = np.asarray(box['velocity'][:2], dtype=float)
            if not np.all(np.isfinite(published)):
                raise ValueError('Non-finite published velocity.')
            key = (box['tracking_name'], box['tracking_id'])
            history = self.history.get(key, [])
            if history and timestamp - history[-1][0] > self.max_gap:
                history = []
            history = [p for p in history if timestamp - p[0] <= self.window + 1e-3]
            history.append((timestamp, xy))
            velocity = np.zeros(2)
            if len(history) >= 2:
                ts = np.array([p[0] - timestamp for p in history])
                ts -= ts.mean()
                points = np.array([p[1] for p in history])
                velocity = (ts[:, None] * (points - points.mean(axis=0))).sum(axis=0) / (ts @ ts)
                cold[i] = False
            self.history[key] = history
            velocities[i] = [velocity, published, published if cold[i] else velocity]
        return velocities, cold


def future_positions(series, timestamp, horizon):
    times, xy = series
    targets = timestamp + np.arange(0.5, horizon + 0.01, 0.5)
    if targets[-1] > times[-1] + 1e-6:
        return None
    # Reject missing annotation intervals, including the interval after the origin.
    first = max(0, np.searchsorted(times, timestamp, side='right') - 1)
    last = min(len(times) - 1, np.searchsorted(times, targets[-1], side='left'))
    if np.any(np.diff(times[first:last + 1]) > 0.75):
        return None
    return np.column_stack([np.interp(targets, times, xy[:, axis]) for axis in range(2)])


def load_submission(path):
    print('Loading ' + str(path), flush=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    result = json.loads(raw)['results']
    return result, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/prediction_diagnostic')
    parser.add_argument('--bootstrap', type=int, default=4000)
    parser.add_argument('--base', default=RUNS[0], help='baseline run (inherited, tracker_runs or hybrid)')
    parser.add_argument('--compare', default=RUNS[1], help='run compared against the baseline')
    parser.add_argument('--match', choices=['geometry', 'gt'], default='geometry',
                        help='geometry: the two runs must report identical boxes, paired by geometry; '
                             'gt: each run is matched to ground truth on its own and the comparison '
                             'is restricted to GT object-frames matched in both runs')
    parser.add_argument('--gt-cache', default=None,
                        help='GT cache directory other than the nuScenes one (e.g. results/kitti/<data>/gt); '
                             'scenes and timestamps are then read from it instead of the nuScenes tables')
    args = parser.parse_args()
    runs = (args.base, args.compare)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    submissions, hashes = [], {}
    for name in runs:
        candidates = [Path(paths['original']) / 'work/results' / name / 'tracking_results.json',
                      ROOT / 'results/tracker_runs' / name / 'tracking_results.json',
                      ROOT / 'results/hybrid' / name / 'tracking_results.json',
                      ROOT / 'results/polymot' / name / 'results.json',
                      ROOT / 'results/kitti/runs' / name / 'tracking_results.json']
        source = next((c for c in candidates if c.is_file()), None)
        if source is None:
            raise FileNotFoundError(name)
        submission, hashes[name] = load_submission(source)
        submissions.append(submission)
    if submissions[0].keys() != submissions[1].keys():
        raise ValueError('Submissions cover different samples.')
    scene_frames = defaultdict(list)
    if args.gt_cache:
        paths['gt_cache'] = str((ROOT / args.gt_cache) if not Path(args.gt_cache).is_absolute() else args.gt_cache)
        for gt_file in sorted(Path(paths['gt_cache']).glob('*.json')):
            if gt_file.name == 'manifest.json':
                continue
            for token, frame in json.loads(gt_file.read_text())['frames'].items():
                if token in submissions[0]:
                    scene_frames[gt_file.stem].append((frame['timestamp_s'], token))
    else:
        metadata = Path(paths['nuscenes']) / 'v1.0-trainval'
        scene_names = {s['token']: s['name'] for s in json.loads((metadata / 'scene.json').read_text())}
        for sample in json.loads((metadata / 'sample.json').read_text()):
            if sample['token'] in submissions[0]:
                scene_frames[scene_names[sample['scene_token']]].append(
                    (sample['timestamp'] * 1e-6, sample['token']))
    scenes = sorted(scene_frames)
    # dimensions: scene, run, predictor, speed band, horizon, [ADE sum, FDE sum, n, cold]
    stats = np.zeros((len(scenes), 2, len(MODES), len(BANDS), len(HORIZONS), 4))
    # Common-origin history availability: 0 both warm, 1 only baseline warm,
    # 2 only flow warm, 3 both cold. This is diagnostic stratification, not a causal intervention.
    mechanism = np.zeros((len(scenes), 4, 2, len(MODES), len(BANDS), len(HORIZONS), 4))
    visible_gt = matched = total_frames = empty_gt_frames = 0
    matched_per_run = [0, 0]
    for si, scene in enumerate(scenes):
        cached = json.loads((Path(paths['gt_cache']) / (scene + '.json')).read_text())['frames']
        gt_series = defaultdict(list)
        for frame in cached.values():
            for obj in frame['objects']:
                gt_series[obj['instance_token']].append((frame['timestamp_s'], obj['translation'][:2]))
        series = {}
        for key, values in gt_series.items():
            values.sort(key=lambda p: p[0])
            series[key] = (np.array([p[0] for p in values]), np.array([p[1] for p in values]))
        predictors = [CausalCV(), CausalCV()]
        for timestamp, token in sorted(scene_frames[scene]):
            total_frames += 1
            if args.match == 'geometry':
                boxes = common_boxes(submissions[0][token], submissions[1][token])
            else:
                boxes = [submissions[0][token], submissions[1][token]]
            observed = [predictors[r].observe(timestamp, boxes[r]) for r in range(2)]
            if token not in cached:
                empty_gt_frames += 1
            gt = [g for g in cached.get(token, {'objects': []})['objects'] if g['num_lidar_pts'] > 0]
            visible_gt += len(gt)
            if args.match == 'geometry':
                pairs = match_one_to_one(gt, boxes[0])
                index_of = [dict(pairs), dict(pairs)]
            else:
                index_of = [dict(match_one_to_one(gt, boxes[r])) for r in range(2)]
                pairs = [(gi, index_of[0][gi]) for gi in index_of[0] if gi in index_of[1]]
            for r in range(2):
                matched_per_run[r] += len(index_of[r])
            matched += len(pairs)
            for gi, _ in pairs:
                obj = gt[gi]
                bis = [index_of[0][gi], index_of[1][gi]]
                stratum = 2 * int(observed[0][1][bis[0]]) + int(observed[1][1][bis[1]])
                speed = np.linalg.norm(obj['velocity']) if obj['velocity'] is not None else math.nan
                for hi, horizon in enumerate(HORIZONS):
                    truth = future_positions(series[obj['instance_token']], timestamp, horizon)
                    if truth is None:
                        continue
                    offsets = np.arange(0.5, horizon + 0.01, 0.5)
                    for ri, (velocities, cold) in enumerate(observed):
                        bi = bis[ri]
                        position = np.asarray(boxes[ri][bi]['translation'][:2])
                        prediction = position + velocities[bi, :, None, :] * offsets[None, :, None]
                        error = np.linalg.norm(prediction - truth[None, :, :], axis=2)
                        for band_i, band in enumerate(BANDS):
                            if band and not speed >= band:
                                continue
                            stats[si, ri, :, band_i, hi, 0] += error.mean(axis=1)
                            stats[si, ri, :, band_i, hi, 1] += error[:, -1]
                            stats[si, ri, :, band_i, hi, 2] += 1
                            stats[si, ri, :, band_i, hi, 3] += cold[bi]
                            mechanism[si, stratum, ri, :, band_i, hi, 0] += error.mean(axis=1)
                            mechanism[si, stratum, ri, :, band_i, hi, 1] += error[:, -1]
                            mechanism[si, stratum, ri, :, band_i, hi, 2] += 1
                            mechanism[si, stratum, ri, :, band_i, hi, 3] += cold[bi]
        if (si + 1) % 10 == 0:
            print('Evaluated %d / %d scenes' % (si + 1, len(scenes)), flush=True)
    seed = 20260922
    weights = np.random.default_rng(seed).multinomial(len(scenes), np.ones(len(scenes)) / len(scenes),
                                                    size=args.bootstrap)
    rows = []
    for mi, mode in enumerate(MODES):
        for band_i, band in enumerate(BANDS):
            for hi, horizon in enumerate(HORIZONS):
                cell = stats[:, :, mi, band_i, hi, :]
                summed = cell.sum(axis=0)
                if not np.array_equal(cell[:, 0, 2], cell[:, 1, 2]):
                    raise AssertionError('Unpaired prediction samples.')
                row = {'predictor': mode, 'speed_min_mps': band, 'horizon_s': horizon,
                       'n': int(summed[0, 2]), 'results': {}, 'flow_minus_none': {}}
                for ri, run in enumerate(runs):
                    row['results'][run] = dict(ade=summed[ri, 0] / summed[ri, 2],
                                              fde=summed[ri, 1] / summed[ri, 2],
                                              cold_pct=100 * summed[ri, 3] / summed[ri, 2])
                for ei, metric in enumerate(('ade', 'fde')):
                    delta = summed[1, ei] / summed[1, 2] - summed[0, ei] / summed[0, 2]
                    denominator = weights @ cell[:, 0, 2]
                    valid = denominator > 0
                    boot = (weights[valid] @ (cell[:, 1, ei] - cell[:, 0, ei])) / denominator[valid]
                    row['flow_minus_none'][metric] = {'delta': delta,
                        'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
                        'bootstrap_fraction_below_zero': float(np.mean(boot < 0))}
                rows.append(row)
    output = {'diagnostic_not_manuscript_reproduction': True,
              'protocol': __doc__, 'history_window_s': 1.0, 'max_history_gap_s': 0.75,
              'geometry_round_decimals': 8, 'match': 'one-to-one minimum distance, same class, <=2m',
              'future': 'linear GT interpolation at 0.5s intervals; annotation gaps >0.75s excluded',
              'scenes': scenes, 'frames': total_frames, 'frames_without_cached_gt': empty_gt_frames,
              'visible_gt_object_frames': visible_gt, 'matched_object_frames': matched,
              'match_mode': args.match, 'matched_object_frames_per_run': dict(zip(runs, matched_per_run)),
              'submission_sha256': hashes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bootstrap_replicates': args.bootstrap, 'bootstrap_seed': seed, 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    np.savez_compressed(out / 'scene_stats.npz', stats=stats, scenes=np.asarray(scenes))
    if not np.allclose(mechanism.sum(axis=1), stats, rtol=1e-12, atol=1e-9):
        raise AssertionError('Mechanism strata do not partition the evaluation samples.')
    np.savez_compressed(out / 'history_strata.npz', stats=mechanism, scenes=np.asarray(scenes),
                        strata=np.asarray(['both_warm', 'only_baseline_warm', 'only_flow_warm', 'both_cold']))
    (out / 'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print('Completed: ' + str(out / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
