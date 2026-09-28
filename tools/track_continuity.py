#!/usr/bin/env python3
"""Identity continuity of a tracker output, on the population a planner sees.

Ground-truth objects that the LiDAR actually saw (num_lidar_pts > 0) are matched
one-to-one to submission boxes of the same class within 2 m -- the same rule the
prediction diagnostic uses -- and for each ground-truth object we count how many
distinct tracking ids its matched frames carry. Reported per ground-truth speed
band, because a policy that only touches young tracks cannot move an average
dominated by parked cars.

  ids_per_object   mean number of distinct ids a ground-truth object receives
  one_id_pct       share of objects that keep a single id throughout
  ready_1s_pct     share of matched object-frames whose track already carries
                   1 s of unbroken history (gap <= 0.75 s), i.e. what a
                   downstream predictor can actually use
  tracks           distinct tracking ids in the submission; a policy that
                   lowered identity switches by emitting fewer tracks would
                   show up here

Ground truth is used for scoring and for the speed bands only.
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

BANDS = ((0.0, 'all'), (1.0, 'fast_1'), (3.0, 'fast_3'))
HISTORY_WINDOW = 1.0
MAX_GAP = 0.75


def resolve(name, paths):
    for candidate in (Path(paths['original']) / 'work/results' / name / 'tracking_results.json',
                      ROOT / 'results/tracker_runs' / name / 'tracking_results.json',
                      ROOT / 'results/hybrid' / name / 'tracking_results.json',
                      ROOT / 'results/polymot' / name / 'results.json'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, action='append',
                        help='submission to measure; repeat for several')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    metadata = Path(paths['nuscenes']) / 'v1.0-trainval'
    scene_names = {s['token']: s['name'] for s in json.loads((metadata / 'scene.json').read_text())}
    samples = json.loads((metadata / 'sample.json').read_text())

    report = {'runs': {}, 'definition': {
        'match': 'one-to-one minimum distance, same class, <= 2 m, GT num_lidar_pts > 0',
        'history_window_s': HISTORY_WINDOW, 'max_gap_s': MAX_GAP,
        'note': 'ready_1s counts matched object-frames whose track already carries '
                'a full history window of the same identity.'}}

    for name in args.run:
        path = resolve(name, paths)
        submission = json.loads(path.read_text())['results']
        frames_of_scene = defaultdict(list)
        for sample in samples:
            if sample['token'] in submission:
                frames_of_scene[scene_names[sample['scene_token']]].append(
                    (sample['timestamp'] * 1e-6, sample['token']))
        tracks = set()
        ids_of_object = defaultdict(set)
        speed_of_object = {}
        frames_by_band = {label: [0, 0] for _, label in BANDS}   # [matched frames, ready frames]
        objects_by_band = {label: set() for _, label in BANDS}
        for scene in sorted(frames_of_scene):
            cached = json.loads((Path(paths['gt_cache']) / f'{scene}.json').read_text())['frames']
            history = defaultdict(list)     # tracking id -> [timestamps]
            for timestamp, token in sorted(frames_of_scene[scene]):
                boxes = submission[token]
                for box in boxes:
                    tracks.add((scene, box['tracking_id']))
                    stamps = history[box['tracking_id']]
                    if stamps and timestamp - stamps[-1] > MAX_GAP:
                        stamps.clear()
                    stamps.append(timestamp)
                if token not in cached:
                    continue
                gt = [g for g in cached[token]['objects'] if g['num_lidar_pts'] > 0]
                pairs = diagnostic.match_one_to_one(
                    [{'cls': g['cls'], 'translation': g['translation']} for g in gt],
                    [{'tracking_name': b['tracking_name'], 'translation': b['translation']} for b in boxes])
                for gi, bi in pairs:
                    obj, box = gt[gi], boxes[bi]
                    key = (scene, obj['instance_token'])
                    ids_of_object[key].add(box['tracking_id'])
                    speed = (float(np.linalg.norm(obj['velocity'][:2]))
                             if obj.get('velocity') is not None else 0.0)
                    speed_of_object[key] = max(speed_of_object.get(key, 0.0), speed)
                    stamps = history[box['tracking_id']]
                    ready = bool(stamps) and (timestamp - stamps[0]) >= HISTORY_WINDOW
                    for threshold, label in BANDS:
                        if speed >= threshold:
                            frames_by_band[label][0] += 1
                            frames_by_band[label][1] += int(ready)
                            objects_by_band[label].add(key)
        entry = {'submission': str(path), 'tracks': len(tracks), 'bands': {}}
        for _, label in BANDS:
            keys = objects_by_band[label]
            counts = np.array([len(ids_of_object[k]) for k in keys]) if keys else np.zeros(0)
            matched, ready = frames_by_band[label]
            entry['bands'][label] = {
                'objects': int(counts.size),
                'matched_object_frames': matched,
                'ids_per_object': float(counts.mean()) if counts.size else None,
                'one_id_pct': float(100 * (counts == 1).mean()) if counts.size else None,
                'ready_1s_pct': float(100 * ready / matched) if matched else None}
        report['runs'][name] = entry
        band = entry['bands']
        print(f"{name:<22} tracks {entry['tracks']:>6} | "
              + ' | '.join(f"{lab}: ids/obj {band[lab]['ids_per_object']:.3f} "
                           f"one_id {band[lab]['one_id_pct']:.1f}% ready1s {band[lab]['ready_1s_pct']:.2f}%"
                           for _, lab in BANDS), flush=True)

    out = Path(args.out) if args.out else ROOT / 'results/track_continuity.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
