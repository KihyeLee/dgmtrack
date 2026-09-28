#!/usr/bin/env python3
"""Run the vendored Poly-MOT on a detection file and save the submission.

This is test.py's tracking loop without its trailing evaluation, so the
result can be scored by `tools/evaluate_saved_official.py` with provenance.
Workers split the sequences by `seq_id % process` exactly as test.py does.

Output boxes whose velocity is NaN (a track written straight from a detection
that carried no velocity) are submitted with velocity [0, 0]; the nuScenes
tracking metrics do not read velocity, and the count is recorded.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLYMOT = ROOT / 'vendor/Poly-MOT'
START_CWD = Path.cwd()
sys.path.insert(0, str(POLYMOT))
os.chdir(POLYMOT)  # Poly-MOT's modules import by relative package name

from dataloader.nusc_loader import NuScenesloader  # noqa: E402
from tracking.nusc_tracker import Tracker  # noqa: E402
from motion_module import kalman_filter  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def track(loader, token, process, out_path):
    tracker = Tracker(config=loader.config)
    results = {}
    nan_velocity = 0
    for frame in loader:
        if process > 1 and frame['seq_id'] % process != token:
            continue
        tracker.tracking(frame)
        boxes = []
        if 'no_val_track_result' not in frame:
            for box in frame['box_track_res']:
                velocity = [float(box.velocity[0]), float(box.velocity[1])]
                if any(math.isnan(v) for v in velocity):
                    velocity = [0.0, 0.0]
                    nan_velocity += 1
                boxes.append({
                    'sample_token': frame['sample_token'],
                    'translation': [float(v) for v in box.center],
                    'size': [float(v) for v in box.wlh],
                    'rotation': [float(v) for v in box.orientation],
                    'velocity': velocity,
                    'tracking_id': str(box.tracking_id),
                    'tracking_name': box.name,
                    'tracking_score': float(box.score),
                })
        results.setdefault(frame['sample_token'], []).extend(boxes)
    for token_key, boxes in results.items():
        boxes.sort(key=lambda b: -b['tracking_score'])
        results[token_key] = boxes[:500]
    Path(out_path).write_text(json.dumps({'results': results, 'nan_velocity_boxes': nan_velocity,
                                          'velocity_update_stats': dict(kalman_filter.VELO_STATS)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detection', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--process', type=int, default=6)
    parser.add_argument('--first-token', default=str(ROOT / 'results/polymot/input/nusc_first_token.json'))
    args = parser.parse_args()

    detection = (START_CWD / args.detection).resolve()
    config_path = (START_CWD / args.config).resolve()
    first_token = str((START_CWD / args.first_token).resolve())
    out_dir = ROOT / 'results/polymot' / args.label
    if out_dir.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(out_dir))
    out_dir.mkdir(parents=True)
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    loader = NuScenesloader(str(detection), first_token, config)
    start = time.monotonic()

    temp = out_dir / 'temp'
    temp.mkdir()
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            pool.apply_async(track, args=(loader, token, args.process, str(temp / f'{token}.json')))
        pool.close()
        pool.join()
    else:
        track(loader, 0, 1, str(temp / '0.json'))

    merged, nan_velocity = {}, 0
    velocity_stats = {}
    for token in range(args.process):
        part = json.loads((temp / f'{token}.json').read_text())
        nan_velocity += part['nan_velocity_boxes']
        for key, value in part.get('velocity_update_stats', {}).items():
            velocity_stats[key] = velocity_stats.get(key, 0) + value
        for key, boxes in part['results'].items():
            merged.setdefault(key, []).extend(boxes)
    submission = {'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False,
                           'use_map': False, 'use_external': False},
                  'results': merged}
    (out_dir / 'results.json').write_text(json.dumps(submission))
    provenance = {
        'label': args.label,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'detection': str(detection), 'detection_sha256': sha256(detection),
        'config': config, 'config_sha256': sha256(config_path),
        'first_token_table': first_token,
        'process': args.process,
        'samples': len(merged), 'boxes': sum(len(v) for v in merged.values()),
        'nan_velocity_boxes_set_to_zero': nan_velocity,
        'velocity_update_stats': velocity_stats,
        'wall_time_seconds': time.monotonic() - start,
        'polymot_files': {name: sha256(POLYMOT / name) for name in
                          ('motion_module/kalman_filter.py', 'motion_module/motion_model.py',
                           'tracking/nusc_tracker.py', 'dataloader/nusc_loader.py')},
        'submission_sha256': sha256(out_dir / 'results.json'),
    }
    (out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: provenance[k] for k in ('label', 'samples', 'boxes',
                                                 'nan_velocity_boxes_set_to_zero',
                                                 'velocity_update_stats', 'wall_time_seconds')}))


if __name__ == '__main__':
    main()
