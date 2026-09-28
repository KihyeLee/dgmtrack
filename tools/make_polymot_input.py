#!/usr/bin/env python3
"""Write the MEGVII detection cache as a Poly-MOT detection file.

Poly-MOT reads a nuScenes detection result file whose sample tokens are in
chronological order inside each scene, and starts a new sequence at every token
listed in its first-token table. This tool writes that file from the keyframe
rows of a detection cache, optionally attaching the per-detection grid velocity
recorded by the restored tracker (`--report dogm`) as the detection velocity.

Variants:
  --velocity none   velocity = [0, 0] for every box; use with has_velo: false
  --velocity grid   velocity = grid velocity where the grid marked it valid,
                    NaN otherwise; needs the masked-update patch in Poly-MOT

Boxes are paired with the grid report by keyframe order and checked by geometry,
so the two files must come from the same detection cache.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def same_geometry(a, b):
    return (a['detection_name'] == b['detection_name']
            and all(abs(x - y) < 1e-6 for x, y in zip(a['translation'], b['translation']))
            and all(abs(x - y) < 1e-6 for x, y in zip(a['size'], b['size'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detections', default='det150_megvii',
                        help='cache name under the original cache tree, or an absolute directory')
    parser.add_argument('--velocity', choices=['none', 'grid', 'detector'], required=True,
                        help="none: zeros (has_velo false); grid: grid velocity where valid, NaN "
                             "otherwise; detector: the detector's own velocity")
    parser.add_argument('--grid-run', default='mg_grid_report',
                        help='tracker run under results/tracker_runs written with --report dogm')
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenes', default=None,
                        help='scene list file (default: configs/val150_scenes.txt)')
    parser.add_argument('--grid-dir', default=None,
                        help='absolute scenes/ dir of the --report dogm run, for splits whose runs '
                             'do not live under results/tracker_runs')
    args = parser.parse_args()

    import workspace_paths

    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    det_dir = (Path(args.detections) if Path(args.detections).is_absolute() or args.detections.startswith('/')
               else Path(paths['original']) / 'cache' / args.detections)
    grid_dir = (Path(args.grid_dir) if args.grid_dir
                else ROOT / 'results/tracker_runs' / args.grid_run / 'scenes')
    scene_file = Path(args.scenes) if args.scenes else ROOT / 'configs/val150_scenes.txt'
    scenes = [s for s in scene_file.read_text().replace(',', ' ').split() if s]

    results = {}
    counts = {'keyframes': 0, 'boxes': 0, 'grid_valid': 0, 'grid_with_covariance': 0,
              'null_velocity_in_cache': 0, 'detector_velocity': 0, 'no_velocity_in_cache': 0}
    for scene in scenes:
        grid_rows = {}
        if args.velocity == 'grid':
            candidates = list(grid_dir.glob(f'*/{scene}.jsonl')) + [grid_dir / scene / f'{scene}.jsonl']
            grid_file = next((c for c in candidates if c.is_file()), None)
            if grid_file is None:
                raise FileNotFoundError(f'no grid report for {scene} under {grid_dir}')
            with grid_file.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get('is_key_frame'):
                        grid_rows[row['sample_token']] = row['boxes']
        with (det_dir / f'{scene}.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if not row.get('is_key_frame'):
                    continue
                counts['keyframes'] += 1
                grid_boxes = grid_rows.get(row['sample_token']) if args.velocity == 'grid' else None
                if grid_boxes is not None and len(grid_boxes) != len(row['boxes']):
                    raise ValueError('Box count differs from the grid report in ' + row['sample_token'])
                out_boxes = []
                for index, box in enumerate(row['boxes']):
                    counts['boxes'] += 1
                    if box.get('velocity') is None:
                        counts['null_velocity_in_cache'] += 1
                    velocity = [0.0, 0.0]
                    if args.velocity == 'detector':
                        detector_velocity = box.get('velocity')
                        if detector_velocity is None:
                            # An oracle cache leaves unmatched detections without a
                            # velocity; Poly-MOT then skips the velocity rows for them.
                            velocity = [math.nan, math.nan]
                            counts['no_velocity_in_cache'] += 1
                        else:
                            velocity = [float(detector_velocity[0]), float(detector_velocity[1])]
                            counts['detector_velocity'] += 1
                    if grid_boxes is not None:
                        g = grid_boxes[index]
                        if not same_geometry(box, g):
                            raise ValueError('Geometry differs from the grid report in ' + row['sample_token'])
                        extra = {}
                        if g['dogm_valid']:
                            velocity = [float(g['velocity'][0]), float(g['velocity'][1])]
                            counts['grid_valid'] += 1
                            if 'dogm_cov' in g:
                                extra = {'grid_cov': [float(v) for v in g['dogm_cov']],
                                         'grid_conf': float(g['dogm_conf']),
                                         'grid_cells': int(g.get('dogm_cells', 0))}
                                counts['grid_with_covariance'] += 1
                        else:
                            velocity = [math.nan, math.nan]
                    else:
                        extra = {}
                    out_boxes.append({
                        'sample_token': row['sample_token'],
                        'translation': box['translation'],
                        'size': box['size'],
                        'rotation': box['rotation'],
                        'velocity': velocity,
                        'detection_name': box['detection_name'],
                        'detection_score': box['detection_score'],
                        'attribute_name': box.get('attribute_name', ''),
                        **extra,
                    })
                results[row['sample_token']] = out_boxes

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'meta': {'use_camera': False, 'use_lidar': True, 'use_radar': False,
                        'use_map': False, 'use_external': False,
                        'source_detections': str(det_dir), 'velocity_variant': args.velocity,
                        'grid_run': args.grid_run if args.velocity == 'grid' else None,
                        'created_utc': datetime.now(timezone.utc).isoformat()},
               'results': results}
    out.write_text(json.dumps(payload))
    print(json.dumps({'out': str(out), **counts}))


if __name__ == '__main__':
    main()
