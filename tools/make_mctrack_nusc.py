#!/usr/bin/env python3
"""Write a nuScenes detection file for MCTrack, optionally carrying the grid velocity.

MCTrack's nuScenes path reads a standard nuScenes detection result file and
converts it to its BaseVersion (preprocess/convert_nuscenes.py), which copies
`velocity` into `global_velocity`. Three conditions are produced from one
detection cache, so nothing but the velocity information differs between them:

  --velocity detector  the detector's own velocity (reference row)
  --velocity none      [0, 0]; the velocity-removed condition
  --velocity grid      [0, 0] plus grid_valid / grid_velocity / grid_cov /
                       grid_conf per box, read from a `--report dogm` run;
                       MCTrack uses them only when MCTRACK_GRID=1

Boxes are paired with the grid report by keyframe order and checked by geometry,
the same rule tools/make_polymot_input.py uses, so a mismatch fails loudly
rather than silently shifting velocities onto the wrong boxes.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def same_geometry(a, b):
    return (a['detection_name'] == b['detection_name']
            and all(abs(x - y) < 1e-6 for x, y in zip(a['translation'], b['translation']))
            and all(abs(x - y) < 1e-6 for x, y in zip(a['size'], b['size'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detections', required=True, help='directory of <scene>.jsonl')
    ap.add_argument('--velocity', choices=['detector', 'none', 'grid'], required=True)
    ap.add_argument('--grid-dir', default=None, help='scenes/ of a --report dogm run')
    ap.add_argument('--scenes', default=str(ROOT / 'configs/val150_scenes.txt'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    det_dir = Path(args.detections)
    grid_dir = Path(args.grid_dir) if args.grid_dir else None
    if args.velocity == 'grid' and grid_dir is None:
        ap.error('--velocity grid needs --grid-dir')
    scenes = [s for s in Path(args.scenes).read_text().replace(',', ' ').split() if s]

    results = {}
    n = {'keyframes': 0, 'boxes': 0, 'grid_valid': 0}
    for scene in scenes:
        grid_rows = {}
        if args.velocity == 'grid':
            cands = list(grid_dir.glob('*/%s.jsonl' % scene)) + [grid_dir / scene / ('%s.jsonl' % scene)]
            gf = next((c for c in cands if c.is_file()), None)
            if gf is None:
                raise FileNotFoundError('no grid report for %s under %s' % (scene, grid_dir))
            for line in gf.open():
                row = json.loads(line)
                if row.get('is_key_frame'):
                    grid_rows[row['sample_token']] = row['boxes']

        with (det_dir / ('%s.jsonl' % scene)).open() as stream:
            for line in stream:
                row = json.loads(line)
                if not row.get('is_key_frame'):
                    continue
                n['keyframes'] += 1
                gboxes = grid_rows.get(row['sample_token']) if args.velocity == 'grid' else None
                if gboxes is not None and len(gboxes) != len(row['boxes']):
                    raise ValueError('grid report box count differs in ' + row['sample_token'])
                out = []
                for i, box in enumerate(row['boxes']):
                    n['boxes'] += 1
                    vel = box.get('velocity') or [0.0, 0.0]
                    rec = {'sample_token': row['sample_token'],
                           'translation': box['translation'], 'size': box['size'],
                           'rotation': box['rotation'],
                           'velocity': list(vel) if args.velocity == 'detector' else [0.0, 0.0],
                           'detection_name': box['detection_name'],
                           'detection_score': box['detection_score'],
                           'attribute_name': box.get('attribute_name', '')}
                    if gboxes is not None:
                        g = gboxes[i]
                        if not same_geometry(box, g):
                            raise ValueError('grid report box %d does not match in %s'
                                             % (i, row['sample_token']))
                        # the grid report writes the grid velocity in 'velocity';
                        # 'dogm_*' carry the validity, covariance and confidence
                        valid = bool(g.get('dogm_valid'))
                        rec['grid_valid'] = valid
                        rec['grid_velocity'] = ([float(g['velocity'][0]), float(g['velocity'][1])]
                                                if valid and g.get('velocity') else [0.0, 0.0])
                        rec['grid_cov'] = [float(v) for v in (g.get('dogm_cov') or [0.0, 0.0, 0.0])]
                        rec['grid_conf'] = float(g.get('dogm_conf') or 0.0)
                        n['grid_valid'] += int(valid)
                    out.append(rec)
                results.setdefault(row['sample_token'], []).extend(out)

    meta = {'use_camera': False, 'use_lidar': True, 'use_radar': False,
            'use_map': False, 'use_external': False,
            'dgmtrack': {'velocity': args.velocity, 'detections': str(det_dir),
                         'grid_dir': str(grid_dir) if grid_dir else None,
                         'created_utc': datetime.now(timezone.utc).isoformat(), 'counts': n}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'meta': meta, 'results': results}))
    print('wrote %s' % out_path)
    print('  scenes %d, keyframes %d, boxes %d%s'
          % (len(scenes), n['keyframes'], n['boxes'],
             ', grid valid %d (%.1f%%)' % (n['grid_valid'], 100.0 * n['grid_valid'] / max(n['boxes'], 1))
             if args.velocity == 'grid' else ''))


if __name__ == '__main__':
    main()
