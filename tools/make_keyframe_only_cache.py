#!/usr/bin/env python3
"""Copy a detection cache, thinning the sweep rows between keyframes.

The proposed module consumes the LiDAR sweeps between keyframes, so a detector
whose cache carries sweeps gives the grid far more evidence before a track's
first update than one that does not. In our three detection caches that factor
is confounded with the detector itself: only VoxelNeXt carries sweeps. Dropping
the sweep rows holds the detector, the boxes and the scores fixed and changes
only how much LiDAR the grid sees, which separates the two explanations.

The tracker updates on keyframes either way, so a configuration that does not
read the grid is unaffected by this cache.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--keep-every', type=int, default=0,
                        help='keep every Nth sweep row between keyframes; 0 keeps none '
                             '(keyframe-only), 1 keeps all')
    args = parser.parse_args()
    paths = json.loads((ROOT / 'configs/paths.json').read_text())
    source = (Path(args.detections) if args.detections.startswith('/')
              else Path(paths['original']) / 'cache' / args.detections)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in (ROOT / 'configs/val150_scenes.txt').read_text().split() if s]
    kept = dropped = swept = 0
    for scene in scenes:
        since_keyframe = 0
        with (source / f'{scene}.jsonl').open() as stream, (out / f'{scene}.jsonl').open('w') as sink:
            for line in stream:
                if json.loads(line).get('is_key_frame'):
                    kept += 1
                    since_keyframe = 0
                    sink.write(line if line.endswith('\n') else line + '\n')
                    continue
                since_keyframe += 1
                if args.keep_every > 0 and since_keyframe % args.keep_every == 0:
                    swept += 1
                    sink.write(line if line.endswith('\n') else line + '\n')
                else:
                    dropped += 1
    (out / 'KEYFRAME_ONLY_README.json').write_text(json.dumps(
        {'source': str(source), 'created_by': 'tools/make_keyframe_only_cache.py',
         'keep_every': args.keep_every, 'keyframe_rows': kept,
         'sweep_rows_kept': swept, 'sweep_rows_dropped': dropped}, indent=2) + '\n')
    print(json.dumps({'keep_every': args.keep_every, 'keyframe_rows': kept,
                      'sweep_rows_kept': swept, 'sweep_rows_dropped': dropped}))


if __name__ == '__main__':
    main()
