#!/usr/bin/env python3
"""Write a detection cache whose keyframe velocities come from ground truth.

This is a diagnostic ceiling, not a method: it answers "how much could the
tracker gain if the external velocity were perfect?". Every keyframe detection
is matched one-to-one to a visible ground-truth object of the same class within
2 m -- the same rule the prediction diagnostic uses -- and its velocity field is
replaced by the annotated velocity. Detections with no ground-truth match keep a
null velocity, so the oracle never invents motion for a false positive.

Sweep rows are copied unchanged; the tracker only updates on keyframes.
Ground truth enters the tracker here, so any run using this cache must be
labelled an oracle and never reported as achievable performance.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('diagnostic', ROOT / 'tools/prediction_diagnostic.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detections', required=True, help='source cache directory (absolute or cache name)')
    parser.add_argument('--out', required=True, help='destination cache directory')
    args = parser.parse_args()
    paths = json.loads((ROOT / 'configs/paths.json').read_text())
    source = (Path(args.detections) if args.detections.startswith('/')
              else Path(paths['original']) / 'cache' / args.detections)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in (ROOT / 'configs/val150_scenes.txt').read_text().split() if s]
    counts = {'keyframes': 0, 'boxes': 0, 'matched': 0, 'sweep_rows': 0}
    for scene in scenes:
        cached = json.loads((Path(paths['gt_cache']) / f'{scene}.json').read_text())['frames']
        with (source / f'{scene}.jsonl').open() as stream, (out / f'{scene}.jsonl').open('w') as sink:
            for line in stream:
                row = json.loads(line)
                if not row.get('is_key_frame'):
                    counts['sweep_rows'] += 1
                    sink.write(line if line.endswith('\n') else line + '\n')
                    continue
                counts['keyframes'] += 1
                boxes = row['boxes']
                counts['boxes'] += len(boxes)
                for box in boxes:
                    box['velocity'] = None
                gt = [g for g in cached.get(row['sample_token'], {'objects': []})['objects']
                      if g['num_lidar_pts'] > 0 and g.get('velocity') is not None]
                if gt:
                    pairs = diagnostic.match_one_to_one(
                        [{'cls': g['cls'], 'translation': g['translation']} for g in gt],
                        [{'tracking_name': b['detection_name'], 'translation': b['translation']} for b in boxes])
                    for gi, bi in pairs:
                        boxes[bi]['velocity'] = [float(gt[gi]['velocity'][0]), float(gt[gi]['velocity'][1])]
                        counts['matched'] += 1
                sink.write(json.dumps(row) + '\n')
    (out / 'ORACLE_README.json').write_text(json.dumps({
        'source': str(source), 'created_by': 'tools/make_oracle_velocity_cache.py',
        'warning': 'Keyframe velocities are ground truth. Diagnostic ceiling only.',
        **counts}, indent=2) + '\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
