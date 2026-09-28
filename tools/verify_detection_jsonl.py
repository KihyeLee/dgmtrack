#!/usr/bin/env python3
"""Check that a detection jsonl directory is complete and not truncated.

A run interrupted mid-scene leaves a partial file that the resume logic would
happily keep, silently producing a detector input with missing frames. The
reference is another detector's jsonl for the same split: both enumerate the
same LIDAR_TOP sample_data, so per scene the row count and the sd_token set
must match exactly.

Prints the scenes that are missing, short, or mismatched; with --delete-bad it
removes them so the exporter regenerates them on the next run.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scene_tokens(path):
    tokens = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return tokens, True          # truncated last line
            tokens.append(row['sd_token'])
    return tokens, False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', required=True, help='detection jsonl directory to check')
    parser.add_argument('--reference', default=None,
                        help='reference jsonl directory; default is the VoxelNeXt cache')
    parser.add_argument('--delete-bad', action='store_true')
    args = parser.parse_args()
    paths = json.loads((ROOT / 'configs/paths.json').read_text())
    target = Path(args.dir)
    reference = Path(args.reference) if args.reference else Path(paths['voxelnext'])
    scenes = [s for s in (ROOT / 'configs/val150_scenes.txt').read_text().split() if s]

    missing, bad, ok = [], [], 0
    for scene in scenes:
        path = target / f'{scene}.jsonl'
        if not path.is_file():
            missing.append(scene)
            continue
        tokens, truncated = scene_tokens(path)
        ref_tokens, _ = scene_tokens(reference / f'{scene}.jsonl')
        if truncated or tokens != ref_tokens:
            reason = 'truncated' if truncated else f'{len(tokens)} rows vs {len(ref_tokens)}'
            bad.append((scene, reason))
            if args.delete_bad:
                path.unlink()
        else:
            ok += 1
    print(json.dumps({'directory': str(target), 'reference': str(reference),
                      'complete_scenes': ok, 'missing': len(missing), 'bad': len(bad),
                      'deleted': args.delete_bad}, indent=2))
    for scene, reason in bad[:10]:
        print('  bad:', scene, reason)
    if missing[:10]:
        print('  missing:', ', '.join(missing[:10]), '...' if len(missing) > 10 else '')


if __name__ == '__main__':
    main()
