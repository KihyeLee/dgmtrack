#!/usr/bin/env python3
"""Merge per-scene tracking submissions into one nuScenes submission file.

The runner executes one scene per process so that a dropped external drive
costs at most a single scene instead of the whole sweep. This stitches the
per-scene outputs back into the single file the evaluator expects.
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True, help='dir holding <scene>/tracking_results.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.in_dir, '*', 'tracking_results.json')))
    if not parts:
        raise SystemExit(f'no per-scene tracking_results.json under {args.in_dir}')

    merged, meta = {}, None
    for p in parts:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        meta = meta or d.get('meta')
        dup = set(merged) & set(d['results'])
        if dup:
            raise SystemExit(f'{p}: {len(dup)} sample tokens already present')
        merged.update(d['results'])

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'results': merged}, f)
    print(f'[merge] {len(parts)} scenes, {len(merged)} samples -> {args.out}')


if __name__ == '__main__':
    main()
