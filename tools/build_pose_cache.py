#!/usr/bin/env python3
"""Build a compact per-frame pose/calibration cache for the C++ tracker.

The tracker only needs four things per LiDAR frame: the timestamp, the point
cloud path, the calibrated-sensor extrinsics and the ego pose. Reading those
from the raw nuScenes tables costs a full parse of sample_data.json (1.3 GB)
and ego_pose.json (645 MB) on every single run, which dominates the runtime of
short experiments and makes ablation sweeps impractical.

This script extracts exactly that subset once, so the tracker can start
instantly.

Usage:
  python tools/build_pose_cache.py --dataroot E:/nuscenes/v1.0-trainval \
      --version v1.0-trainval --split val --out cache/poses.json
"""

import argparse
import json
import os
import time


def log(msg):
    print(f'[pose_cache] {msg}', flush=True)


def load_table(meta_dir, name):
    t0 = time.time()
    with open(os.path.join(meta_dir, f'{name}.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    log(f'{name}.json: {len(data)} rows ({time.time()-t0:.1f}s)')
    return data


def resolve_meta_dir(dataroot, version):
    for cand in (os.path.join(dataroot, version), dataroot):
        if os.path.isfile(os.path.join(cand, 'scene.json')):
            return cand
    raise FileNotFoundError(f'no scene.json under {dataroot} / {dataroot}/{version}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--split', default='val')
    ap.add_argument('--channel', default='LIDAR_TOP')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    meta = resolve_meta_dir(args.dataroot, args.version)
    log(f'meta dir: {meta}')

    scenes = load_table(meta, 'scene')
    samples = load_table(meta, 'sample')
    calib = load_table(meta, 'calibrated_sensor')
    sensors = load_table(meta, 'sensor')
    egos = load_table(meta, 'ego_pose')
    sds = load_table(meta, 'sample_data')

    if args.split:
        from nuscenes.utils.splits import create_splits_scenes
        wanted = set(create_splits_scenes()[args.split])
    else:
        wanted = None

    scene_name = {s['token']: s['name'] for s in scenes}
    keep_scene = {tok for tok, nm in scene_name.items()
                  if wanted is None or nm in wanted}
    sample_scene = {s['token']: s['scene_token'] for s in samples}

    calib_by = {c['token']: c for c in calib}
    ego_by = {e['token']: e for e in egos}
    # sample_data has no channel field; it is reached through
    # calibrated_sensor -> sensor.
    sensor_channel = {s['token']: s['channel'] for s in sensors}
    calib_channel = {c['token']: sensor_channel.get(c['sensor_token'], '')
                     for c in calib}

    frames = {}
    n_skip = 0
    for sd in sds:
        if calib_channel.get(sd['calibrated_sensor_token']) != args.channel:
            continue
        sc = sample_scene.get(sd['sample_token'])
        if sc not in keep_scene:
            continue
        cs = calib_by.get(sd['calibrated_sensor_token'])
        ep = ego_by.get(sd['ego_pose_token'])
        if cs is None or ep is None:
            n_skip += 1
            continue
        frames[sd['token']] = {
            'ts': sd['timestamp'],
            'file': sd['filename'],
            'scene': scene_name[sc],
            'cs_t': cs['translation'],
            'cs_q': cs['rotation'],
            'ep_t': ep['translation'],
            'ep_q': ep['rotation'],
        }

    log(f'{len(frames)} {args.channel} frames over '
        f'{len({f["scene"] for f in frames.values()})} scenes'
        + (f' ({n_skip} skipped)' if n_skip else ''))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'version': args.version,
            'split': args.split,
            'channel': args.channel,
            'dataroot': args.dataroot,
            'frames': frames,
        }, f)
    size_mb = os.path.getsize(args.out) / 1e6
    log(f'wrote {args.out} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
