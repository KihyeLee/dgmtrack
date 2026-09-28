#!/usr/bin/env python3
"""Build a compact ground-truth cache from the nuScenes annotation tables.

The cache holds, per scene, every keyframe's GT boxes with a velocity computed
exactly the way the devkit does it (centered difference over prev/next
annotations, see NuScenes.box_velocity). Downstream velocity evaluation reads
this instead of the 583 MB sample_annotation.json, and never needs the devkit.

Velocities are in the GLOBAL frame, which is also the frame used by the
detection JSONL files and by the DOGM result JSONL files, so all three sources
are directly comparable.

Usage:
  python tools/build_gt_cache.py --dataroot E:/nuscenes/v1.0-trainval \
      --version v1.0-trainval --split val --out cache/gt
  python tools/build_gt_cache.py ... --verify 100   # cross-check vs devkit
"""

import argparse
import json
import math
import os
import random
import sys
import time

# nuScenes detection/tracking class mapping (mirrors
# nuscenes.eval.detection.utils.category_to_detection_name)
CATEGORY_TO_DETECTION = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}

# The 7 classes the nuScenes tracking benchmark scores.
TRACKING_CLASSES = {'car', 'truck', 'bus', 'trailer',
                    'pedestrian', 'motorcycle', 'bicycle'}

MAX_TIME_DIFF = 1.5  # seconds; devkit default


def log(msg):
    print(f'[gt_cache] {msg}', flush=True)


def load_table(meta_dir, name):
    path = os.path.join(meta_dir, f'{name}.json')
    t0 = time.time()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log(f'{name}.json: {len(data)} rows ({time.time()-t0:.1f}s)')
    return data


def resolve_meta_dir(dataroot, version):
    """nuScenes layout is <dataroot>/<version>/*.json, but users often pass the
    version dir directly."""
    for cand in (os.path.join(dataroot, version), dataroot):
        if os.path.isfile(os.path.join(cand, 'scene.json')):
            return cand
    raise FileNotFoundError(
        f'Cannot find scene.json under {dataroot} or {dataroot}/{version}')


def box_velocity(ann, ann_by_token, sample_time, max_time_diff=MAX_TIME_DIFF):
    """Centered difference identical to NuScenes.box_velocity (3D, m/s).

    Returns None when the devkit would return NaN, so callers can distinguish
    "no GT velocity available" from "velocity is zero".
    """
    has_prev = ann['prev'] != ''
    has_next = ann['next'] != ''
    if not has_prev and not has_next:
        return None

    first = ann_by_token[ann['prev']] if has_prev else ann
    last = ann_by_token[ann['next']] if has_next else ann

    t_first = sample_time.get(first['sample_token'])
    t_last = sample_time.get(last['sample_token'])
    if t_first is None or t_last is None:
        return None

    time_diff = t_last - t_first
    # The devkit doubles the budget when a true centered difference is taken.
    limit = max_time_diff * 2 if (has_prev and has_next) else max_time_diff
    if time_diff > limit or time_diff <= 0:
        return None

    return [(last['translation'][i] - first['translation'][i]) / time_diff
            for i in range(3)]


def build(args):
    meta_dir = resolve_meta_dir(args.dataroot, args.version)
    log(f'meta dir: {meta_dir}')

    scenes = load_table(meta_dir, 'scene')
    samples = load_table(meta_dir, 'sample')
    instances = load_table(meta_dir, 'instance')
    categories = load_table(meta_dir, 'category')
    anns = load_table(meta_dir, 'sample_annotation')  # the big one

    # Restrict to the requested split.
    if args.split:
        from nuscenes.utils.splits import create_splits_scenes
        wanted = set(create_splits_scenes()[args.split])
        log(f'split "{args.split}": {len(wanted)} scenes')
    else:
        wanted = None

    scene_by_token = {s['token']: s for s in scenes}
    keep_scene_tokens = {
        tok for tok, s in scene_by_token.items()
        if wanted is None or s['name'] in wanted
    }

    sample_time = {s['token']: s['timestamp'] * 1e-6 for s in samples}
    sample_scene = {s['token']: s['scene_token'] for s in samples}

    cat_name = {c['token']: c['name'] for c in categories}
    inst_cat = {i['token']: cat_name.get(i['category_token'], '')
                for i in instances}

    ann_by_token = {a['token']: a for a in anns}

    # Group annotations by scene, computing velocity as we go.
    per_scene = {}
    n_used = n_novel = 0
    for ann in anns:
        stok = ann['sample_token']
        sc_tok = sample_scene.get(stok)
        if sc_tok not in keep_scene_tokens:
            continue

        raw_cat = inst_cat.get(ann['instance_token'], '')
        cls = CATEGORY_TO_DETECTION.get(raw_cat)
        if cls is None or (args.tracking_only and cls not in TRACKING_CLASSES):
            continue

        vel = box_velocity(ann, ann_by_token, sample_time)
        if vel is None:
            n_novel += 1

        scene_name = scene_by_token[sc_tok]['name']
        per_scene.setdefault(scene_name, {}).setdefault(stok, []).append({
            'token': ann['token'],
            'instance_token': ann['instance_token'],
            'cls': cls,
            'translation': ann['translation'],
            'size': ann['size'],
            'rotation': ann['rotation'],
            # velocity: [vx, vy] global m/s, or null when undefined
            'velocity': None if vel is None else [vel[0], vel[1]],
            'num_lidar_pts': ann.get('num_lidar_pts', 0),
            'visibility': int(ann.get('visibility_token', 0) or 0),
        })
        n_used += 1

    log(f'kept {n_used} annotations over {len(per_scene)} scenes '
        f'({n_novel} without a defined GT velocity)')

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        'version': args.version,
        'split': args.split,
        'dataroot': args.dataroot,
        'tracking_only': args.tracking_only,
        'max_time_diff_s': MAX_TIME_DIFF,
        'velocity_frame': 'global',
        'velocity_method': 'centered_difference (devkit NuScenes.box_velocity)',
        'n_annotations': n_used,
        'n_without_velocity': n_novel,
        'scenes': sorted(per_scene.keys()),
    }
    for scene_name, frames in per_scene.items():
        # frames: sample_token -> [obj, ...]; also record timestamps so
        # downstream code can build per-instance time series.
        out = {
            'scene': scene_name,
            'frames': {
                stok: {'timestamp_s': sample_time[stok], 'objects': objs}
                for stok, objs in frames.items()
            },
        }
        with open(os.path.join(args.out, f'{scene_name}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(out, f)
    with open(os.path.join(args.out, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    log(f'wrote {len(per_scene)} scene files + manifest.json to {args.out}')
    return manifest


def verify(args, n_samples):
    """Cross-check cached velocities against the devkit's own box_velocity."""
    log(f'verification: loading devkit (slow) to check {n_samples} annotations')
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cached = []
    for fn in sorted(os.listdir(args.out)):
        if fn == 'manifest.json' or not fn.endswith('.json'):
            continue
        with open(os.path.join(args.out, fn), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for frame in data['frames'].values():
            cached.extend(frame['objects'])

    random.seed(0)
    picks = random.sample(cached, min(n_samples, len(cached)))
    worst = 0.0
    n_both_nan = n_mismatch_nan = 0
    for obj in picks:
        ref = nusc.box_velocity(obj['token'])  # global frame, may be NaN
        ref_nan = bool(math.isnan(ref[0]))
        if obj['velocity'] is None or ref_nan:
            if (obj['velocity'] is None) == ref_nan:
                n_both_nan += 1
            else:
                n_mismatch_nan += 1
            continue
        d = max(abs(obj['velocity'][0] - ref[0]), abs(obj['velocity'][1] - ref[1]))
        worst = max(worst, d)

    log(f'checked {len(picks)}: max abs diff = {worst:.3e} m/s, '
        f'both-undefined = {n_both_nan}, undefined-mismatch = {n_mismatch_nan}')
    ok = worst < 1e-9 and n_mismatch_nan == 0
    log('VERIFY PASS' if ok else 'VERIFY FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--split', default='val',
                    help='nuScenes split name, or empty for all scenes')
    ap.add_argument('--out', required=True, help='output cache directory')
    ap.add_argument('--tracking-only', action='store_true', default=True,
                    help='keep only the 7 tracking classes (default)')
    ap.add_argument('--all-classes', dest='tracking_only', action='store_false')
    ap.add_argument('--verify', type=int, default=0, metavar='N',
                    help='after building, cross-check N annotations vs devkit')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()

    if not args.verify_only:
        build(args)
    if args.verify:
        if not verify(args, args.verify):
            sys.exit(1)


if __name__ == '__main__':
    main()
