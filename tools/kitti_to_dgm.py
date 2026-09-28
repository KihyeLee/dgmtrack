#!/usr/bin/env python3
"""Convert a KITTI tracking split into the inputs dogm_track already reads.

dogm_track needs three things (src/common/nusc_io.hpp): a pose cache keyed by
sweep token, one detection JSONL per scene, and a LiDAR binary per sweep. The
tracker code is untouched; KITTI is expressed in the nuScenes conventions it
expects:

  pose cache   cs_* = velodyne -> IMU (inverse of calib Tr_imu_to_velo),
               ep_* = IMU -> world from OXTS (KITTI devkit Mercator pose,
               re-based on the first frame of each sequence so global
               coordinates stay small).
  detections   global frame, box centre (not KITTI's bottom centre),
               size [w, l, h], yaw-only quaternion, no velocity field.
               Scores are PointRCNN logits and are mapped through a sigmoid
               so the tracker's --score_thr keeps its nuScenes meaning.
               attribute_name carries
               "kitti|x1,y1,x2,y2|h,w,l,x,y,z,ry|alpha|raw_score" - the
               detection exactly as KITTI wrote it - so the evaluator writes
               the submitted box back without an inverse transform
               (see evaluate_kitti.py).
  timestamps   KITTI tracking ships none; frames are 10 Hz, so
               ts = 1 s + frame * 0.1 s (the 1 s offset keeps ts > 0, which
               the tracker uses to mean "has a previous frame").
  keyframes    --key-every N marks every N-th frame as a keyframe (tracker
               update + submission); the frames between are sweeps that feed
               only the grid. N=1 is native KITTI 10 Hz, N=5 mirrors the
               nuScenes 2 Hz keyframe / sweep structure.

Also writes a GT cache in the format of the nuScenes one (tools/build_gt_cache.py)
so prediction_diagnostic / track_continuity can read it; GT velocity is the
central difference of the global track position.

--identity-poses and --gt-as-detections exist for pipeline tests before
OXTS / detections are available; outputs made with them are marked in the
manifest and must not be reported.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TYPE_OF_ID = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}          # AB3DMOT detection files
CLASS_MAP = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}
FRAME_DT_US = 100000
TS_OFFSET_US = 1000000


# ------------------------------------------------------------------ calib
def read_calib(path):
    """R0_rect (3x3), Tr_velo_cam (4x4), Tr_imu_velo (4x4) in either key style."""
    raw = {}
    for line in Path(path).read_text().splitlines():
        if ':' not in line:
            key, _, rest = line.partition(' ')
        else:
            key, _, rest = line.partition(':')
        vals = [float(v) for v in rest.split()]
        if vals:
            raw[key.strip()] = np.array(vals)

    def pick(*names):
        for n in names:
            if n in raw:
                return raw[n]
        raise KeyError('%s: none of %s' % (path, names))

    def to44(v):
        m = np.eye(4)
        m[:3, :4] = v.reshape(3, 4)
        return m

    r0 = pick('R0_rect', 'R_rect').reshape(3, 3)
    return {'R0': r0,
            'Tr_velo_cam': to44(pick('Tr_velo_to_cam', 'Tr_velo_cam')),
            'Tr_imu_velo': to44(pick('Tr_imu_to_velo', 'Tr_imu_velo'))}


def rect_to_velo_matrix(calib):
    """4x4 taking rectified camera coordinates to velodyne coordinates."""
    r0 = np.eye(4)
    r0[:3, :3] = calib['R0']
    return np.linalg.inv(calib['Tr_velo_cam']) @ np.linalg.inv(r0)


# ------------------------------------------------------------------- oxts
def oxts_poses(path):
    """IMU -> world 4x4 per frame (KITTI devkit convertOxtsToPose), re-based on frame 0."""
    rows = [list(map(float, l.split())) for l in Path(path).read_text().splitlines() if l.strip()]
    er = 6378137.0
    scale = math.cos(rows[0][0] * math.pi / 180.0)
    poses = []
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        tx = scale * lon * math.pi * er / 180.0
        ty = scale * er * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
        rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        m = np.eye(4)
        m[:3, :3] = rz @ ry @ rx
        m[:3, 3] = [tx, ty, alt]
        poses.append(m)
    base = np.linalg.inv(poses[0])
    return [base @ p for p in poses]


def rot_to_quat_wxyz(r):
    """Rotation matrix to unit quaternion (w, x, y, z)."""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


# ------------------------------------------------------------ box transforms
class Frames:
    """Per-sequence geometry: rect camera -> velodyne -> world."""

    def __init__(self, calib, imu_world):
        self.T_rv = rect_to_velo_matrix(calib)            # rect -> velo
        self.T_vi = np.linalg.inv(calib['Tr_imu_velo'])   # velo -> imu
        self.imu_world = imu_world

    def velo_world(self, frame):
        return self.imu_world[frame] @ self.T_vi

    def box_to_global(self, frame, h, w, l, x, y, z, ry):
        """KITTI camera box (bottom centre, ry about camera y) -> global centre, yaw."""
        c_rect = np.array([x, y - h / 2.0, z, 1.0])
        # object x-axis (length direction) in rectified camera coordinates
        d_rect = np.array([math.cos(ry), 0.0, -math.sin(ry), 0.0])
        T = self.velo_world(frame) @ self.T_rv
        c = T @ c_rect
        d = T @ d_rect
        yaw = math.atan2(d[1], d[0])
        return c[:3], yaw

    def global_to_box(self, frame, center, yaw, h):
        """Inverse of box_to_global: (x, y, z bottom centre, ry) in rect camera."""
        T = np.linalg.inv(self.velo_world(frame) @ self.T_rv)
        c = T @ np.array([center[0], center[1], center[2], 1.0])
        d = T @ np.array([math.cos(yaw), math.sin(yaw), 0.0, 0.0])
        ry = math.atan2(-d[2], d[0])
        return c[0], c[1] + h / 2.0, c[2], ry


def yaw_quat(yaw):
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


# ------------------------------------------------------------------ readers
def read_labels(path):
    """KITTI tracking label file -> {frame: [dict]} (DontCare kept, for the evaluator)."""
    out = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) < 17:
            continue
        out[int(f[0])].append({
            'track_id': int(f[1]), 'type': f[2], 'truncated': float(f[3]), 'occluded': int(float(f[4])),
            'alpha': float(f[5]), 'bbox': [float(v) for v in f[6:10]],
            'h': float(f[10]), 'w': float(f[11]), 'l': float(f[12]),
            'x': float(f[13]), 'y': float(f[14]), 'z': float(f[15]), 'ry': float(f[16])})
    return out


def read_ab3dmot_detections(det_root, det_name, split, seq):
    """AB3DMOT detection txt: frame,type,x1,y1,x2,y2,score,h,w,l,x,y,z,ry,alpha."""
    out = defaultdict(list)
    for cls in ('Car', 'Pedestrian', 'Cyclist'):
        path = Path(det_root) / ('%s_%s_%s' % (det_name, cls, split)) / ('%s.txt' % seq)
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            f = line.replace(',', ' ').split()
            if len(f) < 15:
                continue
            out[int(float(f[0]))].append({
                'type': TYPE_OF_ID[int(float(f[1]))], 'bbox': [float(v) for v in f[2:6]], 'score': float(f[6]),
                'h': float(f[7]), 'w': float(f[8]), 'l': float(f[9]),
                'x': float(f[10]), 'y': float(f[11]), 'z': float(f[12]), 'ry': float(f[13]), 'alpha': float(f[14])})
    return out


def read_seqmap(path):
    """(seq, first, end) with end EXCLUSIVE: the seqmap's last field is a frame count
    (0006 lists 000270 and has 270 OXTS rows / frames 0-269; 0001's last label is 446
    for 000447). AB3DMOT's evaluator reads it as inclusive and only gains an empty slot."""
    seqs = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f:
            seqs.append(('%04d' % int(f[0]), int(f[2]), int(f[3])))
    return seqs


def points_in_box(pts_velo, center_velo, yaw_velo, w, l, h):
    d = pts_velo[:, :3] - center_velo
    c, s = math.cos(-yaw_velo), math.sin(-yaw_velo)
    x = c * d[:, 0] - s * d[:, 1]
    y = s * d[:, 0] + c * d[:, 1]
    return int(np.sum((np.abs(x) <= l / 2) & (np.abs(y) <= w / 2) & (np.abs(d[:, 2]) <= h / 2)))


# --------------------------------------------------------------------- main
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def diagnostic_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('diagnostic', ROOT / 'tools/prediction_diagnostic.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kitti-root', required=True,
                    help='KITTI tracking root holding training/{velodyne,oxts,calib}')
    ap.add_argument('--ab3dmot', required=True, help='AB3DMOT checkout (labels, seqmap, detections); read only')
    ap.add_argument('--split', default='val')
    ap.add_argument('--det-name', default='pointrcnn')
    ap.add_argument('--key-every', type=int, default=1)
    ap.add_argument('--out', required=True, help='output directory (inside this workspace)')
    ap.add_argument('--det-root', default=None,
                    help='detection root holding <det-name>_<Class>_<split>/<seq>.txt '
                         '(default: <ab3dmot>/data/KITTI/detection)')
    ap.add_argument('--calib-dir', default=None, help='fallback calib dir (default: <ab3dmot>/data/KITTI/mini/training/calib)')
    ap.add_argument('--identity-poses', action='store_true', help='TEST ONLY: ego fixed at the origin')
    ap.add_argument('--gt-as-detections', action='store_true', help='TEST ONLY: detections = GT boxes shifted 3 cm, score 10')
    ap.add_argument('--no-lidar-count', action='store_true', help='skip num_lidar_pts (velodyne absent)')
    ap.add_argument('--oracle-velocity', action='store_true',
                    help='CEILING ONLY: each keyframe detection matched one-to-one to a visible GT object of '
                         'the same class within 2 m gets its velocity (rule of make_oracle_velocity_cache.py); '
                         'unmatched ones get none. Run with --vel_source detector --fusion kf.')
    args = ap.parse_args()

    kitti = Path(args.kitti_root)
    ab = Path(args.ab3dmot)
    out = Path(args.out)
    label_dir = ab / 'scripts/KITTI/label'
    seqmap = ab / ('scripts/KITTI/evaluate_tracking.seqmap.%s' % args.split)
    calib_fallback = Path(args.calib_dir) if args.calib_dir else ab / 'data/KITTI/mini/training/calib'
    det_root = Path(args.det_root) if args.det_root else ab / 'data/KITTI/detection'
    (out / 'detections' / args.det_name).mkdir(parents=True, exist_ok=True)
    (out / 'gt').mkdir(parents=True, exist_ok=True)

    poses_json = {}
    scenes = []
    counts = {}
    inputs = {}
    for seq, first, last in read_seqmap(seqmap):
        scene = 'kitti-%s' % seq
        scenes.append(scene)
        calib_path = kitti / 'training/calib' / ('%s.txt' % seq)
        if not calib_path.is_file():
            calib_path = calib_fallback / ('%s.txt' % seq)
        calib = read_calib(calib_path)
        inputs[scene] = {'calib': str(calib_path), 'calib_sha256': sha256(calib_path)}
        n_frames = last - first
        if args.identity_poses:
            imu_world = [np.eye(4) for _ in range(n_frames)]
        else:
            oxts_path = kitti / 'training/oxts' / ('%s.txt' % seq)
            imu_world = oxts_poses(oxts_path)
            inputs[scene]['oxts_sha256'] = sha256(oxts_path)
            if len(imu_world) < n_frames:
                raise ValueError('%s: %d oxts rows < %d frames' % (seq, len(imu_world), n_frames))
        geo = Frames(calib, imu_world)
        labels = read_labels(label_dir / ('%s.txt' % seq))
        dets = (None if args.gt_as_detections
                else read_ab3dmot_detections(det_root, args.det_name, args.split, seq))

        det_lines, gt_frames = [], {}
        tracks = defaultdict(dict)       # track_id -> frame -> global centre
        n_det = 0
        for frame in range(first, last):
            token = '%s_%06d' % (scene, frame)
            ts = TS_OFFSET_US + frame * FRAME_DT_US
            T_vw = geo.velo_world(frame)
            R_vi = geo.T_vi[:3, :3]
            poses_json[token] = {
                'ts': ts, 'file': 'training/velodyne/%s/%06d.bin' % (seq, frame), 'scene': scene,
                'cs_t': geo.T_vi[:3, 3].tolist(), 'cs_q': rot_to_quat_wxyz(R_vi),
                'ep_t': imu_world[frame][:3, 3].tolist(), 'ep_q': rot_to_quat_wxyz(imu_world[frame][:3, :3])}
            if args.gt_as_detections:
                # shifted 3 cm: an exact copy of the GT box makes AB3DMOT's polygon
                # clipping divide by zero on coincident edges
                source = [dict(g, x=g['x'] + 0.03, z=g['z'] + 0.03) for g in labels.get(frame, [])]
            else:
                source = dets.get(frame, [])
            boxes = []
            for d in source:
                if d['type'] not in CLASS_MAP:
                    continue
                c, yaw = geo.box_to_global(frame, d['h'], d['w'], d['l'], d['x'], d['y'], d['z'], d['ry'])
                raw = 10.0 if args.gt_as_detections else d['score']
                boxes.append({
                    'translation': c.tolist(), 'size': [d['w'], d['l'], d['h']], 'rotation': yaw_quat(yaw),
                    'detection_name': CLASS_MAP[d['type']], 'detection_score': sigmoid(raw),
                    'attribute_name': 'kitti|%s|%s|%r|%r' % (
                        ','.join('%r' % v for v in d['bbox']),
                        ','.join('%r' % d[k] for k in ('h', 'w', 'l', 'x', 'y', 'z', 'ry')),
                        d['alpha'], raw)})
            n_det += len(boxes)
            det_lines.append({'sample_token': token, 'sd_token': token,
                              'is_key_frame': (frame - first) % args.key_every == 0,
                              'timestamp': ts, 'boxes': boxes})
            objs = []
            velo = None
            vpath = kitti / 'training/velodyne' / seq / ('%06d.bin' % frame)
            if not args.no_lidar_count and vpath.is_file():
                velo = np.fromfile(vpath, dtype=np.float32).reshape(-1, 4)
            for g in labels.get(frame, []):
                if g['type'] == 'DontCare':
                    continue
                c, yaw = geo.box_to_global(frame, g['h'], g['w'], g['l'], g['x'], g['y'], g['z'], g['ry'])
                tracks[g['track_id']][frame] = c
                npts = -1
                if velo is not None:
                    cv = np.linalg.inv(T_vw) @ np.append(c, 1.0)
                    dv = np.linalg.inv(T_vw)[:3, :3] @ np.array([math.cos(yaw), math.sin(yaw), 0.0])
                    npts = points_in_box(velo, cv[:3], math.atan2(dv[1], dv[0]), g['w'], g['l'], g['h'])
                objs.append({'token': '%s_%d' % (token, g['track_id']),
                             'instance_token': '%s_%d' % (scene, g['track_id']),
                             'cls': CLASS_MAP.get(g['type'], g['type'].lower()), 'kitti_type': g['type'],
                             'translation': c.tolist(), 'size': [g['w'], g['l'], g['h']],
                             'rotation': yaw_quat(yaw), 'velocity': None, 'num_lidar_pts': npts,
                             'truncated': g['truncated'], 'occluded': g['occluded']})
            gt_frames[token] = {'timestamp_s': ts * 1e-6, 'objects': objs}
        # GT velocity: central difference of the global centre (one-sided at track ends)
        for token, fr in gt_frames.items():
            frame = int(token.rsplit('_', 1)[1])
            for o in fr['objects']:
                tr = tracks[int(o['instance_token'].rsplit('_', 1)[1])]
                a = tr.get(frame - 1, tr.get(frame))
                b = tr.get(frame + 1, tr.get(frame))
                span = ((frame + 1 if frame + 1 in tr else frame) - (frame - 1 if frame - 1 in tr else frame)) * 0.1
                o['velocity'] = ((b - a)[:2] / span).tolist() if span > 0 else None
        n_oracle = 0
        if args.oracle_velocity:
            match = diagnostic_module().match_one_to_one
            for fr in det_lines:
                if not fr['is_key_frame']:
                    continue
                gt = [o for o in gt_frames[fr['sample_token']]['objects']
                      if o['num_lidar_pts'] > 0 and o['velocity'] is not None]
                pairs = match([{'cls': g['cls'], 'translation': g['translation']} for g in gt],
                              [{'tracking_name': b['detection_name'], 'translation': b['translation']}
                               for b in fr['boxes']])
                for gi, bi in pairs:
                    fr['boxes'][bi]['velocity'] = [float(v) for v in gt[gi]['velocity']]
                    n_oracle += 1
        (out / 'detections' / args.det_name / ('%s.jsonl' % scene)).write_text(
            '\n'.join(json.dumps(fr) for fr in det_lines) + '\n')
        (out / 'gt' / ('%s.json' % scene)).write_text(json.dumps({'scene': scene, 'frames': gt_frames}))
        counts[scene] = {'frames': n_frames, 'detections': n_det, 'oracle_matched': n_oracle,
                         'gt_objects': sum(len(f['objects']) for f in gt_frames.values())}
        print('%s frames %4d detections %6d gt %6d' % (scene, n_frames, n_det, counts[scene]['gt_objects']), flush=True)

    (out / 'poses.json').write_text(json.dumps({'frames': poses_json}))
    (out / 'scenes.txt').write_text(','.join(scenes) + '\n')
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': 'tools/kitti_to_dgm.py',
        'argv': sys.argv[1:], 'split': args.split, 'det_name': args.det_name, 'key_every': args.key_every,
        'test_only': bool(args.identity_poses or args.gt_as_detections),
        'oracle_velocity': bool(args.oracle_velocity),
        'score_mapping': 'sigmoid(PointRCNN logit); raw logit kept in attribute_name',
        'class_map': CLASS_MAP, 'timestamps': 'nominal 10 Hz, ts_us = 1e6 + frame * 1e5',
        'scenes': counts, 'inputs': inputs, 'converter_sha256': sha256(__file__)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
