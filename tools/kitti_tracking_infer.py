"""Run an OpenPCDet KITTI model over the KITTI *tracking* sequences.

Every KITTI number in this project came from the AB3DMOT release of PointRCNN
(2019) detections, so a reviewer cannot tell whether the grid prior helps only
because the detector is weak. This regenerates detections for the same
validation sequences with any KITTI model in the OpenPCDet zoo, writing the
AB3DMOT detection format so tools/kitti_to_dgm.py reads them unchanged.

The tracking split ships no images here, so FOV point filtering and the 2D box
projection use a per-sequence image size recovered from the clipping of the 2D
boxes in label_02 (max x2 = width - 1, max y2 = height - 1): 1242x375
everywhere except 0014-0017 (1224x370) and 0020 (1241x376). Only points at the
image border are affected and the 2D box plays no part in the 3D evaluation.

Scores are written as logits, because kitti_to_dgm.py puts the score column of
an AB3DMOT file through a sigmoid.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import calibration_kitti, common_utils

IMAGE_SHAPE = {'0014': (370, 1224), '0015': (370, 1224), '0016': (370, 1224),
               '0017': (370, 1224), '0020': (376, 1241)}
DEFAULT_IMAGE_SHAPE = (375, 1242)
TYPE_ID = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3}   # AB3DMOT detection files


def read_tracking_calib(path):
    """Tracking calib writes 'P2:' with a colon but 'R_rect'/'Tr_velo_cam' without."""
    raw = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, rest = line.partition(':') if ':' in line else line.partition(' ')
        vals = [float(v) for v in rest.split()]
        if vals:
            raw[key.strip()] = np.array(vals, dtype=np.float32)

    def pick(*names):
        for n in names:
            if n in raw:
                return raw[n]
        raise KeyError('%s: none of %s (have %s)' % (path, names, sorted(raw)))

    return {'P2': pick('P2').reshape(3, 4),
            'P3': pick('P3').reshape(3, 4),
            'R0': pick('R_rect', 'R0_rect').reshape(3, 3),
            'Tr_velo2cam': pick('Tr_velo_cam', 'Tr_velo_to_cam').reshape(3, 4)}


def read_seqmap(path):
    """seqmap rows are 'seq empty first count'; the count is exclusive of the last frame."""
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f:
            out.append(('%04d' % int(f[0]), int(f[2]), int(f[3])))
    return out


class TrackingDataset(KittiDataset):
    """KittiDataset with its info list replaced by tracking frames.

    Only get_lidar / get_calib need overriding: __getitem__ takes the LiDAR
    path and the calib from these, so the FOV filter and the point feature
    encoding stay exactly the ones the checkpoint was trained with.
    """

    def __init__(self, dataset_cfg, class_names, kitti_root, seq_frames, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False,
                         root_path=Path(kitti_root), logger=logger)
        self.velo_dir = self.root_split_path / 'velodyne'
        self.calib_dir = self.root_split_path / 'calib'
        self._calib_cache = {}
        infos = []
        for seq, first, end in seq_frames:
            shape = np.array(IMAGE_SHAPE.get(seq, DEFAULT_IMAGE_SHAPE), dtype=np.int32)
            for frame in range(first, end):
                if not (self.velo_dir / seq / ('%06d.bin' % frame)).is_file():
                    continue
                infos.append({'point_cloud': {'lidar_idx': '%s/%06d' % (seq, frame), 'num_features': 4},
                              'image': {'image_idx': '%s/%06d' % (seq, frame), 'image_shape': shape}})
        self.kitti_infos = infos

    def get_lidar(self, idx):
        seq, frame = idx.split('/')
        return np.fromfile(str(self.velo_dir / seq / (frame + '.bin')), dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        seq = idx.split('/')[0]
        if seq not in self._calib_cache:
            self._calib_cache[seq] = calibration_kitti.Calibration(
                read_tracking_calib(self.calib_dir / (seq + '.txt')))
        return self._calib_cache[seq]


def wrap_pi(a):
    """KITTI writes rotation_y and alpha in [-pi, pi]; pcdet leaves them unwrapped.

    Without this the angles come out 2*pi away from the AB3DMOT reference files
    (e.g. -7.8511 instead of -1.5679 on sequence 0001, frame 0).
    """
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def logit(score, eps=1e-6):
    s = min(max(float(score), eps), 1.0 - eps)
    return math.log(s / (1.0 - s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--kitti_root', required=True, help='.../KITTI/tracking (has training/velodyne)')
    ap.add_argument('--seqmap', required=True)
    ap.add_argument('--out', required=True, help='detection root; writes <name>_<Class>_<split>/<seq>.txt')
    ap.add_argument('--det_name', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='stop after N frames (smoke test)')
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    seq_frames = read_seqmap(args.seqmap)
    dataset = TrackingDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.kitti_root, seq_frames, logger)
    logger.info('frames: %d over %d sequences' % (len(dataset), len(seq_frames)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=dataset.collate_batch, drop_last=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    lines = {}          # (class, seq) -> list of rows
    counts = {}
    done = 0
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model.forward(batch)
            annos = dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)
            for anno in annos:
                seq, frame = anno['frame_id'].split('/')
                frame = int(frame)
                for i in range(len(anno['name'])):
                    cls = str(anno['name'][i])
                    if cls not in TYPE_ID:
                        continue
                    bb = anno['bbox'][i]
                    l, h, w = anno['dimensions'][i]      # camera dims are l, h, w
                    x, y, z = anno['location'][i]
                    lines.setdefault((cls, seq), []).append(
                        '%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % (
                            frame, TYPE_ID[cls], bb[0], bb[1], bb[2], bb[3], logit(anno['score'][i]),
                            h, w, l, x, y, z,
                            wrap_pi(anno['rotation_y'][i]), wrap_pi(anno['alpha'][i])))
                    counts[cls] = counts.get(cls, 0) + 1
                done += 1
            if args.limit and done >= args.limit:
                break
            if done % 400 < args.batch_size:
                logger.info('%d / %d frames' % (done, len(dataset)))

    out = Path(args.out)
    written = {}
    for cls in TYPE_ID:
        d = out / ('%s_%s_%s' % (args.det_name, cls, args.split))
        d.mkdir(parents=True, exist_ok=True)
        for seq, _, _ in seq_frames:
            rows = lines.get((cls, seq), [])
            (d / ('%s.txt' % seq)).write_text('\n'.join(rows) + ('\n' if rows else ''))
            written['%s/%s' % (cls, seq)] = len(rows)
    manifest = {'cfg_file': args.cfg_file, 'ckpt': args.ckpt, 'det_name': args.det_name,
                'split': args.split, 'frames': done, 'boxes_per_class': counts,
                'rows_per_file': written, 'image_shape_default': list(DEFAULT_IMAGE_SHAPE),
                'image_shape_overrides': {k: list(v) for k, v in IMAGE_SHAPE.items()},
                'score_column': 'logit of the detector confidence (kitti_to_dgm applies sigmoid)',
                'score_thresh': cfg.MODEL.POST_PROCESSING.SCORE_THRESH,
                'nms_thresh': cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH}
    (out / ('%s_%s_manifest.json' % (args.det_name, args.split))).write_text(json.dumps(manifest, indent=1))
    logger.info('frames %d, boxes %s' % (done, counts))
    logger.info('wrote %s' % out)


if __name__ == '__main__':
    main()
