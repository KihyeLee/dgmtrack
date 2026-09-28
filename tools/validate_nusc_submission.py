#!/usr/bin/env python3
"""Check that a tracking submission is one the nuScenes server will accept.

There is no ground truth for the test split, so nothing can be scored locally.
What can be checked is exactly what the official loader checks, plus the things
that silently invalidate a submission:

  - the devkit's own load_prediction() parses the file into TrackingBox objects
  - every sample token of the split is a key (the devkit asserts this against GT
    on the server; a missing scene is the usual cause of a rejected upload)
  - no sample exceeds max_boxes_per_sample of the official configuration
  - tracking_name is one of the evaluated classes, tracking_id is a string,
    translation/size/rotation/velocity have the right arity and no NaN
  - meta is present with the four required boolean modality fields

Writes a JSON report. Exit status is non-zero if any run fails.
"""
import argparse
import json
import math
import os
import sys

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.splits import create_splits_scenes

REQUIRED_META = ('use_camera', 'use_lidar', 'use_radar', 'use_map', 'use_external')


def sample_tokens_of_split(nusc, split):
    wanted = set(create_splits_scenes()[split])
    toks = []
    for scene in nusc.scene:
        if scene['name'] not in wanted:
            continue
        tok = scene['first_sample_token']
        while tok:
            toks.append(tok)
            tok = nusc.get('sample', tok)['next']
    return toks


def check(path, nusc, split, cfg):
    rep = {'file': path, 'errors': [], 'warnings': []}
    if not os.path.isfile(path):
        rep['errors'].append('file does not exist')
        return rep
    raw = json.load(open(path))
    meta = raw.get('meta') or {}
    for k in REQUIRED_META:
        if k not in meta:
            rep['errors'].append('meta is missing %s' % k)
        elif not isinstance(meta[k], bool):
            rep['errors'].append('meta.%s is not a boolean' % k)
    rep['meta'] = meta

    # the official loader; it raises on anything it cannot parse
    try:
        boxes, _ = load_prediction(path, cfg.max_boxes_per_sample, TrackingBox, verbose=False)
    except Exception as exc:                                   # noqa: BLE001
        rep['errors'].append('devkit load_prediction failed: %s: %s' % (type(exc).__name__, exc))
        return rep

    split_toks = set(sample_tokens_of_split(nusc, split))
    got = set(boxes.sample_tokens)
    rep['samples_in_split'] = len(split_toks)
    rep['samples_in_file'] = len(got)
    missing, extra = split_toks - got, got - split_toks
    if missing:
        rep['errors'].append('%d split samples absent from the file (e.g. %s)'
                             % (len(missing), sorted(missing)[:3]))
    if extra:
        rep['errors'].append('%d samples in the file are not in the split (e.g. %s)'
                             % (len(extra), sorted(extra)[:3]))

    classes = set(cfg.class_names)
    n_boxes, bad_cls, n_ids, worst = 0, set(), set(), 0
    for tok in boxes.sample_tokens:
        bs = boxes[tok]
        worst = max(worst, len(bs))
        for b in bs:
            n_boxes += 1
            n_ids.add(b.tracking_id)
            if b.tracking_name not in classes:
                bad_cls.add(b.tracking_name)
            if not isinstance(b.tracking_id, str):
                rep['errors'].append('tracking_id is not a string: %r' % (b.tracking_id,))
            for name, val, k in (('translation', b.translation, 3), ('size', b.size, 3),
                                 ('rotation', b.rotation, 4), ('velocity', b.velocity, 2)):
                if len(val) != k:
                    rep['errors'].append('%s has %d values, expected %d' % (name, len(val), k))
                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in val):
                    rep['errors'].append('%s contains NaN or null' % name)
    rep['boxes'] = n_boxes
    rep['tracks'] = len(n_ids)
    rep['max_boxes_in_one_sample'] = worst
    rep['max_boxes_per_sample_allowed'] = cfg.max_boxes_per_sample
    if worst > cfg.max_boxes_per_sample:
        rep['errors'].append('a sample has %d boxes, over the limit of %d'
                             % (worst, cfg.max_boxes_per_sample))
    if bad_cls:
        rep['errors'].append('tracking_name outside the evaluated classes: %s' % sorted(bad_cls))
    # a submission that is accepted but empty for a class scores zero there
    if n_boxes == 0:
        rep['errors'].append('no boxes at all')
    rep['errors'] = sorted(set(rep['errors']))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-test')
    ap.add_argument('--split', default='test')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    cfg = config_factory('tracking_nips_2019')
    reports = [check(p, nusc, args.split, cfg) for p in args.runs]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump({'version': args.version, 'split': args.split, 'reports': reports},
              open(args.out, 'w'), indent=2)
    bad = 0
    for r in reports:
        ok = not r['errors']
        bad += 0 if ok else 1
        print('%-70s %s' % (os.path.basename(os.path.dirname(r['file'])) or r['file'],
                            'OK  boxes %s tracks %s samples %s/%s'
                            % (r.get('boxes'), r.get('tracks'), r.get('samples_in_file'),
                               r.get('samples_in_split')) if ok else 'FAILED'))
        for e in r['errors']:
            print('    - %s' % e)
    print('wrote %s' % args.out)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
