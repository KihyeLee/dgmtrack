#!/usr/bin/env python3
"""Tracker-output velocity error against track age, per KITTI detector.

Why: on 2026-09-28 the policy's Car sAMOTA gain fell from +0.0146 (PointRCNN) to
+0.0023 (PV-RCNN) while the IDS gain grew. The oracle still gains +0.0316 sAMOTA
over the PV-RCNN baseline, so the baseline is NOT saturated - something else
changed. The mechanism to test: with better detections the tracker's own
constant-velocity estimate converges sooner, so the grid's early-age advantage
(the whole basis of the policy) is smaller.

This reuses tools/early_track_velocity.py unchanged - same matching, same
bootstrap, same age binning - and only points it at the new runs. Writes
results/early_track_velocity_moderndet/summary.json.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = Path(os.environ.get('DGM_WORK', '/media/mt-pc-0099/NVMe4TB/dgmtrack_work'))   # runs and caches
OUT = ROOT / 'results/early_track_velocity_moderndet'

spec = importlib.util.spec_from_file_location('etv', ROOT / 'tools/early_track_velocity.py')
etv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(etv)

DETS = {'pvrcnn': {'none': 'k5_none_pvrcnn', 'policy': 'k5_polA_pvrcnn', 'oracle': 'k5_oracle_pvrcnn'},
        'pointrcnn3c': {'none': 'k5_none_pointrcnn3c', 'policy': 'k5_polA_pointrcnn3c'},
        'voxelrcnn': {'none': 'k5_none_voxelrcnn', 'policy': 'k5_polA_voxelrcnn',
                      'oracle': 'k5_oracle_voxelrcnn'}}
DATA = {'pvrcnn': 'k5_pvrcnn', 'pointrcnn3c': 'k5_pointrcnn3c', 'voxelrcnn': 'k5_voxelrcnn',
        'k5_oracle_pvrcnn': 'k5_oracle_pvrcnn', 'k5_oracle_voxelrcnn': 'k5_oracle_voxelrcnn'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260928)
    summary = {'source': 'tools/early_track_velocity_moderndet.py',
               'reuses': 'tools/early_track_velocity.py (same matching, gate, bootstrap)',
               'gate_m': etv.GATE, 'fast_mps': etv.FAST, 'max_age': etv.MAX_AGE, 'n_boot': etv.NBOOT,
               'detectors': {}}
    for det, runs in DETS.items():
        if not (W / 'runs' / runs['none']).is_dir():
            print('skip', det, '(not run)')
            continue
        R, keys = {}, None
        for name, run in runs.items():
            etv.KIT_GT = str(W / 'data' / DATA.get(run, DATA[det]) / 'gt')
            get, k = etv.kit_run(str(W / 'runs' / run))
            keys = keys or k
            R[name] = etv.records(etv.kit_gt(k), get)
            np.savez_compressed(OUT / ('records_%s_%s.npz' % (det, name)), **R[name])
            print(det, name, len(R[name]['err_v']), 'matches', flush=True)
        summary['detectors'][det] = {'fast_err_v': etv.curves(R, 'err_v', True, rng),
                                    'all_err_v': etv.curves(R, 'err_v', False, rng),
                                    'fast_err_p1': etv.curves(R, 'err_p1', True, rng)}

    # the comparison the experiment is for: how good is the tracker's own velocity
    # at ages 0-2 under each detector, with no grid velocity at all
    cmp = {}
    for a in range(4):
        row = {}
        for det in DETS:
            r = summary['detectors'][det]['fast_err_v']['runs']['none'][a]
            p = summary['detectors'][det]['fast_err_v']['runs']['policy'][a]
            row[det] = {'n': r['n'], 'none_median': r['median'], 'policy_median': p['median'],
                        'policy_minus_none':
                            summary['detectors'][det]['fast_err_v']['policy_minus_none'][a]['diff'],
                        'ci': summary['detectors'][det]['fast_err_v']['policy_minus_none'][a]['ci']}
        cmp[a] = row
    summary['tracker_own_velocity_by_detector'] = cmp
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    print('\nage | detector      |    n | none  | policy | policy-none [95%]')
    for a, row in cmp.items():
        for det, v in row.items():
            n0 = v['none_median']
            n1 = v['policy_median']
            d = v['policy_minus_none']
            print('%3d | %-13s | %4d | %s | %s | %s [%+.2f, %+.2f]' % (
                a, det, v['n'],
                ('%5.2f' % n0) if n0 is not None else '  n/a',
                ('%6.2f' % n1) if n1 is not None else '   n/a',
                ('%+.2f' % d) if d is not None else ' n/a', v['ci'][0], v['ci'][1]))
    print('\nwrote', OUT / 'summary.json')


if __name__ == '__main__':
    main()
