#!/usr/bin/env python3
"""HOTA / AssA / DetA / IDF1 on KITTI val with TrackEval (the reference implementation).

KITTI ranks tracking by HOTA in the 2D image plane for Car and Pedestrian. Each
run evaluated by tools/evaluate_kitti.py already has, under
results/kitti/eval/<label>/work/, the ground truth in the evaluated frame
numbering (renumbered for 2 Hz) and the submission in KITTI format with the
2D boxes of the matched detections. This tool hands exactly those files to
TrackEval's Kitti2DBox dataset (same truncation / occlusion / height filters as
the official devkit) and collects HOTA, AssA, DetA, LocA, IDF1, MOTA, IDSW.

TrackEval (vendor/TrackEval, upstream 12c8791) still uses np.float / np.int /
np.bool, removed in numpy 1.24. They were plain aliases of the builtins, so the
aliases are restored before import; no TrackEval file is modified.

usage: tools/hota_kitti.py --runs k5_none k5_abl_age ... --out results/kitti/hota_summary.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for alias, builtin in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, alias):
        setattr(np, alias, builtin)
sys.path.insert(0, str(ROOT / 'vendor/TrackEval'))
import trackeval  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    summary = {}
    for run in args.runs:
        work = ROOT / 'results/kitti/eval' / run / 'work'
        te = ROOT / 'results/kitti/hota' / run
        if te.exists():
            shutil.rmtree(te)
        gt = te / 'gt'
        (gt / 'label_02').mkdir(parents=True)
        for f in (work / 'scripts/KITTI/label').glob('*.txt'):
            shutil.copy(f, gt / 'label_02' / f.name)
        shutil.copy(work / 'scripts/KITTI/evaluate_tracking.seqmap.val', gt / 'evaluate_tracking.seqmap.training')
        trk = te / 'trackers' / run / 'data'
        trk.mkdir(parents=True)
        for f in (work / 'results/KITTI' / run / 'data_0').glob('*.txt'):
            shutil.copy(f, trk / f.name)
        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({'USE_PARALLEL': False, 'PRINT_RESULTS': False, 'PRINT_CONFIG': False,
                         'TIME_PROGRESS': False, 'OUTPUT_SUMMARY': True, 'OUTPUT_DETAILED': True,
                         'PLOT_CURVES': False})
        ds_cfg = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
        ds_cfg.update({'GT_FOLDER': str(gt), 'TRACKERS_FOLDER': str(te / 'trackers'), 'TRACKERS_TO_EVAL': [run],
                       'SPLIT_TO_EVAL': 'training', 'PRINT_CONFIG': False})
        evaluator = trackeval.Evaluator(eval_cfg)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        res, _ = evaluator.evaluate([trackeval.datasets.Kitti2DBox(ds_cfg)], metrics)
        r = res['Kitti2DBox'][run]
        out = {}
        for cls in ('car', 'pedestrian'):
            c = r['COMBINED_SEQ'][cls]
            out[cls] = {k: float(np.mean(c['HOTA'][k])) for k in ('HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'AssRe')}
            out[cls].update(IDF1=float(c['Identity']['IDF1']), MOTA=float(c['CLEAR']['MOTA']),
                            IDSW=int(c['CLEAR']['IDSW']), Frag=int(c['CLEAR']['Frag']))
            out[cls]['per_seq_HOTA'] = {s: float(np.mean(v[cls]['HOTA']['HOTA'])) for s, v in r.items()
                                        if s != 'COMBINED_SEQ'}
        summary[run] = out
        print('%-18s car HOTA %.4f AssA %.4f DetA %.4f IDF1 %.4f IDSW %4d | ped HOTA %.4f AssA %.4f IDSW %4d'
              % (run, out['car']['HOTA'], out['car']['AssA'], out['car']['DetA'], out['car']['IDF1'],
                 out['car']['IDSW'], out['pedestrian']['HOTA'], out['pedestrian']['AssA'], out['pedestrian']['IDSW']),
              flush=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
