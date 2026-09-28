#!/usr/bin/env python3
"""Where the KITTI sAMOTA difference between detectors actually comes from.

The 2026-09-28 experiment found the policy's Car sAMOTA gain falling from +0.0146
(PointRCNN) to +0.0023 (PV-RCNN) while IDS and MOTA gains held or grew. sAMOTA is
the mean of sMOTA over 40 fixed recall thresholds, and a threshold the run cannot
reach contributes 0. This reads the evaluator's own per-threshold printout
(evaluator_stdout.txt, first class block = Car) and reports, per arm:

  - sAMOTA as published (sum / 40), to confirm the parse reproduces summary.json
  - the two lowest recall thresholds, where a single operating point can have TP = 0
  - the mean with those excluded, and at how many thresholds the policy is better

This is a diagnostic, NOT a redefinition of the metric: the preregistered verdict
stays the published sAMOTA. Writes results/kitti/SAMOTA_THRESHOLD_DIAG.json.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NTH = 40                  # AB3DMOT sweeps recall 0.025 .. 1.0 in 0.025 steps
ARMS = [('PointRCNN (AB3DMOT-released), thr 0.30', 'k5_none', 'k5_abl_age'),
        ('PointRCNN (own inference), thr 0.1', 'md_k5_none_pointrcnn3c', 'md_k5_polA_pointrcnn3c'),
        ('PointRCNN (own inference), thr 0.30', 'md_k5_none_thr30_pointrcnn3c', 'md_k5_polA_thr30_pointrcnn3c'),
        ('PV-RCNN, thr 0.1', 'md_k5_none_pvrcnn', 'md_k5_polA_pvrcnn'),
        ('PV-RCNN, thr 0.30', 'md_k5_none_thr30_pvrcnn', 'md_k5_polA_thr30_pvrcnn'),
        ('Voxel R-CNN (Car), released thr 0.30', 'md_k5_none_voxelrcnn', 'md_k5_polA_voxelrcnn')]


def car_curve(label):
    """Per-threshold Car rows. The first 'best results' marker ends the Car block."""
    p = ROOT / 'results/kitti/eval' / label / 'evaluator_stdout.txt'
    if not p.is_file():
        return None
    car = p.read_text().split('evaluation: best results with single threshold')[0]
    rows = []
    for m in re.finditer(r'confidence threshold ([\d.]+), recall ([\d.]+)=+\s*\n.*?\n([-\d. ]+)\n', car):
        v = m.group(3).split()
        rows.append({'recall_target': float(m.group(2)), 'conf': float(m.group(1)),
                     'smota': float(v[0]), 'mota': float(v[1]), 'ids': int(v[5]),
                     'tp': int(v[11]), 'fp': int(v[12]), 'fn': int(v[13])})
    return rows


def main():
    out = {'note': __doc__.strip().splitlines()[0], 'n_thresholds': NTH, 'arms': {}}
    print('%-38s %8s %8s %8s | %8s %8s %8s | %s' % (
        'arm', 'sAM none', 'sAM polA', 'diff', 'r>0.05 n', 'r>0.05 p', 'diff', 'polA better'))
    for name, a, b in ARMS:
        A, B = car_curve(a), car_curve(b)
        if not (A and B):
            print('%-38s (missing)' % name)
            continue
        sA, sB = sum(z['smota'] for z in A) / NTH, sum(z['smota'] for z in B) / NTH
        kA = [z for z in A if z['recall_target'] > 0.05]
        kB = [z for z in B if z['recall_target'] > 0.05]
        eA, eB = sum(z['smota'] for z in kA) / (NTH - 2), sum(z['smota'] for z in kB) / (NTH - 2)
        pairs = list(zip(A, B))
        win = sum(1 for x, y in pairs if y['smota'] > x['smota'])
        low = [{'recall_target': x['recall_target'], 'none_smota': x['smota'], 'none_tp': x['tp'],
                'policy_smota': y['smota'], 'policy_tp': y['tp']} for x, y in pairs[:2]]
        out['arms'][name] = {'base_run': a, 'compare_run': b,
                             'samota_published': {'none': sA, 'policy': sB, 'diff': sB - sA},
                             'samota_excluding_two_lowest': {'none': eA, 'policy': eB, 'diff': eB - eA},
                             'thresholds_reached': {'none': len(A), 'policy': len(B)},
                             'policy_better_at': win, 'of': len(pairs),
                             'two_lowest_thresholds': low}
        print('%-38s %8.4f %8.4f %+8.4f | %8.4f %8.4f %+8.4f | %d/%d' % (
            name, sA, sB, sB - sA, eA, eB, eB - eA, win, len(pairs)))
    (ROOT / 'results/kitti/SAMOTA_THRESHOLD_DIAG.json').write_text(json.dumps(out, indent=2) + '\n')
    print('\nwrote results/kitti/SAMOTA_THRESHOLD_DIAG.json')


if __name__ == '__main__':
    main()
