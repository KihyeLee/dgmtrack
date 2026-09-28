#!/usr/bin/env python3
"""Scene-level paired bootstrap of the official 40-threshold metrics.

AMOTA and AMOTP are recomputed from per-scene counts; IDS and MOTA are read at
each class's best-MOTA threshold, selected once on the full validation set.

Reads the per-scene counts written by `tools/accumulate_scene_counts.py` for two
runs and resamples scenes with replacement, using the same resampled scenes for
both runs. Scenes are the independent unit: a scene's frames are not
independent of each other.

The score thresholds of each run stay fixed at the values the official
evaluator selected on the full validation set; they are not re-selected inside
each resample. The resulting interval therefore describes the variation of the
metric at a fixed operating point, not the variation of an evaluation that also
re-tunes its thresholds. Both are legitimate estimands and they answer different
questions; this one matches how the numbers in a results table are read.

The interval is a percentile interval of the paired difference. `P(improve)` is
the share of resamples in which the difference favours the second run. It is a
resampling frequency, not a posterior probability that the effect is real.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORST_MOTAR = 0.0   # cfg.metric_worst['amota']
WORST_MOTP = 2.0    # cfg.metric_worst['amotp']


def load(run):
    path = ROOT / 'results/scene_counts' / run / 'scene_counts.npz'
    data = np.load(path, allow_pickle=False)
    return {
        'counts': data['counts'],          # (classes, thresholds, scenes, counts)
        'thresholds': data['thresholds'],
        'recalls': data['recalls'],
        'classes': [str(c) for c in data['classes']],
        'scenes': [str(s) for s in data['scenes']],
        'count_names': [str(c) for c in data['count_names']],
    }


def metrics_from_counts(totals, thresholds, count_names):
    """AMOTA and AMOTP per class from counts already summed over scenes."""
    gt = totals[..., count_names.index('gt')]
    tp = totals[..., count_names.index('tp')]
    fn = totals[..., count_names.index('fn')]
    fp = totals[..., count_names.index('fp')]
    ids = totals[..., count_names.index('ids')]
    detections = totals[..., count_names.index('detections')]
    distance_sum = totals[..., count_names.index('distance_sum')]

    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.where(gt > 0, tp / gt, np.nan)
        denominator = recall * gt
        motar = 1.0 - ((fn + ids + fp) - (1.0 - recall) * gt) / denominator
        motar = np.maximum(0.0, motar)
        motar = np.where(denominator > 0, motar, np.nan)
        motp = np.where(detections > 0, distance_sum / detections, np.nan)

    unachieved = np.isnan(thresholds)
    motar = np.where(unachieved, np.nan, motar)
    motp = np.where(unachieved, np.nan, motp)

    def average(values, worst):
        all_nan = np.all(np.isnan(values), axis=-1)
        filled = np.where(np.isnan(values), worst, values)
        result = np.mean(filled, axis=-1)
        return np.where(all_nan, np.nan, result)

    return average(motar, WORST_MOTAR), average(motp, WORST_MOTP)


def operating_points(totals, thresholds, count_names):
    """Per-class index of the best-MOTA threshold on counts summed over scenes.

    The official evaluator reads MOTA, IDS and the other point metrics at the
    threshold with the highest MOTA of each class. This is selected once on the
    full validation set and held fixed inside the bootstrap.
    """
    gt = totals[..., count_names.index('gt')]
    fn = totals[..., count_names.index('fn')]
    fp = totals[..., count_names.index('fp')]
    ids = totals[..., count_names.index('ids')]
    with np.errstate(divide='ignore', invalid='ignore'):
        mota = np.maximum(0.0, 1.0 - (fn + ids + fp) / gt)
    mota = np.where(np.isnan(thresholds) | (gt == 0), np.nan, mota)
    points = []
    for row in mota:
        points.append(None if np.all(np.isnan(row)) else int(np.nanargmax(row)))
    return points


def point_metrics(totals, points, count_names):
    """IDS summed over classes and mean MOTA at fixed per-class operating points."""
    ids_total = 0.0
    motas = []
    for class_index, point in enumerate(points):
        if point is None:
            continue
        row = totals[class_index, point]
        gt = row[count_names.index('gt')]
        errors = row[count_names.index('fn')] + row[count_names.index('ids')] + row[count_names.index('fp')]
        ids_total += row[count_names.index('ids')]
        motas.append(max(0.0, 1.0 - errors / gt) if gt > 0 else np.nan)
    return ids_total, float(np.nanmean(motas)) if motas else np.nan


def overall(per_class):
    with np.errstate(invalid='ignore'):
        return np.nanmean(per_class, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True)
    parser.add_argument('--compare', required=True)
    parser.add_argument('--resamples', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    base = load(args.base)
    compare = load(args.compare)
    if set(base['scenes']) != set(compare['scenes']) or base['classes'] != compare['classes']:
        raise ValueError('The two runs do not share scenes and classes')
    # The evaluator orders scenes by the submission's sample order; align them.
    order = [compare['scenes'].index(scene) for scene in base['scenes']]
    compare['counts'] = compare['counts'][:, :, order, :]
    compare['scenes'] = list(base['scenes'])
    scenes = base['scenes']
    classes = base['classes']
    names = base['count_names']

    point = {}
    points = {}
    for label, data in (('base', base), ('compare', compare)):
        totals = data['counts'].sum(axis=2)
        amota, amotp = metrics_from_counts(totals, data['thresholds'], names)
        points[label] = operating_points(totals, data['thresholds'], names)
        ids_total, mota = point_metrics(totals, points[label], names)
        point[label] = {
            'amota': float(overall(amota)), 'amotp': float(overall(amotp)),
            'ids': float(ids_total), 'mota': float(mota),
            'operating_point_index': points[label],
            'class_amota': {c: float(v) for c, v in zip(classes, amota)},
            'class_amotp': {c: float(v) for c, v in zip(classes, amotp)},
        }

    rng = np.random.default_rng(args.seed)
    n = len(scenes)
    differences = {name: np.empty(args.resamples) for name in ('amota', 'amotp', 'ids', 'mota')}
    for i in range(args.resamples):
        draw = rng.integers(0, n, size=n)
        weights = np.bincount(draw, minlength=n).astype(float)
        results = {}
        for label, data in (('base', base), ('compare', compare)):
            totals = np.einsum('ctsk,s->ctk', data['counts'], weights)
            amota, amotp = metrics_from_counts(totals, data['thresholds'], names)
            ids_total, mota = point_metrics(totals, points[label], names)
            results[label] = (overall(amota), overall(amotp), ids_total, mota)
        for index, name in enumerate(('amota', 'amotp', 'ids', 'mota')):
            differences[name][i] = results['compare'][index] - results['base'][index]

    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'base_run': args.base,
        'compare_run': args.compare,
        'scenes': len(scenes),
        'resamples': args.resamples,
        'seed': args.seed,
        'thresholds_fixed_at_full_sample_selection': True,
        'point_estimates': point,
        'paired_difference': {},
    }
    for metric in ('amota', 'amotp', 'ids', 'mota'):
        values = differences[metric]
        observed = point['compare'][metric] - point['base'][metric]
        better = values < 0 if metric in ('amotp', 'ids') else values > 0
        report['paired_difference'][metric] = {
            'observed': observed,
            'ci95': [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            'mean_of_resamples': float(values.mean()),
            'p_improve': float(better.mean()),
        }

    out = Path(args.out) if args.out else (ROOT / 'results/official40'
                                           / f'bootstrap_{args.base}_vs_{args.compare}.json')
    out.write_text(json.dumps(report, indent=2) + '\n')
    for label in ('base', 'compare'):
        print(label, {k: round(point[label][k], 6) for k in ('amota', 'amotp', 'ids', 'mota')})
    print(json.dumps(report['paired_difference'], indent=2))
    print('wrote ' + str(out))


if __name__ == '__main__':
    main()
