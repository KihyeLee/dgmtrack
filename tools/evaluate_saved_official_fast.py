#!/usr/bin/env python3
"""Official nuScenes tracking evaluation with the per-threshold work run in parallel.

Everything that produces numbers is the unmodified devkit: loading, filtering,
track creation, `TrackingEvaluation.compute_thresholds`,
`TrackingEvaluation.accumulate_threshold` and motmetrics `compute`. This driver
only distributes the (class, threshold) accumulations over worker processes
and then assembles the TrackingMetricData and TrackingMetrics exactly as
`TrackingEvaluation.accumulate()` and `TrackingEval.evaluate()` do. The output
directory layout and provenance match `tools/evaluate_saved_official.py`, with
`evaluator: parallel-thresholds` recorded. `--validate-against <label>` checks
the result against an existing serial evaluation to full float precision.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP
from nuscenes.eval.tracking.data_classes import TrackingMetricData, TrackingMetricDataList, TrackingMetrics
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.eval.tracking.utils import create_motmetrics

ROOT = Path(__file__).resolve().parents[1]
_EVALUATIONS = {}
_CAPTURE = {'enabled': False, 'last': None}
COUNT_NAMES = ['gt', 'tp', 'fn', 'fp', 'ids', 'detections', 'distance_sum', 'frames']
_ORIGINAL_MERGE = MOTAccumulatorCustom.merge_event_dataframes


def _merge_capturing(dfs, *args, **kwargs):
    _CAPTURE['last'] = list(dfs)
    return _ORIGINAL_MERGE(dfs, *args, **kwargs)


def scene_counts(accumulator):
    """Per-scene counts with motmetrics' definitions (same as tools/accumulate_scene_counts.py)."""
    events = accumulator.events
    if len(events) == 0:
        return np.zeros(len(COUNT_NAMES))
    noraw = events[events.Type != 'RAW']
    types = noraw.Type.values
    matches, switches = types == 'MATCH', types == 'SWITCH'
    detected = matches | switches
    return np.array([float(matches.sum() + switches.sum() + (types == 'MISS').sum()), float(matches.sum()),
                     float((types == 'MISS').sum()), float((types == 'FP').sum()), float(switches.sum()),
                     float(detected.sum()), float(np.nansum(noraw.D.values[detected])),
                     float(events.index.get_level_values(0).nunique())])


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_job(job):
    """One (class, threshold) accumulation, run in a forked worker."""
    class_name, threshold = job
    evaluation = _EVALUATIONS[class_name]
    _CAPTURE['last'] = None
    acc, _ = evaluation.accumulate_threshold(threshold)
    handler = create_motmetrics()
    summary = handler.compute(acc, metrics=MOT_METRIC_MAP.keys(), name=evaluation.name_gen(threshold))
    counts = None
    if _CAPTURE['enabled'] and _CAPTURE['last'] is not None:
        counts = np.stack([scene_counts(a) for a in _CAPTURE['last']])
    return class_name, threshold, summary, counts


def accumulate_parallel(evaluations, workers):
    """Mirror of TrackingEvaluation.accumulate() with the threshold loop parallelised."""
    prepared = {}
    jobs = []
    for class_name, evaluation in evaluations.items():
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in evaluation.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            prepared[class_name] = None
            continue
        thresholds, recalls = evaluation.compute_thresholds(gt_box_count)
        prepared[class_name] = (thresholds, recalls, gt_box_count, gt_track_ids)
        for t, threshold in enumerate(thresholds):
            if np.isnan(threshold) or threshold in thresholds[:t]:
                continue
            jobs.append((class_name, threshold))

    results = {}
    counts_by_job = {}
    with multiprocessing.Pool(workers) as pool:
        for class_name, threshold, summary, counts in pool.imap_unordered(_threshold_job, jobs):
            results[(class_name, threshold)] = summary
            counts_by_job[(class_name, threshold)] = counts

    md_list = TrackingMetricDataList()
    for class_name, evaluation in evaluations.items():
        md = TrackingMetricData()
        if prepared[class_name] is None:
            md_list.set(class_name, md)
            continue
        thresholds, recalls, gt_box_count, gt_track_ids = prepared[class_name]
        md.confidence = thresholds
        md.recall_hypo = recalls
        thresh_metrics = []
        for t, threshold in enumerate(thresholds):
            if np.isnan(threshold) or threshold in thresholds[:t]:
                continue
            thresh_metrics.append(results[(class_name, threshold)])
        summary = pandas.concat(thresh_metrics) if thresh_metrics else []
        unachieved_thresholds = np.array([t for t in thresholds if np.isnan(t)])
        num_unachieved_thresholds = len(unachieved_thresholds)
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        num_duplicate_thresholds = len(valid_thresholds) - len(np.unique(valid_thresholds))
        assert num_unachieved_thresholds + num_duplicate_thresholds + len(thresh_metrics) == evaluation.num_thresholds
        rep_counts = [np.sum(thresholds == t) for t in np.unique(valid_thresholds)]
        for mot_name, metric_name in MOT_METRIC_MAP.items():
            if metric_name == '':
                continue
            if len(thresh_metrics) == 0:
                worst = evaluation.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == 'ml':
                        worst = len(gt_track_ids)
                    elif metric_name in ['gt', 'fn']:
                        worst = gt_box_count
                    elif metric_name in ['fp', 'ids', 'frag']:
                        worst = np.nan
                    else:
                        raise NotImplementedError
                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)
                assert len(rep_counts) == len(values)
                values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])
                all_values = [np.nan] * num_unachieved_thresholds
                all_values.extend(values)
            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)
        md_list.set(class_name, md)
    return md_list, prepared, counts_by_job


def aggregate(cfg, md_list):
    """Mirror of TrackingEval.evaluate() step 2."""
    metrics = TrackingMetrics(cfg)
    for class_name in cfg.class_names:
        md = md_list[class_name]
        best_thresh_idx = None if np.all(np.isnan(md.mota)) else np.nanargmax(md.mota)
        if best_thresh_idx is not None:
            for metric_name in MOT_METRIC_MAP.values():
                if metric_name == '':
                    continue
                metrics.add_label_metric(metric_name, class_name, md.get_metric(metric_name)[best_thresh_idx])
        for metric_name in AVG_METRIC_MAP.keys():
            values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
            assert len(values) == TrackingMetricData.nelem
            if np.all(np.isnan(values)):
                value = np.nan
            else:
                values[np.isnan(values)] = cfg.metric_worst[metric_name]
                value = float(np.nanmean(values))
            metrics.add_label_metric(metric_name, class_name, value)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', help='directory name under the original work/results tree')
    parser.add_argument('--submission', help='submission file in this workspace; needs --label')
    parser.add_argument('--label')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--validate-against', help='existing label whose serial result must be matched exactly')
    parser.add_argument('--scene-counts', action='store_true',
                        help='also record per-scene counts at every threshold (results/scene_counts/<label>/)')
    args = parser.parse_args()
    if bool(args.run) == bool(args.submission):
        parser.error('pass either --run or --submission')
    name = args.run or args.label
    if not name or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        parser.error('bad or missing label')
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    submission = (Path(paths['original']) / 'work/results' / args.run / 'tracking_results.json' if args.run
                  else Path(args.submission).resolve())
    if not submission.is_file():
        raise FileNotFoundError(submission)
    output = ROOT / 'results/official40' / name
    if output.exists() and not args.validate_against:
        previous = output / 'provenance.json'
        if previous.is_file() and json.loads(previous.read_text()).get('status') == 'complete':
            print('EXISTING EVALUATION (complete): ' + str(output), flush=True)
            return
        raise FileExistsError('Refusing to overwrite: ' + str(output))
    if args.validate_against:
        output = ROOT / 'results/official40' / (name + '_fastcheck')
        if output.exists():
            raise FileExistsError(output)
    output.mkdir(parents=True)

    config = config_factory('tracking_nips_2019')
    assert config.num_thresholds == 40
    submission_hash = sha256(submission)
    provenance = {
        'run': name, 'submission_kind': 'inherited' if args.run else 'workspace',
        'evaluator': 'parallel-thresholds (devkit functions unchanged; see tools/evaluate_saved_official_fast.py)',
        'workers': args.workers,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'submission': str(submission), 'submission_sha256': submission_hash,
        'dataroot': paths['nuscenes'], 'version': 'v1.0-trainval', 'eval_set': 'val',
        'devkit_version': importlib.metadata.version('nuscenes-devkit'),
        'package_versions': {n: importlib.metadata.version(n) for n in ('motmetrics', 'numpy', 'pandas', 'scipy')},
        'evaluator_sha256': sha256(Path(inspect.getfile(TrackingEval))),
        'algo_sha256': sha256(Path(inspect.getfile(TrackingEvaluation))),
        'runner_sha256': sha256(Path(__file__)),
        'config': config.serialize(), 'render_curves': False, 'status': 'running', 'process_id': os.getpid(),
    }
    (output / 'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    start = time.monotonic()
    print('OFFICIAL EVALUATION START (parallel thresholds): ' + name, flush=True)
    try:
        evaluation = TrackingEval(config=config, result_path=str(submission), eval_set='val',
                                  output_dir=str(output), nusc_version='v1.0-trainval',
                                  nusc_dataroot=paths['nuscenes'], verbose=False, render_classes=[])
        TrackingMetricData.nelem = config.num_thresholds
        for class_name in config.class_names:
            _EVALUATIONS[class_name] = TrackingEvaluation(
                evaluation.tracks_gt, evaluation.tracks_pred, class_name, config.dist_fcn_callable,
                config.dist_th_tp, config.min_recall, num_thresholds=config.num_thresholds,
                metric_worst=config.metric_worst, verbose=False, output_dir=str(output), render_classes=[])
        if args.scene_counts:
            _CAPTURE['enabled'] = True
            MOTAccumulatorCustom.merge_event_dataframes = staticmethod(_merge_capturing)
        md_list, prepared, counts_by_job = accumulate_parallel(_EVALUATIONS, args.workers)
        if args.scene_counts:
            scenes = list(evaluation.tracks_gt.keys())
            classes = list(config.class_names)
            counts = np.zeros((len(classes), config.num_thresholds, len(scenes), len(COUNT_NAMES)))
            thresholds = np.full((len(classes), config.num_thresholds), np.nan)
            recalls = np.full((len(classes), config.num_thresholds), np.nan)
            for ci, class_name in enumerate(classes):
                if prepared[class_name] is None:
                    continue
                th, rc = prepared[class_name][0], prepared[class_name][1]
                thresholds[ci], recalls[ci] = th, rc
                for index, threshold in enumerate(th):
                    if np.isnan(threshold):
                        continue
                    counts[ci, index] = counts_by_job[(class_name, threshold)]
            sc_dir = ROOT / 'results/scene_counts' / (name + ('_fastcheck' if args.validate_against else ''))
            sc_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sc_dir / 'scene_counts.npz', counts=counts, thresholds=thresholds, recalls=recalls,
                                classes=np.array(classes), scenes=np.array(scenes), count_names=np.array(COUNT_NAMES))
            (sc_dir / 'provenance.json').write_text(json.dumps({
                'run': name, 'submission': str(submission), 'submission_sha256': submission_hash,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'source': 'evaluate_saved_official_fast.py --scene-counts',
                'count_names': COUNT_NAMES, 'consistency_problems': []}, indent=2) + '\n')
        metrics = aggregate(config, md_list)
        metrics.add_runtime(time.monotonic() - start)
        summary = metrics.serialize()
        summary['meta'] = evaluation.meta.copy()
        (output / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'metrics_details.json').write_text(json.dumps(md_list.serialize(), indent=2))
    except BaseException as error:
        provenance.update(status='failed', error=repr(error), wall_time_seconds=time.monotonic() - start)
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        raise
    provenance.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_time_seconds=time.monotonic() - start)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('amota', 'amotp', 'mota', 'ids', 'frag', 'fn', 'fp')}), flush=True)

    if args.validate_against:
        reference = ROOT / 'results/official40' / args.validate_against
        ref_summary = json.loads((reference / 'metrics_summary.json').read_text())
        ref_details = json.loads((reference / 'metrics_details.json').read_text())
        new_details = json.loads((output / 'metrics_details.json').read_text())
        mismatches = [k for k in ('amota', 'amotp', 'mota', 'motp', 'recall', 'ids', 'frag', 'fp', 'fn', 'tp',
                                  'tid', 'lgd', 'mt', 'ml', 'faf', 'motar', 'gt')
                      if not (ref_summary[k] == summary[k] or
                              (isinstance(ref_summary[k], float) and np.isnan(ref_summary[k]) and np.isnan(summary[k])))]
        detail_mismatch = 0
        for cls in ref_details:
            for metric, values in ref_details[cls].items():
                a = np.array(values, dtype=float); b = np.array(new_details[cls][metric], dtype=float)
                if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0):
                    detail_mismatch += 1
        verdict = {'validated_against': args.validate_against, 'summary_mismatches': mismatches,
                   'detail_array_mismatches': detail_mismatch, 'identical': not mismatches and detail_mismatch == 0}
        (output / 'validation.json').write_text(json.dumps(verdict, indent=2) + '\n')
        print(json.dumps(verdict), flush=True)
    print('OFFICIAL EVALUATION COMPLETE: ' + name, flush=True)


if __name__ == '__main__':
    main()
