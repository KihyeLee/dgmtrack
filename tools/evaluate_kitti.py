#!/usr/bin/env python3
"""KITTI 3D MOT evaluation of a dogm_track run, with AB3DMOT's evaluator.

The metric code is AB3DMOT's scripts/KITTI/evaluate.py, imported read-only
from the original checkout (configs/paths.json -> original tree). This driver
only (1) writes the submission back in KITTI tracking format and (2) runs the
same threshold loop as AB3DMOT's evaluate(), returning the numbers instead of
printing them.

Writing back: with --box_output det the tracker submits each matched
detection's box unchanged, so every submitted box is found in the detection
JSONL by (sample_token, translation) and the KITTI fields stored there by
tools/kitti_to_dgm.py (2D box, camera 3D box, alpha) are written verbatim.
No inverse transform is involved. The KITTI score is tracking_score.

Keyframes: with --key-every N > 1 only keyframes are submitted; frames are
renumbered k -> (k - first) / N and the ground truth is subset the same way,
so the evaluator sees a contiguous N-times-slower sequence rather than gaps.

Output: results/kitti/eval/<label>/{summary.json, provenance.json, work/}.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITTI_TYPE = {'car': 'Car', 'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_detection_index(det_dir):
    """(sample_token, translation tuple) -> parsed KITTI fields."""
    index = {}
    for path in sorted(Path(det_dir).glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            fr = json.loads(line)
            for b in fr['boxes']:
                tag, bbox, box3d, alpha, raw = b['attribute_name'].split('|')
                assert tag == 'kitti', b['attribute_name']
                index[(fr['sample_token'], tuple(b['translation']))] = {
                    'bbox': [float(v) for v in bbox.split(',')],
                    'box3d': [float(v) for v in box3d.split(',')], 'alpha': float(alpha)}
    return index


def write_kitti_submission(results, index, data_dir, seqmap, key_every):
    """One KITTI tracking txt per sequence; returns counts for provenance."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)
    id_of = {}
    unmatched = 0
    for token, boxes in results.items():
        scene, frame = token.rsplit('_', 1)
        seq = scene.split('-', 1)[1]
        frame = int(frame)
        first = seqmap[seq][0]
        assert (frame - first) % key_every == 0, token
        out_frame = (frame - first) // key_every
        for b in boxes:
            d = index.get((token, tuple(b['translation'])))
            if d is None:
                unmatched += 1
                continue
            tid = id_of.setdefault(b['tracking_id'], len(id_of))
            h, w, l, x, y, z, ry = d['box3d']
            rows[seq].append((out_frame, tid, '%d %d %s 0 0 %r %r %r %r %r %r %r %r %r %r %r %r %r' % (
                out_frame, tid, KITTI_TYPE[b['tracking_name']], d['alpha'], *d['bbox'],
                h, w, l, x, y, z, ry, b['tracking_score'])))
    for seq in seqmap:
        lines = [r[2] for r in sorted(rows.get(seq, []))]
        (data_dir / ('%s.txt' % seq)).write_text('\n'.join(lines) + ('\n' if lines else ''))
    return {'boxes': sum(len(v) for v in rows.values()), 'tracks': len(id_of), 'unmatched': unmatched}


def write_ground_truth(label_src, gt_dir, seqmap, key_every):
    """Label files (subset + renumbered when key_every > 1) and the seqmap the evaluator reads."""
    (gt_dir / 'label').mkdir(parents=True, exist_ok=True)
    lines_map = []
    for seq, (first, last) in seqmap.items():
        keep = []
        for line in (Path(label_src) / ('%s.txt' % seq)).read_text().splitlines():
            f = line.split(' ')
            frame = int(f[0])
            if frame < first or frame >= last or (frame - first) % key_every:
                continue
            f[0] = str((frame - first) // key_every)
            keep.append(' '.join(f))
        (gt_dir / 'label' / ('%s.txt' % seq)).write_text('\n'.join(keep) + '\n')
        # same convention as AB3DMOT's seqmap: last field = number of (key)frames
        n = (last - first + key_every - 1) // key_every
        lines_map.append('%s empty %06d %06d' % (seq, 0, n))
    (gt_dir / 'evaluate_tracking.seqmap.val').write_text('\n'.join(lines_map))


def run_ab3dmot(ev, mailpy, label, work):
    """Mirror of AB3DMOT evaluate() for 3D IoU, returning per-class numbers."""
    out = {}
    # saveToStats() prints through a module-level `mail` that AB3DMOT only
    # defines under __main__.
    ev.mail = mailpy.Mail('')
    cwd = os.getcwd()
    os.chdir(work)
    try:
        for c in ('car', 'pedestrian', 'cyclist'):
            e = ev.trackingEvaluation(t_sha=label, mail=mailpy.Mail(''), cls=c, eval_3diou=True,
                                      eval_2diou=False, num_hypo=1, thres=None)
            if not e.loadTracker():
                continue
            if not e.loadGroundtruth():
                raise ValueError('ground truth not found')
            if len(e.groundtruth) != len(e.tracker):
                raise ValueError('sequence count mismatch')
            dump = open(os.path.join(e.t_path, '../summary_%s_average_eval3D.txt' % c), 'w+')
            meter = ev.stat(t_sha=label, cls=c, suffix='eval3D', dump=dump)
            e.compute3rdPartyMetrics()
            best_mota, best_threshold = 0, -10000
            thresholds, recalls = e.getThresholds(e.scores, e.num_gt)
            for th, rc in zip(thresholds, recalls):
                e.reset()
                e.compute3rdPartyMetrics(th, rc)
                meter.update({'mota': e.MOTA, 'motp': e.MOTP, 'moda': e.MODA, 'modp': e.MODP,
                              'precision': e.precision, 'F1': e.F1, 'fp': e.fp, 'fn': e.fn,
                              'recall': e.recall, 'sMOTA': e.sMOTA})
                if e.MOTA > best_mota:
                    best_threshold, best_mota = th, e.MOTA
                e.saveToStats(dump, th, rc)
            e.reset()
            e.compute3rdPartyMetrics(best_threshold)
            e.saveToStats(dump)
            meter.output()
            meter.print_summary()
            dump.close()
            out[c] = {'sAMOTA': meter.sAMOTA, 'AMOTA': meter.amota, 'AMOTP': meter.amotp,
                      'MOTA': e.MOTA, 'MOTP': e.MOTP, 'recall': e.recall, 'precision': e.precision,
                      'IDS': e.id_switches, 'FRAG': e.fragments, 'FP': e.fp, 'FN': e.fn,
                      'MT': e.MT, 'ML': e.ML, 'n_gt': e.n_gt, 'best_threshold': best_threshold}
    finally:
        os.chdir(cwd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=None, help='tracker run directory holding tracking_results.json')
    ap.add_argument('--kitti-results', default=None,
                    help='instead of --run: a directory of KITTI tracking txt files already in the '
                         'evaluated frame numbering (e.g. AB3DMOT data_0/); copied verbatim')
    ap.add_argument('--data', required=True, help='tools/kitti_to_dgm.py output directory')
    ap.add_argument('--label', required=True)
    ap.add_argument('--det-name', default='pointrcnn')
    ap.add_argument('--ab3dmot', default=None, help='AB3DMOT checkout (default: sibling of the original tree)')
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / 'tools'))
    import workspace_paths
    paths = workspace_paths.load(str(ROOT / 'configs/paths.json'))
    ab = Path(args.ab3dmot) if args.ab3dmot else Path(paths['original']).parents[0] / 'AB3DMOT'
    data = Path(args.data)
    manifest = json.loads((data / 'manifest.json').read_text())
    key_every = manifest['key_every']
    out = ROOT / 'results/kitti/eval' / args.label
    if (out / 'summary.json').exists():
        raise FileExistsError(out)
    work = out / 'work'
    if work.exists():
        shutil.rmtree(work)

    seqmap = {}
    for line in (ab / ('scripts/KITTI/evaluate_tracking.seqmap.%s' % manifest['split'])).read_text().splitlines():
        f = line.split()
        if f:
            seqmap['%04d' % int(f[0])] = (int(f[2]), int(f[3]))

    if bool(args.run) == bool(args.kitti_results):
        ap.error('pass exactly one of --run / --kitti-results')
    if args.kitti_results:
        src = Path(args.kitti_results)
        dst = work / 'results/KITTI' / args.label / 'data_0'
        dst.mkdir(parents=True)
        for seq in seqmap:
            shutil.copy(src / ('%s.txt' % seq), dst / ('%s.txt' % seq))
        submission = src
        written = {'copied_from': str(src), 'unmatched': 0,
                   'boxes': sum(len(open(dst / ('%s.txt' % s)).read().splitlines()) for s in seqmap)}
    else:
        submission = Path(args.run) / 'tracking_results.json'
        results = json.loads(submission.read_text())['results']
        index = load_detection_index(data / 'detections' / args.det_name)
        written = write_kitti_submission(results, index, work / 'results/KITTI' / args.label / 'data_0',
                                         seqmap, key_every)
    if written['unmatched']:
        raise ValueError('%d submitted boxes not found among detections (box_output must be det)'
                         % written['unmatched'])
    write_ground_truth(ab / 'scripts/KITTI/label', work / 'scripts/KITTI', seqmap, key_every)

    # AB3DMOT's @jit helpers (e.g. roty) build arrays from mixed int/float
    # lists, which current numba refuses to type. Running them as plain
    # Python is the same computation; it only costs speed.
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    # AB3DMOT_libs imports the author's toolbox (xinshuo_io); it sits next to AB3DMOT.
    sys.path.insert(0, str(ab.parent / 'Xinshuo_PyToolbox'))
    sys.path.insert(0, str(ab))
    sys.path.insert(0, str(ab / 'scripts/KITTI'))
    import evaluate as ev
    import mailpy
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        metrics = run_ab3dmot(ev, mailpy, args.label, work)
    (out / 'evaluator_stdout.txt').write_text(log.getvalue())
    summary = {'label': args.label, 'metrics_3d_iou': metrics, 'written': written, 'key_every': key_every}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'provenance.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'submission': str(submission),
        'submission_sha256': sha256(submission) if submission.is_file() else None, 'data_manifest': manifest,
        'ab3dmot_evaluate_sha256': sha256(ab / 'scripts/KITTI/evaluate.py'),
        'runner_sha256': sha256(__file__), 'test_only': manifest.get('test_only', False)}, indent=2) + '\n')
    for c, m in metrics.items():
        print('%-10s sAMOTA %.4f AMOTA %.4f MOTA %.4f IDS %d FRAG %d recall %.3f'
              % (c, m['sAMOTA'], m['AMOTA'], m['MOTA'], m['IDS'], m['FRAG'], m['recall']))


if __name__ == '__main__':
    main()
