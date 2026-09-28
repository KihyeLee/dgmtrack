#!/usr/bin/env python3
"""Tracker-output velocity error against time since track birth.

Preregistered in results/ablation_20260922/PREREGISTRATION.md ("탄생 이후 시간에 따른 추적기 출력 속도 오차",
2026-09-24 23:35), before any number was computed. No tracker is run: the saved submissions are read.

  nuScenes val (150 scenes, VoxelNeXt boxes, our tracker): velocity removed (a0_none), policy (vox_abl_age),
      oracle (vox_oracle_vel); detector velocity (n_det) as a reference line.
  KITTI val (11 sequences, 2 Hz = key_every 5, Car): k5_none, k5_abl_age, k5_oracle.

Per keyframe, submitted boxes (no score filter) are matched one-to-one to visible GT (num_lidar_pts > 0) of the same
class by Hungarian assignment on BEV centre distance, gated at 2 m. Track age = keyframes since the tracking id
first appeared in the submission (0 = birth), binned 0..10 (10 = 10 or more); both datasets run at 2 Hz, so
time since birth = 0.5 s x age. Metric: median |v_out - v_GT| (2-D, m/s); primary population GT speed >= 3 m/s.
Also recorded (report only): all objects, and the 1 s constant-velocity prediction error p + v*dt against the same
GT instance two keyframes later. Uncertainty: 1,000 scene bootstrap resamples, 95% percentile intervals; the
difference of two runs is paired (same resampled scenes).

Writes results/early_track_velocity/summary.json (+ records_<dataset>_<run>.npz).
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.environ.get('DGM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results/early_track_velocity')
NUS_GT = 'E:/AMSL/ubuntu/src/DGMTrack/cache/gt'                     # read-only original
NUS_RUNS = {'none': 'E:/AMSL/ubuntu/src/DGMTrack/work/results/a0_none/tracking_results.json',
            'policy': ROOT + '/results/tracker_runs/vox_abl_age/tracking_results.json',
            'oracle': ROOT + '/results/tracker_runs/vox_oracle_vel/tracking_results.json',
            'detvel': 'E:/AMSL/ubuntu/src/DGMTrack/work/results/n_det/tracking_results.json'}
KIT_GT = ROOT + '/results/kitti/val_k5/gt'
KIT_RUNS = {'none': ROOT + '/results/kitti/runs/k5_none', 'policy': ROOT + '/results/kitti/runs/k5_abl_age',
            'oracle': ROOT + '/results/kitti/runs/k5_oracle'}
MAX_AGE, GATE, FAST, NBOOT = 10, 2.0, 3.0, 1000
FIELDS = ('scene', 'frame', 'age', 'gt_speed', 'err_v', 'err_p1', 'inst', 'tid')


# ------------------------------------------------------------------ inputs
def nus_gt():
    scenes = {}
    for f in sorted(glob.glob(NUS_GT + '/scene-*.json')):
        g = json.load(open(f))
        toks = sorted(g['frames'], key=lambda t: g['frames'][t]['timestamp_s'])
        scenes[g['scene']] = [(t, g['frames'][t]['timestamp_s'],
                               [o for o in g['frames'][t]['objects'] if o.get('num_lidar_pts', 1) > 0]) for t in toks]
    return scenes


def nus_run(path):
    res = json.load(open(path))['results']
    return lambda tok: [(b['tracking_id'], b['tracking_name'], b['translation'][:2], b['velocity'][:2])
                        for b in res.get(tok, [])]


def kit_gt(key_tokens):
    scenes = {}
    for f in sorted(glob.glob(KIT_GT + '/kitti-*.json')):
        g = json.load(open(f))
        keys = key_tokens.get(g['scene'], [])
        scenes[g['scene']] = [(t, g['frames'][t]['timestamp_s'],
                               [o for o in g['frames'][t]['objects']
                                if o.get('kitti_type') == 'Car' and o.get('num_lidar_pts', 1) > 0])
                              for t in keys if t in g['frames']]
    return scenes


def kit_run(path):
    res, keys = {}, {}
    for f in sorted(glob.glob(path + '/kitti-*.jsonl')):
        for line in open(f):
            d = json.loads(line)
            if not d.get('is_key_frame', True):
                continue
            res[d['sample_token']] = [(b['tracking_id'], b['detection_name'], b['translation'][:2], b['velocity'][:2])
                                      for b in d['boxes'] if b['detection_name'] == 'car']
            keys.setdefault(os.path.basename(f)[:-6], []).append((d['timestamp'], d['sample_token']))
    keys = {s: [t for _, t in sorted(v)] for s, v in keys.items()}
    return (lambda tok: res.get(tok, [])), keys


# ------------------------------------------------------------------ matching
def records(scenes, get, cls_of=lambda o: o['cls']):
    rec = {k: [] for k in FIELDS}
    for scene, frames in scenes.items():
        born = {}
        # GT position per (frame index, instance) for the 1 s prediction check
        gpos = [{o['instance_token']: np.array(o['translation'][:2]) for o in objs} for _, _, objs in frames]
        for fi, (tok, ts, objs) in enumerate(frames):
            boxes = get(tok)
            for tid, *_ in boxes:
                born.setdefault(tid, fi)
            for c in set(cls_of(o) for o in objs) & set(b[1] for b in boxes):
                G = [o for o in objs if cls_of(o) == c]
                B = [b for b in boxes if b[1] == c]
                gp = np.array([o['translation'][:2] for o in G])
                bp = np.array([b[2] for b in B])
                d = np.linalg.norm(gp[:, None, :] - bp[None, :, :], axis=2)
                gi, bi = linear_sum_assignment(np.where(d <= GATE, d, 1e6))
                for i, j in zip(gi, bi):
                    if d[i, j] > GATE:
                        continue
                    o, b = G[i], B[j]
                    if o.get('velocity') is None or b[3] is None:
                        continue
                    vg, vo = np.array(o['velocity'][:2], float), np.array(b[3], float)
                    if not np.all(np.isfinite(vg)):
                        continue
                    p1 = np.nan
                    if fi + 2 < len(frames) and o['instance_token'] in gpos[fi + 2]:
                        dt = frames[fi + 2][1] - ts
                        p1 = float(np.linalg.norm(np.array(b[2]) + vo * dt - gpos[fi + 2][o['instance_token']]))
                    rec['scene'].append(scene)
                    rec['frame'].append(fi)
                    rec['age'].append(fi - born[b[0]])
                    rec['gt_speed'].append(float(np.linalg.norm(vg)))
                    rec['err_v'].append(float(np.linalg.norm(vo - vg)))
                    rec['err_p1'].append(p1)
                    rec['inst'].append(o['instance_token'])
                    rec['tid'].append(str(b[0]))
    return {k: np.array(v) for k, v in rec.items()}


# ------------------------------------------------------------------ statistics
def per_scene(r, metric, fast):
    """{age bin: [array of values per scene]} for the population."""
    m = np.isfinite(r[metric]) & ((r['gt_speed'] >= FAST) if fast else True)
    scenes = sorted(set(r['scene']))
    age = np.minimum(r['age'], MAX_AGE)
    out = {}
    for a in range(MAX_AGE + 1):
        sel = m & (age == a)
        out[a] = [r[metric][sel & (r['scene'] == s)] for s in scenes]
    return scenes, out


def curves(runs, metric, fast, rng):
    """median + 95% CI per run and age, and paired policy - none differences."""
    scene_sets = [set(r['scene']) for r in runs.values()]
    scenes = sorted(set.intersection(*scene_sets))
    idx = rng.integers(0, len(scenes), size=(NBOOT, len(scenes)))
    res = {}
    boot = {}
    for name, r in runs.items():
        _, ps = per_scene(r, metric, fast)
        sc = sorted(set(r['scene']))
        pos = {s: i for i, s in enumerate(sc)}
        order = [pos[s] for s in scenes]
        res[name], boot[name] = [], {}
        for a in range(MAX_AGE + 1):
            arrs = [ps[a][i] for i in order]
            allv = np.concatenate(arrs) if arrs else np.array([])
            bs = np.array([np.median(np.concatenate([arrs[k] for k in row])) if sum(len(arrs[k]) for k in row) else np.nan
                           for row in idx])
            boot[name][a] = bs
            res[name].append({'age': a, 'n': int(allv.size), 'median': float(np.median(allv)) if allv.size else None,
                              'ci': [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
                              if allv.size else None})
    diff = []
    if 'none' in runs and 'policy' in runs:
        for a in range(MAX_AGE + 1):
            dv = boot['policy'][a] - boot['none'][a]
            m0 = res['none'][a]['median']
            m1 = res['policy'][a]['median']
            diff.append({'age': a, 'diff': None if m0 is None or m1 is None else m1 - m0,
                         'ci': [float(np.nanpercentile(dv, 2.5)), float(np.nanpercentile(dv, 97.5))]})
    return {'n_scenes': len(scenes), 'runs': res, 'policy_minus_none': diff}


def verdict(c):
    d, none = c['policy_minus_none'], c['runs']['none']
    h1 = all(d[a]['ci'][1] < 0 for a in (0, 1, 2))
    h2 = all(abs(d[a]['diff']) <= 0.1 * none[a]['median'] for a in range(5, MAX_AGE + 1))
    return {'H1_early_gain': bool(h1), 'H2_no_later_harm': bool(h2),
            'H1_upper_bounds': [d[a]['ci'][1] for a in (0, 1, 2)],
            'H2_rel_diff': [d[a]['diff'] / none[a]['median'] for a in range(5, MAX_AGE + 1)]}


def example(rn, rp, ro, age0_median):
    """Preregistered rule: typical (not best) car, continuous from birth for 10 keyframes in both runs."""
    def chains(r):
        ok, groups = {}, {}
        for n, k in enumerate(zip(r['scene'], r['tid'])):
            groups.setdefault(k, []).append(n)
        for k, rows in groups.items():
            rows = np.array(rows)
            fr, ag, inst = r['frame'][rows], r['age'][rows], r['inst'][rows]
            o = np.argsort(ag)
            fr, ag, inst = fr[o], ag[o], inst[o]
            if len(ag) >= 10 and np.array_equal(ag[:10], np.arange(10)) and len(set(inst[:10])) == 1 and fr[0] > 0:
                ok[(k[0], inst[0])] = k[1]
        return ok
    cn, cp = chains(rn), chains(rp)
    best = None
    for (scene, inst), tid in cn.items():
        if (scene, inst) not in cp:
            continue
        sel = (rn['scene'] == scene) & (rn['tid'] == tid) & (rn['age'] == 0)
        if not sel.any() or rn['gt_speed'][sel][0] < FAST:
            continue
        gap = abs(rn['err_v'][sel][0] - age0_median)
        if best is None or gap < best[0]:
            best = (gap, scene, inst)
    if best is None:
        return None
    _, scene, inst = best

    def series(r):
        sel = (r['scene'] == scene) & (r['inst'] == inst)
        o = np.argsort(r['frame'][sel])
        return {'frame': r['frame'][sel][o].tolist(), 'age': r['age'][sel][o].tolist(),
                'err_v': r['err_v'][sel][o].tolist(), 'gt_speed': r['gt_speed'][sel][o].tolist(),
                'tid': r['tid'][sel][o].tolist()}
    return {'scene': scene, 'instance': inst, 'none': series(rn), 'policy': series(rp),
            'oracle': series(ro) if ro is not None else None}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260924)
    summary = {'preregistration': 'results/ablation_20260922/PREREGISTRATION.md (2026-09-24 23:35)',
               'gate_m': GATE, 'fast_mps': FAST, 'max_age': MAX_AGE, 'n_boot': NBOOT}

    gt = nus_gt()
    R = {}
    for name, path in NUS_RUNS.items():
        R[name] = records(gt, nus_run(path))
        np.savez_compressed(os.path.join(OUT, 'records_nuscenes_%s.npz' % name), **R[name])
        print('nuScenes', name, len(R[name]['err_v']), 'matches', flush=True)
    summary['nuscenes'] = {'fast_err_v': curves(R, 'err_v', True, rng),
                           'all_err_v': curves(R, 'err_v', False, rng),
                           'fast_err_p1': curves(R, 'err_p1', True, rng)}
    summary['nuscenes']['verdict'] = verdict(summary['nuscenes']['fast_err_v'])
    a0 = summary['nuscenes']['fast_err_v']['runs']['none'][0]['median']
    summary['nuscenes']['example'] = example(R['none'], R['policy'], R['oracle'], a0)

    K, keys = {}, None
    for name, path in KIT_RUNS.items():
        get, k = kit_run(path)
        keys = keys or k
        K[name] = records(kit_gt(k), get)
        np.savez_compressed(os.path.join(OUT, 'records_kitti_%s.npz' % name), **K[name])
        print('KITTI', name, len(K[name]['err_v']), 'matches', flush=True)
    summary['kitti'] = {'fast_err_v': curves(K, 'err_v', True, rng),
                        'all_err_v': curves(K, 'err_v', False, rng),
                        'fast_err_p1': curves(K, 'err_p1', True, rng)}
    summary['kitti']['verdict_report_only'] = verdict(summary['kitti']['fast_err_v'])
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print('verdict nuScenes', summary['nuscenes']['verdict'])
    print('verdict KITTI (report only)', summary['kitti']['verdict_report_only'])


if __name__ == '__main__':
    main()
