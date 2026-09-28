#!/usr/bin/env python3
"""Collect the KITTI runs into one table and apply the preregistered verdict.

Verdict rule (results/ablation_20260922/PREREGISTRATION.md, "추가 사전 등록: KITTI"):
primary protocol key-every 5, class Car; baseline -> proposal must raise sAMOTA
AND AMOTA, raise MOTA, and lower IDS. The centre-distance bootstrap is reported
but not used for the verdict. Writes results/kitti/SUMMARY.json and
docs/KITTI_RESULTS_<date>.md (generated - do not edit by hand).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['k5_none', 'k5_abl_age', 'k5_oracle', 'k1_none', 'k1_abl_age']
CLASSES = ['car', 'pedestrian', 'cyclist']


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.is_file() else None


def verdict(base, comp):
    b, c = base['metrics_3d_iou']['car'], comp['metrics_3d_iou']['car']
    checks = {'sAMOTA_up': c['sAMOTA'] > b['sAMOTA'], 'AMOTA_up': c['AMOTA'] > b['AMOTA'],
              'MOTA_up': c['MOTA'] > b['MOTA'], 'IDS_down': c['IDS'] < b['IDS']}
    return checks, all(checks.values())


def main():
    ev = {r: load('results/kitti/eval/%s/summary.json' % r) for r in RUNS}
    boot = {k: load('results/kitti/bootstrap_%s.json' % k)
            for k in ('k5_none_vs_k5_abl_age', 'k1_none_vs_k1_abl_age', 'k5_none_vs_k5_oracle')}
    down = {k: load('results/kitti/prediction_diagnostic_%s/summary.json' % k) for k in ('k5_abl_age', 'k1_abl_age')}
    out = {'created_utc': datetime.now(timezone.utc).isoformat(), 'eval': ev, 'bootstrap': boot}
    lines = ['# KITTI 결과 (자동 생성 — 손으로 고치지 않는다)', '',
             '생성: `tools/summarize_kitti.py`. 사전 등록: `results/ablation_20260922/PREREGISTRATION.md` "KITTI" 절.', '',
             '## AB3DMOT 3D MOT 평가 (3D IoU 0.25)', '',
             '| run | 클래스 | sAMOTA | AMOTA | MOTA | IDS | FRAG | recall |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in RUNS:
        if not ev[r]:
            lines.append('| %s | (없음) | | | | | | |' % r)
            continue
        for c in CLASSES:
            m = ev[r]['metrics_3d_iou'].get(c)
            if m:
                lines.append('| %s | %s | %.4f | %.4f | %.4f | %d | %d | %.3f |'
                             % (r, c, m['sAMOTA'], m['AMOTA'], m['MOTA'], m['IDS'], m['FRAG'], m['recall']))
    lines += ['', '## 판정 (주 분석 key-every 5, Car)', '']
    for base, comp, tag in (('k5_none', 'k5_abl_age', '주 분석 (2 Hz)'), ('k1_none', 'k1_abl_age', '보조 분석 (10 Hz)')):
        if ev[base] and ev[comp]:
            checks, ok = verdict(ev[base], ev[comp])
            out.setdefault('verdict', {})[tag] = {'checks': checks, 'reproduced': ok}
            b, c = ev[base]['metrics_3d_iou']['car'], ev[comp]['metrics_3d_iou']['car']
            lines.append('- **%s**: sAMOTA %+.4f, AMOTA %+.4f, MOTA %+.4f, IDS %+d → %s'
                         % (tag, c['sAMOTA'] - b['sAMOTA'], c['AMOTA'] - b['AMOTA'], c['MOTA'] - b['MOTA'],
                            c['IDS'] - b['IDS'], '**재현됨**' if ok else '**재현 안 됨** (%s)'
                            % ', '.join(k for k, v in checks.items() if not v)))
    lines += ['', '판정은 주 분석만 쓴다. 보조 분석은 사전 등록대로 보고만 한다.', '',
              '## 구간 (nuScenes식 중심거리 AMOTA, 11시퀀스 쌍대응 4,000회 — 판정에 쓰지 않음)', '',
              '| 비교 | AMOTA | MOTA | IDS |', '|---|---|---|---|']
    for k, v in boot.items():
        if v:
            d = v['paired_difference']
            lines.append('| %s | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] | %+.0f [%+.0f, %+.0f] |' % (
                k, d['amota']['observed'], *d['amota']['ci95'], d['mota']['observed'], *d['mota']['ci95'],
                d['ids']['observed'], *d['ids']['ci95']))
    lines += ['', '## 하류 1초 ADE (≥3 m/s, history CV)', '']
    for k, s in down.items():
        if not s:
            continue
        for row in s['rows']:
            if row['predictor'] == 'history_cv' and row['horizon_s'] == 1 and row['speed_min_mps'] in (0, 3):
                names = list(row['results'])
                b, c = row['results'][names[0]], row['results'][names[1]]
                d = row['flow_minus_none']['ade']
                lines.append('- %s, ≥%g m/s (n=%d): %.3f → %.3f (%+.1f%%) [%+.3f, %+.3f], cold %.1f → %.1f%%'
                             % (k, row['speed_min_mps'], row['n'], b['ade'], c['ade'],
                                100 * d['delta'] / b['ade'], *d['ci95'], b['cold_pct'], c['cold_pct']))
    (ROOT / 'results/kitti/SUMMARY.json').write_text(json.dumps(out, indent=2) + '\n')
    doc = ROOT / ('docs/KITTI_RESULTS_%s.md' % datetime.now().strftime('%Y%m%d'))
    doc.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
