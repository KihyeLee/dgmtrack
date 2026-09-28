#!/usr/bin/env bash
# MCTrack on nuScenes, as preregistered in results/ablation_20260922/PREREGISTRATION.md
# ("사전 등록 — MCTrack nuScenes 이식, 2026-09-28").
#
# One VoxelNeXt detection cache, three conditions that differ only in the velocity
# information: the detector's own velocity, velocity removed, and velocity removed
# plus the grid policy of Section III-D. Every tracker value is MCTrack's released
# nuScenes configuration; nothing is tuned here.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
P=${DGM_PYTHON:-/home/mt-pc-0099/.venvs/dgmtrack-ubuntu/bin/python}
W=/media/mt-pc-0099/NVMe4TB/dgmtrack_work/mctrack_nusc
M=vendor/MCTrack
NUSC="$($P -c "import json;print(json.load(open('configs/paths.json'))['nuscenes'])")"
LOG=logs/mctrack_nusc.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
export PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
mkdir -p logs "$W/dets" "$W/base"

for v in detector none grid; do
  [ -f "$W/det_$v.json" ] || { say "FAILED: $W/det_$v.json missing"; exit 2; }
  mkdir -p "$W/dets/vx_$v"
  [ -f "$W/dets/vx_$v/val.json" ] || cp "$W/det_$v.json" "$W/dets/vx_$v/val.json"
done

# ---- BaseVersion ------------------------------------------------------------
for v in detector none grid; do
  if [ ! -f "$W/base/vx_$v/val.json" ]; then
    say "baseversion vx_$v"
    ( cd "$M" && "$P" preprocess/convert_nuscenes.py --raw_data_path "$NUSC" \
        --dets_path "$W/dets" --save_path "$W/base" --detector "vx_$v" --split val ) >> "$LOG" 2>&1 \
      || { say "FAILED baseversion vx_$v"; exit 2; }
  fi
done

# ---- tracking ---------------------------------------------------------------
run() {   # label, detector dir, MCTRACK_GRID
  local label=$1 det=$2 grid=$3
  local out="$W/runs/$label"
  [ -f "$out/results.json" ] && { say "track $label: exists"; return; }
  mkdir -p "$out"
  local cfg="$W/cfg_$label.yaml"
  # $4 = "pos" -> the CV pose filter measures position only (M=2), which is what
  # MCTrack itself uses where the detector reports no velocity (its KITTI config).
  # Feeding a zero velocity into the M=4 measurement instead would not remove the
  # velocity, it would observe it as zero and pull the estimate toward zero.
  "$P" - "$M/config/nuscenes.yaml" "$cfg" "$W/base/" "$det" "$out/" "${4:-full}" <<'PY'
import re, sys
src, dst, root, det, save, mode = sys.argv[1:7]
t = open(src).read()
t = re.sub(r'^DETECTIONS_ROOT:.*$', 'DETECTIONS_ROOT: "%s"' % root, t, flags=re.M)
t = re.sub(r'^DETECTOR:.*$', 'DETECTOR: %s' % det, t, flags=re.M)
t = re.sub(r'^SAVE_PATH:.*$', 'SAVE_PATH: "%s"' % save, t, flags=re.M)
if mode == 'pos':
    out, in_cv = [], False
    for ln in t.split('\n'):
        if re.match(r'^  CV:\s*$', ln):
            in_cv = True
        elif re.match(r'^  [A-Z]', ln):
            in_cv = False
        if in_cv and re.match(r'^    M:\s*4', ln):
            ln = re.sub(r'M:\s*4', 'M: 2', ln)
        if in_cv and re.match(r'^\s*R:\s*\[', ln):
            vals = re.findall(r'-?\d+\.?\d*', ln.split('[')[1])
            ln = re.sub(r'\[.*\]', '[%s, %s]' % (vals[0], vals[1]), ln)
        out.append(ln)
    t = '\n'.join(out)
open(dst, 'w').write(t)
PY
  say "track $label (MCTRACK_GRID=$grid)"
  ( cd "$M" && MCTRACK_CFG="$cfg" MCTRACK_GRID="$grid" "$P" main.py --dataset nuscenes --process 10 ) \
      >> "$out/run.log" 2>&1 || { say "FAILED track $label"; exit 2; }
  # MCTrack writes results under a timestamped directory
  local found
  found=$(find "$out" -name 'results.json' -not -path "$out/results.json" | head -1)
  [ -n "$found" ] && cp "$found" "$out/results.json"
  [ -f "$out/results.json" ] || { say "FAILED: no results.json for $label"; exit 2; }
  say "track $label done"
}
run mct_detvel vx_detector 0 full
run mct_novelo  vx_none     0 pos
run mct_grid    vx_grid     1 pos
say "MCTRACK NUSC CHAIN COMPLETE"
