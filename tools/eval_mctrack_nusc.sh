#!/usr/bin/env bash
# Official 40-threshold evaluation of the three MCTrack nuScenes runs, then the
# paired scene bootstrap of the pre-registered comparison (novelo vs grid).
# Run after run_mctrack_nusc.sh reports "MCTRACK NUSC CHAIN COMPLETE".
set -u
cd "$(dirname "$0")/.."
source env.sh
W=/media/mt-pc-0099/NVMe4TB/dgmtrack_work/mctrack_nusc
LOG=logs/mctrack_nusc_eval.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

for r in mct_detvel mct_novelo mct_grid; do
  [ -f "$W/runs/$r/results.json" ] || { say "MISSING $r/results.json"; exit 2; }
done

# about 10 GB each and this machine has ~22 GB free, so two at a time, not three
fail=0
run_eval() {
  say "eval $1"
  "$DGM_PYTHON" tools/evaluate_saved_official_fast.py \
      --submission "$W/runs/$1/results.json" --label "$1" --scene-counts \
      >> "logs/eval_$1.log" 2>&1
}
pids=()
for r in mct_novelo mct_grid mct_detvel; do          # the pre-registered pair first
  if [ -d "results/official40/$r" ]; then say "eval $r: exists"; continue; fi
  run_eval "$r" & pids+=($!)
  if [ "${#pids[@]}" -ge 2 ]; then
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    pids=()
  fi
done
for p in "${pids[@]:-}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { say "EVAL FAILED — see logs/eval_mct_*.log"; exit 3; }
say "all evaluations done"

for r in mct_detvel mct_novelo mct_grid; do
  "$DGM_PYTHON" - "$r" <<'PY' | tee -a "$LOG"
import json, sys
r = sys.argv[1]
m = json.load(open('results/official40/%s/metrics_summary.json' % r))
print('%-12s AMOTA %.4f  AMOTP %.4f  MOTA %.4f  IDS %d  RECALL %.4f'
      % (r, m['amota'], m['amotp'], m['mota'], m['ids'], m['recall']))
PY
done

say "paired bootstrap novelo vs grid"
"$DGM_PYTHON" tools/paired_bootstrap_official40.py \
    --base mct_novelo --compare mct_grid --resamples 4000 --seed 0 \
    >> "$LOG" 2>&1 || say "BOOTSTRAP FAILED"
say "MCTRACK NUSC EVAL COMPLETE"
