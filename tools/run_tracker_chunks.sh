#!/usr/bin/env bash
# Run the restored tracker with the same 10-scene chunking the saved runs used,
# so that particle random streams line up and a saved run can be reproduced
# exactly. usage: tools/run_tracker_chunks.sh <tag> <detection cache name> <args...>
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
TAG="$1"; DET="$2"; shift 2
ORIG="$(python3 -c "import json;print(json.load(open('configs/paths.json'))['original'])")"
NUSC="$(python3 -c "import json;print(json.load(open('configs/paths.json'))['nuscenes'])")"
OUT="results/tracker_runs/$TAG"
mkdir -p "$OUT/scenes"
echo "$(date '+%F %T') [$TAG] chunked; args: $*" >> "$OUT/run.log"
i=0
while read -r chunk; do
  i=$((i+1)); dir="$OUT/scenes/c$(printf '%03d' $((i-1)))"
  [ -f "$dir/tracking_results.json" ] && continue
  # DET is a cache name under the original cache tree, or an absolute directory.
  case "$DET" in /*) DET_DIR="$DET";; *) DET_DIR="$ORIG/cache/$DET";; esac
  ./build/dogm_track --pose_cache "$ORIG/cache/poses.json" --det_dir "$DET_DIR" \
      --dataroot "$NUSC" --scenes "$chunk" --out_dir "$dir" "$@" >> "$OUT/run.log" 2>&1 \
      || { echo "[$TAG] chunk $i FAILED" | tee -a "$OUT/run.log"; exit 2; }
done < configs/val150_chunks.txt
"$DGM_PYTHON" tools/merge_tracking.py --in-dir "$OUT/scenes" --out "$OUT/tracking_results.json" >> "$OUT/run.log" 2>&1
cp "$OUT/scenes/c000/run_meta.json" "$OUT/run_meta.json"
sha256sum "$OUT/tracking_results.json" | tee -a "$OUT/run.log"
echo "$(date '+%F %T') [$TAG] DONE" | tee -a "$OUT/run.log"
