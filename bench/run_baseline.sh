#!/usr/bin/env bash
# Baseline sculpt-quality benchmark — run once per code change, compare against bench/baseline.
# Settings per roadmap: 25 steps, cfg 3.0, seed 7, best-of-3 seed candidates.
# Uses the frozen binary in bench/bin so concurrent rebuilds can't change results mid-run.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=bench/bin/LiToSmoke
OUT="${1:-bench/baseline}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"
for img in testset/*.jpg; do
  name="$(basename "$img" .jpg)"
  log="$OUT/$name.log"
  echo "=== $name ($(date +%H:%M:%S)) ===" | tee -a "$SUMMARY"
  t0=$SECONDS
  "$BIN" engine weights "$img" "$OUT/$name.ply" 25 3.0 7 3 > "$log" 2>&1
  rc=$?
  t1=$SECONDS
  if [[ $rc -ne 0 ]]; then
    echo "  ENGINE FAILED (rc=$rc) after $((t1-t0))s — see $log" | tee -a "$SUMMARY"
    continue
  fi
  grep -E "ENGINE DONE|IoU" "$log" | tail -4 >> "$SUMMARY"
  "$BIN" mesh "$OUT/$name.gs.ply" "$OUT/${name}_mesh" 256 0.5 >> "$log" 2>&1 \
    && grep "MESH:" "$log" | tail -1 >> "$SUMMARY"
  "$BIN" render "$OUT/$name.ply" "$OUT/${name}_render.png" >> "$log" 2>&1
  echo "  engine ${t0:+$((t1-t0))}s total $((SECONDS-t0))s" >> "$SUMMARY"
done
echo "BASELINE COMPLETE $(date)" | tee -a "$SUMMARY"
