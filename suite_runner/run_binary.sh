#!/usr/bin/env bash
# Binary study on all three datasets, one after the other; resumable (rerun to continue).
cd "$(dirname "$0")/.."
CONDA="/c/Users/wifi stuff/anaconda3"
export PATH="$CONDA:$CONDA/Library/bin:$CONDA/Scripts:$PATH"
LOG=suite_runner/binary.log
"$CONDA/python.exe" suite_runner/make_binary.py full
for ds in favorita walmart m5; do
  echo "=== $ds start $(date '+%a %H:%M:%S')" >> "$LOG"
  "$CONDA/python.exe" suite_runner/run_inprocess.py "binary_$ds.ipynb" "$LOG" && echo "=== $ds finished $(date '+%a %H:%M:%S')" >> "$LOG" || echo "=== $ds FAILED $(date '+%a %H:%M:%S')" >> "$LOG"
done
echo "BINARY_DONE $(date '+%a %H:%M:%S')" >> "$LOG"
