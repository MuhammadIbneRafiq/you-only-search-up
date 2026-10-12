#!/usr/bin/env bash
# Runs the weekend suite notebooks one after another. Every notebook is resumable (results_suite/<run>/tasks.csv),
# so after a restart just run this script again: finished tasks are skipped.
cd "$(dirname "$0")/.."
CONDA="/c/Users/wifi stuff/anaconda3"
export PATH="$CONDA:$CONDA/Library/bin:$CONDA/Scripts:$PATH"
LOG=suite_runner/suite.log
for r in main seeds depth6 hier walmart favorita effect sensitivity beam100; do
  echo "=== $r start $(date '+%a %H:%M:%S')" >> "$LOG"
  "$CONDA/python.exe" suite_runner/run_inprocess.py "suite_$r.ipynb" "$LOG" && echo "=== $r finished $(date '+%a %H:%M:%S')" >> "$LOG" || echo "=== $r FAILED $(date '+%a %H:%M:%S')" >> "$LOG"
done
echo "SUITE_DONE $(date '+%a %H:%M:%S')" >> "$LOG"
