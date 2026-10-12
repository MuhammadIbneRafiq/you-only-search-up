#!/usr/bin/env bash
cd "$(dirname "$0")/.."
CONDA="/c/Users/wifi stuff/anaconda3"
export PATH="$CONDA:$CONDA/Library/bin:$CONDA/Scripts:$PATH"
LOG=suite_runner/original_exact.log
"$CONDA/python.exe" suite_runner/make_suite.py full original_exact
echo "=== original_exact start $(date '+%a %H:%M:%S')" >> "$LOG"
"$CONDA/python.exe" suite_runner/run_inprocess.py suite_original_exact.ipynb "$LOG" && echo "=== original_exact finished $(date '+%a %H:%M:%S')" >> "$LOG" || echo "=== original_exact FAILED $(date '+%a %H:%M:%S')" >> "$LOG"
