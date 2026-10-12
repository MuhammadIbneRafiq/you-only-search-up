#!/usr/bin/env bash
cd "$(dirname "$0")/.."
CONDA="/c/Users/wifi stuff/anaconda3"
export PATH="$CONDA:$CONDA/Library/bin:$CONDA/Scripts:$PATH"
LOG=suite_runner/suite.log
"$CONDA/python.exe" suite_runner/make_suite.py full original
echo "=== original start $(date '+%a %H:%M:%S')" >> "$LOG"
"$CONDA/python.exe" suite_runner/run_inprocess.py suite_original.ipynb "$LOG" && echo "=== original finished $(date '+%a %H:%M:%S')" >> "$LOG" || echo "=== original FAILED $(date '+%a %H:%M:%S')" >> "$LOG"
