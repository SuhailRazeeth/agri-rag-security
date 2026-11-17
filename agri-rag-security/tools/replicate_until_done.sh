#!/usr/bin/env bash
# Revision replication of Study S3 (8 configurations x 100 indirect attacks) on one
# Groq model, on the free tier. Re-runs the resumable script every 30 minutes until
# all 800 rows exist, so the daily token cap only pauses the run.
# Usage:  bash tools/replicate_until_done.sh groq/openai/gpt-oss-120b [key_slot]
#         key_slot=2 uses GROQ_API_KEY_2 from .env (co-author's own Groq account)
set -u
MODEL="$1"
if [ -n "${2:-}" ]; then export GROQ_KEY_SLOT="$2"; fi
TAG="${MODEL##*/}"
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
OUT="raw_defenses_expanded_${TAG}.csv"
LOG="results/log_replicate_${TAG}.txt"
while true; do
  echo "=== $(date) start" >> "$LOG"
  "$PY" -m src.run_defenses --models "$MODEL" --sets indirect \
        --config-key defenses_revision --out "$OUT" >> "$LOG" 2>&1
  N=$("$PY" -c "import pandas as pd,os;f='results/$OUT';print(len(pd.read_csv(f)) if os.path.exists(f) else 0)")
  echo "=== $(date) rows=$N" >> "$LOG"
  if [ "$N" -ge 800 ]; then echo "DONE $MODEL rows=$N"; break; fi
  sleep 1800
done
