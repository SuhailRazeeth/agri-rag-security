#!/usr/bin/env bash
# Split variant of replicate_until_done.sh: runs only the listed defense
# configurations for one model on one Groq key, writing to its own CSV, and
# retries every 30 minutes until each listed configuration has 100 rows.
# Usage:  bash tools/replicate_split.sh MODEL KEY_SLOT OUT_SUFFIX DEF1 [DEF2 ...]
#         KEY_SLOT "-" = GROQ_API_KEY (own key), "2" = GROQ_API_KEY_2 (co-author)
set -u
MODEL="$1"; SLOT="$2"; SUF="$3"; shift 3; DEFS="$*"
if [ "$SLOT" != "-" ]; then export GROQ_KEY_SLOT="$SLOT"; fi
TAG="${MODEL##*/}"
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
OUT="raw_defenses_expanded_${TAG}${SUF}.csv"
LOG="results/log_replicate_${TAG}${SUF}.txt"
NEED=$(( 100 * $# ))
while true; do
  echo "=== $(date) start [$DEFS]" >> "$LOG"
  "$PY" -m src.run_defenses --models "$MODEL" --sets indirect \
        --config-key defenses_revision --out "$OUT" --defenses $DEFS >> "$LOG" 2>&1
  N=$("$PY" -c "import pandas as pd,os;f='results/$OUT';d=pd.read_csv(f) if os.path.exists(f) else None;print(0 if d is None else int(d.defense.isin('$DEFS'.split()).sum()))")
  echo "=== $(date) rows=$N/$NEED" >> "$LOG"
  if [ "$N" -ge "$NEED" ]; then echo "DONE $MODEL [$DEFS] rows=$N"; break; fi
  sleep 1800
done
