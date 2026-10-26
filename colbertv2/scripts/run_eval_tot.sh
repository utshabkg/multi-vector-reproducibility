#!/usr/bin/env bash
set -euo pipefail

# Wrapper: produce ranking with run_search.py and evaluate on TREC ToT dev set.
# Usage:
# ./run_eval_tot.sh [index_path] [queries_path] [qrels_path] [out_ranking] [k] [checkpoint] [collection]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PY_CANDIDATE="$(conda run -n colbertv2 which python 2>/dev/null || echo python)"
if [ -x "$PY_CANDIDATE" ]; then
  PY="$PY_CANDIDATE"
else
  PY="$(command -v python3 || command -v python)"
fi

INDEX_PATH="${1:-indices/trec-tot-2025-colbertv2}"
# Derive a short index name for tmp files and idmap (basename of index path)
INDEX_NAME="$(basename "$INDEX_PATH")"
QUERIES_PATH="${2:-$REPO_ROOT/data/trec-tot-2025/raw/queries/dev1-2025-queries.jsonl}"
QRELS_PATH="${3:-$REPO_ROOT/data/trec-tot-2025/raw/qrel/dev1-2025-qrel.txt}"
OUT_RANKING="${4:-$REPO_ROOT/replicability/colbert/results/tot_dev1.ranking.tsv}"
K=${5:-100}
CHECKPOINT="${6:-colbert-ir/colbertv2.0}"
COLLECTION="${7:-$REPO_ROOT/data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl}"

mkdir -p "$(dirname "$OUT_RANKING")"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/replicability/colbert/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%F_%H%M%S)
LOG="$LOG_DIR/run_eval_tot_$TIMESTAMP.log"

echo "Logging to: $LOG"

exec > >(tee -a "$LOG") 2>&1

echo "Using Python: $PY"
echo "Index: $INDEX_PATH"
echo "Queries: $QUERIES_PATH"
echo "Qrels: $QRELS_PATH"
echo "Output ranking: $OUT_RANKING"

echo "Running search..."

# If queries are JSONL, convert to TSV for the ColBERT `Queries` loader
QUERIES_ARG="$QUERIES_PATH"
if [[ "$QUERIES_PATH" == *.jsonl ]]; then
  TMP_DIR="$REPO_ROOT/replicability/colbert/tmp"
  mkdir -p "$TMP_DIR"
  CONVERTED_QUERIES="$TMP_DIR/${INDEX_NAME}_queries.tsv"
  echo "Converting JSONL queries $QUERIES_PATH -> $CONVERTED_QUERIES"
  "$PY" "$REPO_ROOT/replicability/colbert/scripts/convert_jsonl_to_tsv.py" "$QUERIES_PATH" "$CONVERTED_QUERIES"
  QUERIES_ARG="$CONVERTED_QUERIES"
fi

# Prefer converted collection if present (produced during indexing)
COLLECTION_ARG="$COLLECTION"
POSSIBLE_CONV_COLL="$REPO_ROOT/replicability/colbert/tmp/${INDEX_NAME}_collection.tsv"
if [ -f "$POSSIBLE_CONV_COLL" ]; then
  echo "Using converted collection $POSSIBLE_CONV_COLL"
  COLLECTION_ARG="$POSSIBLE_CONV_COLL"
fi

"$PY" "$REPO_ROOT/replicability/colbert/scripts/run_search.py" \
  --index "$INDEX_PATH" \
  --queries "$QUERIES_ARG" \
  --k $K \
  --output "$OUT_RANKING" \
  --checkpoint "$CHECKPOINT" \
  --collection "$COLLECTION_ARG" \
  --run_root "$REPO_ROOT" \
  --overwrite

# If an idmap exists from indexing, remap internal ids back to original pids
IDMAP="$REPO_ROOT/replicability/colbert/tmp/${INDEX_NAME}_idmap.tsv"
CONVERTED_RANKING="${OUT_RANKING%.tsv}.origpid.tsv"
if [ -f "$IDMAP" ]; then
  echo "Remapping internal pids in $OUT_RANKING -> $CONVERTED_RANKING using $IDMAP"
  "$PY" "$REPO_ROOT/replicability/colbert/scripts/convert_ranking_ids.py" "$OUT_RANKING" "$IDMAP" --out "$CONVERTED_RANKING"
else
  echo "No idmap found at $IDMAP; skipping remap"
  CONVERTED_RANKING="$OUT_RANKING"
fi

# Run TREC evaluation (inlined to keep single-file workflow).
OUT_PREFIX="${OUT_RANKING%.ranking.tsv}"
TREC_RUN="${OUT_PREFIX}.trec.run"
TREC_OUT="${OUT_PREFIX}_trec_eval.txt"
JSON_OUT="${OUT_PREFIX}_eval.json"
RUN_NAME="colbertv2"

echo "Converting TSV -> TREC run: $CONVERTED_RANKING -> $TREC_RUN"
awk -F"\t" '{printf("%s Q0 %s %s %s %s\n",$1,$2,$3,$4,ENVIRON["RUN_NAME"])}' RUN_NAME="$RUN_NAME" "$CONVERTED_RANKING" > "$TREC_RUN"

if command -v trec_eval >/dev/null 2>&1; then
  echo "Running trec_eval..."
  trec_eval -m all_trec "$QRELS_PATH" "$TREC_RUN" > "$TREC_OUT"
else
  echo "trec_eval not found. Attempting pytrec_eval fallback (requires python and pytrec_eval)."
  python3 - <<PY
import sys, json
try:
  import pytrec_eval
  print('pytrec_eval available — using it')
  from collections import defaultdict
  qrels = defaultdict(dict)
  with open('$QRELS_PATH') as f:
    for line in f:
      parts = line.strip().split()
      if not parts: continue
      qid,_,did,rel = parts
      qrels[qid][did] = int(rel)
  runs = defaultdict(dict)
  with open('$TREC_RUN') as f:
    for line in f:
      parts = line.strip().split()
      if not parts: continue
      qid = parts[0]; did = parts[2]; score = float(parts[4])
      runs[qid][did] = score
  measures = {'map','recip_rank','ndcg_cut_10','recall_1000'}
  evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
  res = evaluator.evaluate(runs)
  agg = {}
  for m in measures:
    vals = [r[m] for r in res.values()]
    agg[m] = sum(vals)/len(vals) if vals else 0.0
  with open('$TREC_OUT','w') as fo:
    for k,v in sorted(agg.items()):
      fo.write(f"{k} all {v}\n")
  print('WROTE', '$TREC_OUT')
except Exception as e:
  print('pytrec_eval not available or failed:', e, file=sys.stderr)
  print('Falling back to pure-Python evaluator')
  # call compute_trec_metrics.py
  import subprocess
  subprocess.check_call(['python3', '$REPO_ROOT/replicability/colbert/scripts/compute_trec_metrics.py', '$QRELS_PATH', '$TREC_RUN', '$JSON_OUT', '$QUERIES_PATH'])
  # write a simple text summary from the JSON
  with open('$JSON_OUT') as f:
    obj = json.load(f)
  with open('$TREC_OUT', 'w') as fo:
    for k,v in obj.items():
      fo.write(f"{k} all {v}\n")
  print('WROTE', '$TREC_OUT')
PY
fi

echo "Writing JSON metrics -> $JSON_OUT"
python3 - <<PY
import json
out = {}
with open('$TREC_OUT') as f:
    for line in f:
        parts = line.strip().split()
        if not parts: continue
        metric = parts[0]
        value = float(parts[-1])
        out[metric] = value
with open('$JSON_OUT','w') as fo:
    json.dump(out, fo, indent=2)
print('WROTE', '$JSON_OUT')
PY

echo "Done. Converted ranking: $CONVERTED_RANKING"

# Archive idmap into the index folder for reproducibility, if present.
if [ -n "${IDMAP-}" ] && [ -f "$IDMAP" ]; then
  ARCHIVE_IDMAP="$INDEX_PATH/${INDEX_NAME}_idmap.tsv"
  if [ ! -f "$ARCHIVE_IDMAP" ]; then
    echo "Archiving idmap to $ARCHIVE_IDMAP"
    cp "$IDMAP" "$ARCHIVE_IDMAP" || echo "Warning: failed to copy idmap to $ARCHIVE_IDMAP"
  else
    echo "Idmap already archived at $ARCHIVE_IDMAP"
  fi
fi

# By default remove large temporary converted files to save space.
# Set KEEP_TMP=1 in the environment to retain them for debugging.
if [ "${KEEP_TMP:-0}" -eq 0 ]; then
  echo "Removing temporary converted files (set KEEP_TMP=1 to keep them)"
  if [ -n "${CONVERTED_QUERIES-}" ] && [ -f "$CONVERTED_QUERIES" ]; then
    rm -v "$CONVERTED_QUERIES" || true
  fi
  if [ -n "${POSSIBLE_CONV_COLL-}" ] && [ -f "$POSSIBLE_CONV_COLL" ]; then
    rm -v "$POSSIBLE_CONV_COLL" || true
  fi
  if [ -n "${IDMAP-}" ] && [ -f "$IDMAP" ]; then
    rm -v "$IDMAP" || true
  fi
else
  echo "KEEP_TMP=1: leaving temporary files in place for inspection"
fi
