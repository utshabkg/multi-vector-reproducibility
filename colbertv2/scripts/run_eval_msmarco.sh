#!/usr/bin/env bash
set -euo pipefail

# Wrapper: produce ranking with run_search.py and evaluate with ColBERT utility evaluator.
# Run without args to use defaults (shared paths), or provide 4 positional args:
# ./run_eval_msmarco.sh [index_path] [queries_path] [qrels_path] [out_ranking] [k] [checkpoint] [collection]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Prefer the conda env python if present
PY_CANDIDATE="$(conda run -n colbertv2 which python 2>/dev/null || echo python)"
if [ -x "$PY_CANDIDATE" ]; then
  PY="$PY_CANDIDATE"
else
  PY="$(command -v python3 || command -v python)"
fi

INDEX_PATH="${1:-indices/msmarco-passage-colbertv2}"
QUERIES_PATH="${2:-data/msmarco-passage/queries.dev.small.tsv}"
QRELS_PATH="${3:-data/msmarco-passage/qrels.dev.small.tsv}"
OUT_RANKING="${4:-$REPO_ROOT/replicability/colbert/results/msmarco_full.ranking.tsv}"
K=${5:-100}
CHECKPOINT="${6:-colbert-ir/colbertv2.0}"
COLLECTION="${7:-data/msmarco-passage/collection.tsv}"

mkdir -p "$(dirname "$OUT_RANKING")"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/replicability/colbert/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%F_%H%M%S)
LOG="$LOG_DIR/run_eval_msmarco_$TIMESTAMP.log"

echo "Logging to: $LOG"

# Route all stdout/stderr through tee to the log file for full provenance
exec > >(tee -a "$LOG") 2>&1

echo "Using Python: $PY"
echo "Index: $INDEX_PATH"
echo "Queries: $QUERIES_PATH"
echo "Qrels: $QRELS_PATH"
echo "Output ranking: $OUT_RANKING"

echo "Running search..."
"$PY" "$REPO_ROOT/replicability/colbert/scripts/run_search.py" \
  --index "$INDEX_PATH" \
  --queries "$QUERIES_PATH" \
  --k $K \
  --output "$OUT_RANKING" \
  --checkpoint "$CHECKPOINT" \
  --collection "$COLLECTION" \
  --run_root "$REPO_ROOT"

echo "Running MS MARCO evaluator..."
"$PY" -m utility.evaluate.msmarco_passages --ranking "$OUT_RANKING" --qrels "$QRELS_PATH"

echo "Done. Ranking: $OUT_RANKING"
