#!/usr/bin/env bash
set -euo pipefail

# Helper to build a ColBERT-v2 index for TREC ToT (trec-tot-2025)
# Usage:
# ./index_colbertv2_tot.sh /path/to/trec-tot-corpus.jsonl /path/to/checkpoint [index_name] [indices_root]

COLLECTION_PATH="${1:-data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl}"
CHECKPOINT="${2:-colbert-ir/colbertv2.0}"
INDEX_NAME="${3:-trec-tot-2025-colbertv2}"
INDICES_ROOT="${4:-indices}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY_CANDIDATE="$(conda run -n colbertv2 which python 2>/dev/null || echo python)"
if [ -x "$PY_CANDIDATE" ]; then
  PY="$PY_CANDIDATE"
else
  PY="$(command -v python3 || command -v python)"
fi

LOG_DIR="$REPO_ROOT/replicability/colbert/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%F_%H%M%S)
LOG="$LOG_DIR/index_${INDEX_NAME}_$TIMESTAMP.log"

echo "Collection: $COLLECTION_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Index name: $INDEX_NAME"
echo "Using Python: $PY"

NRANKS=$($PY - <<'PY'
try:
    import torch
    n = torch.cuda.device_count()
    print(n if n and n>0 else 1)
except Exception:
    print(1)
PY
)

echo "Detected nranks=$NRANKS"
echo "Logging to $LOG"

# If collection is JSONL, convert to TSV first (ColBERT-v2 expects TSV loader)
COLLECTION_ARG="$COLLECTION_PATH"
if [[ "$COLLECTION_PATH" == *.jsonl ]]; then
  TMP_DIR="$REPO_ROOT/replicability/colbert/tmp"
  mkdir -p "$TMP_DIR"
  CONVERTED="$TMP_DIR/${INDEX_NAME}_collection.tsv"
  MAP_OUT="$TMP_DIR/${INDEX_NAME}_idmap.tsv"
  echo "Converting JSONL collection $COLLECTION_PATH -> $CONVERTED (mapping -> $MAP_OUT)"
  "$PY" "$REPO_ROOT/replicability/colbert/scripts/convert_jsonl_to_tsv.py" "$COLLECTION_PATH" "$CONVERTED" "$MAP_OUT"
  COLLECTION_ARG="$CONVERTED"
fi

"$PY" "$REPO_ROOT/replicability/colbert/scripts/index_colbertv2.py" \
  --collection "$COLLECTION_ARG" \
  --index-name "$INDEX_NAME" \
  --checkpoint "$CHECKPOINT" \
  --output-root "$INDICES_ROOT" \
  --indices-root "$INDICES_ROOT" \
  --nranks "$NRANKS" \
  --overwrite true 2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]:-0}
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "Indexing finished successfully. Log: $LOG"
else
  echo "Indexing exited with code $EXIT_CODE. Check log: $LOG"
fi
