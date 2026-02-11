#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run the BM25 triples generator with canonical defaults.
# Run with: bash replicability/colbert/scripts/run_generate_triples_bm25.sh

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PY=python3

# Allow overriding via env vars if desired
INDEX=${TRECBM25_INDEX:-indices/trec-tot-2025-bm25}
QUERIES=${TOT_QUERIES:-data/trec-tot-2025/raw/queries/train-2025-queries.jsonl}
QRELS=${TOT_QRELS:-data/trec-tot-2025/raw/qrel/train-2025-qrel.txt}
OUT=${TOT_TRIPLES_OUT:-indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples}
TOPK=${TOT_TOPK:-100}
NEGS=${TOT_NEGS:-8}

LOGDIR=${TOT_LOGDIR:-replicability/colbert/logs}
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/generate_triples_$(date +%Y%m%d-%H%M%S).log"

echo "Generating triples (using train split by default)" | tee -a "$LOGFILE"
echo "Index: $INDEX" | tee -a "$LOGFILE"
echo "Output: $OUT" | tee -a "$LOGFILE"

# Call generator without CLI args so defaults inside the script (train split) are used.
"$PY" "$SCRIPT_DIR/generate_triples_bm25.py" 2>&1 | tee -a "$LOGFILE"

echo "Triples generation complete. (Defaults used: train queries/qrels -> $OUT)" | tee -a "$LOGFILE"
