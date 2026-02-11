#!/bin/bash
# Fine-tune ColBERT-v2 on ToT dataset using generated triples
# Expected triples: data/processed/trec-tot-2025-triple-bm25/train.triples

# Parse command line arguments
TEST_MODE=false
if [[ "$1" == "--test" ]] || [[ "$1" == "--dry-run" ]]; then
    TEST_MODE=true
    echo "🧪 TEST MODE: Running with minimal parameters to validate setup"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="replicability/colbert/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
if $TEST_MODE; then
    LOG_FILE="$LOG_DIR/train_colbertv2_tot_test_${TIMESTAMP}.log"
else
    LOG_FILE="$LOG_DIR/train_colbertv2_tot_${TIMESTAMP}.log"
fi

echo "Starting ColBERT-v2 ToT fine-tuning..."
echo "Log file: $LOG_FILE"
echo "=========================================="

# Training configuration
CHECKPOINT_BASE="colbert-ir/colbertv2.0"
TRIPLES_PATH="indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples"
COLLECTION_PATH="data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl"
QUERIES_PATH="data/trec-tot-2025/raw/queries/train-2025-queries.jsonl"
OUTPUT_DIR="models/colbertv2-fine-tot"

if $TEST_MODE; then
    OUTPUT_DIR="models/colbertv2-fine-tot_test"
    # Create test triples subset (first 100 lines = ~12 queries)
    mkdir -p /tmp/colbert_test
    head -100 "$TRIPLES_PATH" > /tmp/colbert_test/test_triples.tsv
    TRIPLES_PATH="/tmp/colbert_test/test_triples.tsv"
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_BASE"
echo "  Triples: $TRIPLES_PATH"
echo "  Collection: $COLLECTION_PATH"
echo "  Queries: $QUERIES_PATH"
echo "  Output: $OUTPUT_DIR"
if $TEST_MODE; then
    echo "  Mode: TEST (50 steps, subset data)"
else
    echo "  Mode: FULL (2000 steps, full data)"
fi
echo "=========================================="

# Set training parameters based on mode
if $TEST_MODE; then
    MAXSTEPS=50
    SAVE_EVERY=25
    BATCH_SIZE=8
    WARMUP=10
else
    MAXSTEPS=2000
    SAVE_EVERY=500
    BATCH_SIZE=32
    WARMUP=1000
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_BASE"
echo "  Triples: $TRIPLES_PATH"
echo "  Collection: $COLLECTION_PATH"
echo "  Queries: $QUERIES_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Execute training with logging
if $TEST_MODE; then
    python -u replicability/colbert/scripts/train_colbert_tot_triples.py \
      --triples_file "$TRIPLES_PATH" \
      --corpus_file "$COLLECTION_PATH" \
      --queries_file "$QUERIES_PATH" \
      --checkpoint "$CHECKPOINT_BASE" \
      --checkpoint_dir "$OUTPUT_DIR" \
      --test_mode \
      --maxsteps 50 \
      --checkpoint_every 25 \
      --max_samples 100 \
      --val_samples 20 \
      --batch_size 8 \
      --warmup_steps 10 \
      2>&1 | tee "$LOG_FILE"
else
    python -u replicability/colbert/scripts/train_colbert_tot_triples.py \
      --triples_file "$TRIPLES_PATH" \
      --corpus_file "$COLLECTION_PATH" \
      --queries_file "$QUERIES_PATH" \
      --checkpoint "$CHECKPOINT_BASE" \
      --checkpoint_dir "$OUTPUT_DIR" \
      --maxsteps 2000 \
      --checkpoint_every 500 \
      --batch_size 32 \
      --warmup_steps 1000 \
      2>&1 | tee "$LOG_FILE"
fi

EXIT_CODE=$?
echo ""
echo "Training completed with exit code: $EXIT_CODE"
echo "Log saved to: $LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    if $TEST_MODE; then
        echo "✅ Test successful! Ready for full training."
        echo "Run: bash replicability/colbert/scripts/run_train_colbertv2_tot.sh"
    else
        echo "✅ Training successful!"
    fi
else
    echo "❌ Training failed. Check log for details."
fi
