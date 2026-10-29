#!/bin/bash
# Master script to fine-tune ConstBERT on TREC ToT 2025
# This runs the complete pipeline: triple generation → fine-tuning → evaluation

set -e  # Exit on error

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ConstBERT ToT Fine-tuning Pipeline                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
TEST_MODE=false
if [[ "$1" == "--test" ]] || [[ "$1" == "--dry-run" ]]; then
    TEST_MODE=true
    echo "🧪 TEST MODE: Running with minimal parameters"
fi

# Paths - using standard ML splits
CORPUS_FILE="data/trec-tot-2025/trec-tot-2025-corpus.jsonl"

# Training data
TRAIN_QUERIES="data/trec-tot-2025/queries/train-2025-queries.jsonl"
TRAIN_QRELS="data/trec-tot-2025/qrel/train-2025-qrel.txt"
TRIPLES_FILE="indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples"

# Validation data (monitor during training)
DEV1_QUERIES="data/trec-tot-2025/queries/dev1-2025-queries.jsonl"
DEV1_QRELS="data/trec-tot-2025/qrel/dev1-2025-qrel.txt"

# Test data (single-run evaluation)
TEST_QUERIES="data/trec-tot-2025/queries/test-2025-queries.jsonl"
TEST_QRELS="data/trec-tot-2025/qrel/test-2025-qrel.txt"

OUTPUT_DIR="checkpoints/constbert_tot_finetuned"

if $TEST_MODE; then
    OUTPUT_DIR="checkpoints/constbert_tot_finetuned_test"
fi

# Create directories
mkdir -p data/tot/finetuning
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="logs/constbert_finetuning_${TIMESTAMP}.log"

echo "📝 Configuration:"
echo "   Corpus: $CORPUS_FILE"
echo ""
echo "   Training:"
echo "     Queries: $TRAIN_QUERIES (143 queries)"
echo "     Qrels: $TRAIN_QRELS"
echo "     Triples: $TRIPLES_FILE"
echo ""
echo "   Validation (monitoring):"
echo "     DEV1 Queries: $DEV1_QUERIES (142 queries)"
echo "     DEV1 Qrels: $DEV1_QRELS"
echo ""
echo "   Test (final evaluation):"
echo "     TEST Queries: $TEST_QUERIES (622 queries)"
echo "     TEST Qrels: $TEST_QRELS"
echo ""
echo "   Model output: $OUTPUT_DIR"
echo "   Log file: $LOG_FILE"
echo ""

# Function to log and execute
run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ STEP: $step_name"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    "$@" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ ERROR: Step '$step_name' failed!"
        exit 1
    fi
    
    echo "✅ Step '$step_name' completed successfully"
}

# Step 1: Verify training triples exist
if [ ! -f "$TRIPLES_FILE" ]; then
    echo "❌ ERROR: Triples file not found: $TRIPLES_FILE"
    echo "   Please generate triples first."
    exit 1
else
    TRIPLE_COUNT=$(wc -l < "$TRIPLES_FILE")
    echo "✅ Using existing triples file: $TRIPLES_FILE ($TRIPLE_COUNT triples)"
fi

# Step 2: Fine-tune ConstBERT on TRAIN data
FINETUNE_ARGS=(
    --triples_file "$TRIPLES_FILE"
    --corpus_file "$CORPUS_FILE"
    --queries_file "$TRAIN_QUERIES"
    --output_dir "$OUTPUT_DIR"
    --batch_size 8
    --grad_accum_steps 4
    --learning_rate 5e-6
    --warmup_steps 500
    --max_steps 5000
)

if $TEST_MODE; then
    FINETUNE_ARGS+=(--test_mode --max_steps 50)
fi

run_step "Fine-tune ConstBERT on TRAIN split" python -u experiments/rq5/10_finetune_constbert_tot.py "${FINETUNE_ARGS[@]}"

# Test mode is a training smoke test only. The evaluator always loads
# checkpoints/constbert_tot_finetuned (not $OUTPUT_DIR) and writes the same result files as a full run.
if $TEST_MODE; then
    echo ""
    echo "🧪 TEST MODE: fine-tuning finished, model saved to $OUTPUT_DIR"
    echo "   Evaluation skipped in test mode. Full log: $LOG_FILE"
    exit 0
fi

# Step 3: Evaluate on DEV1/DEV2/DEV3 and TEST splits
# 10_eval_finetuned_tot_test.py reads the checkpoint from checkpoints/constbert_tot_finetuned
# and writes results/finetune-results_tot_<split>.json
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP: Evaluate Fine-tuned Model on DEV1/DEV2/DEV3 and TEST Splits"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

run_step "Evaluate on DEV1/DEV2/DEV3 and TEST splits" python -u experiments/rq5/10_eval_finetuned_tot_test.py \
    --splits dev1 dev2 dev3 test

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           PIPELINE COMPLETED                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results:"
echo "   Fine-tuned model: $OUTPUT_DIR"
echo "   Evaluation: results/finetune-results_tot_{dev1,dev2,dev3,test}.json"
echo "   Training log: ${OUTPUT_DIR}/training_log.json"
echo "   Full log: $LOG_FILE"
echo ""
echo "Data splits used:"
echo "   ✅ TRAIN (143 queries) → Fine-tuning"
echo "   ✅ DEV1/DEV2/DEV3 (142/143/536 queries) → Evaluation"
echo "   ✅ TEST (622 queries) → Evaluation (single run)"
echo ""
echo "✅ All steps completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review results in results/finetune-results_tot_{dev1,dev2,dev3,test}.json"
echo "  2. Run same pipeline for ColBERT with identical splits"
echo "  3. For the 3-seed TEST row of Table 9, run scripts/run_multiseed_finetuning.sh, then scripts/eval_multiseed_models.sh"
