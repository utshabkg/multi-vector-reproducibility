#!/bin/bash
# Multi-seed fine-tuning runner
# Runs 3 seeds for both ConstBERT and ColBERT using both GPUs in parallel
# Maximizes speed by running 2 jobs concurrently

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

SEEDS=(42 123 456)
RESULTS_DIR="$BASE_DIR/results/multiseed"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Multi-Seed Fine-tuning for Statistical Significance"
echo "========================================"
echo "Seeds: ${SEEDS[*]}"
echo "Base dir: $BASE_DIR"
echo "Results: $RESULTS_DIR"
echo ""

# Function to run ConstBERT with a specific seed
run_constbert() {
    local SEED=$1
    local GPU=$2
    local OUTPUT_DIR="$BASE_DIR/checkpoints/constbert_tot_seed${SEED}"
    local LOG_FILE="$RESULTS_DIR/constbert_seed${SEED}.log"
    
    echo "[$(date '+%H:%M:%S')] ConstBERT seed=$SEED starting on GPU $GPU..."
    
    CUDA_VISIBLE_DEVICES=$GPU python experiments/10_finetune_constbert_tot.py \
        --triples_file indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples \
        --corpus_file data/trec-tot-2025/trec-tot-2025-corpus.jsonl \
        --queries_file data/trec-tot-2025/queries/train-2025-queries.jsonl \
        --output_dir "$OUTPUT_DIR" \
        --max_steps 5000 \
        --batch_size 8 \
        --grad_accum_steps 4 \
        --learning_rate 5e-6 \
        --warmup_steps 500 \
        --seed $SEED \
        2>&1 | tee "$LOG_FILE"
    
    echo "[$(date '+%H:%M:%S')] ConstBERT seed=$SEED completed"
}

# Function to run ColBERT with a specific seed  
run_colbert() {
    local SEED=$1
    local GPU=$2
    local COLBERT_DIR="$BASE_DIR/colbert-reproduce-copy/colbert-replicability/colbert"
    local OUTPUT_DIR="$COLBERT_DIR/models/colbertv2-tot-seed${SEED}"
    local LOG_FILE="$RESULTS_DIR/colbert_seed${SEED}.log"
    
    echo "[$(date '+%H:%M:%S')] ColBERT seed=$SEED starting on GPU $GPU..."
    
    cd "$COLBERT_DIR"
    PYTHONPATH="$COLBERT_DIR" CUDA_VISIBLE_DEVICES=$GPU RANDOM_SEED=$SEED python scripts/train_colbert_tot_triples.py \
        --triples_file indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples \
        --corpus_file data/trec-tot-2025/trec-tot-2025-corpus.jsonl \
        --queries_file data/trec-tot-2025/queries/train-2025-queries.jsonl \
        --checkpoint colbert-ir/colbertv2.0 \
        --checkpoint_dir "$OUTPUT_DIR" \
        --maxsteps 2000 \
        --batch_size 32 \
        --learning_rate 3e-6 \
        --warmup_steps 1000 \
        --use_amp \
        2>&1 | tee "$LOG_FILE"
    cd "$BASE_DIR"
    
    echo "[$(date '+%H:%M:%S')] ColBERT seed=$SEED completed"
}

export -f run_constbert run_colbert
export BASE_DIR RESULTS_DIR

# Run jobs in parallel batches of 2 (one per GPU)
echo ""
echo "Starting fine-tuning jobs..."
echo "Strategy: 2 jobs in parallel (1 per GPU)"
echo ""

START_TIME=$(date +%s)

# Batch 1: seed 42 (ConstBERT GPU0, ColBERT GPU1)
echo "=== Batch 1: seed 42 ==="
run_constbert 42 0 &
PID1=$!
run_colbert 42 1 &
PID2=$!
wait $PID1 $PID2

# Batch 2: seed 123 (ConstBERT GPU0, ColBERT GPU1)
echo "=== Batch 2: seed 123 ==="
run_constbert 123 0 &
PID1=$!
run_colbert 123 1 &
PID2=$!
wait $PID1 $PID2

# Batch 3: seed 456 (ConstBERT GPU0, ColBERT GPU1)
echo "=== Batch 3: seed 456 ==="
run_constbert 456 0 &
PID1=$!
run_colbert 456 1 &
PID2=$!
wait $PID1 $PID2

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "All jobs completed!"
echo "Total time: $((ELAPSED / 60)) minutes"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run evaluation on all checkpoints"
echo "2. Compute mean ± std for paper"
echo ""
echo "Checkpoints saved to:"
for seed in "${SEEDS[@]}"; do
    echo "  - checkpoints/constbert_tot_seed${seed}/"
    echo "  - colbert-replicability/colbert/models/colbertv2-tot-seed${seed}/"
done
