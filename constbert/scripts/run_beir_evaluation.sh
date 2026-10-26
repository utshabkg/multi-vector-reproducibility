#!/bin/bash
#
# OPTIMIZED BEIR Evaluation Script
# Configured for: 120 CPU cores, 700GB RAM, 2x RTX 6000 Ada GPUs
#
# Usage: bash run_beir_evaluation_optimized.sh
#

set -e  # Exit on error

echo "=============================================="
echo "OPTIMIZED BEIR Evaluation"
echo "=============================================="
echo "Hardware: 120 cores, 700GB RAM, 2x RTX 6000"
echo "=============================================="
echo ""

# OPTIMIZED Configuration
MODEL="constbert"
DATA_DIR="./data/beir"
OUTPUT_DIR="./results"

# OPTIMIZED PARAMETERS (tuned for 48GB GPU)
BATCH_SIZE=1024          # Increased for 48GB VRAM
QUERY_BATCH_SIZE=512     # Process 512 queries at once
NUM_WORKERS=40           # Use ~1/3 of CPU cores for data loading
NLIST=1024               # Increased from 256 for better partitioning
NPROBE=128               # Increased from 64 for better recall
NGPUS=1                  # Single GPU (multi-GPU can hang with FAISS)

# Create log directory
mkdir -p logs

# All 13 BEIR datasets (ordered by size for efficient testing)
DATASETS=(
    "scifact"           # 5K docs - FAST TEST
    "arguana"           # 8.7K docs
    "nfcorpus"          # 3.6K docs  
    "scidocs"           # 25K docs
    "fiqa"              # 57K docs
    "trec-covid"        # 171K docs
    "webis-touche2020"  # 382K docs
    "quora"             # 523K docs
    "nq"                # 2.7M docs
    "dbpedia-entity"    # 4.6M docs
    "hotpotqa"          # 5.2M docs
    "climate-fever"     # 5.4M docs
    "fever"             # 5.4M docs
)

echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE (encoding)"
echo "Query batch size: $QUERY_BATCH_SIZE (search)"
echo "Workers: $NUM_WORKERS"
echo "GPUs: $NGPUS"
echo "FAISS: nlist=$NLIST, nprobe=$NPROBE"
echo ""
echo "Datasets (${#DATASETS[@]}): ${DATASETS[@]}"
echo ""

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Track timing
TOTAL_START=$(date +%s)

# Evaluate each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=============================================="
    echo "EVALUATING: $DATASET"
    echo "=============================================="
    
    START=$(date +%s)
    
    # Run evaluation
    python -u experiments/11_beir_evaluation.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --query-batch-size "$QUERY_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --nlist "$NLIST" \
        --nprobe "$NPROBE" \
        --ngpus "$NGPUS" \
        --gpu 2>&1 | tee "logs/11_beir_${DATASET}_optimized.log"
    
    END=$(date +%s)
    ELAPSED=$((END - START))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))
    
    echo ""
    echo "✓ Completed $DATASET in ${MINS}m ${SECS}s"
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=============================================="
echo "BEIR EVALUATION COMPLETE!"
echo "=============================================="
echo "Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "Results saved to: $OUTPUT_DIR/11_beir_*.json"
echo ""

# Optional: Run summary analysis
if command -v python &> /dev/null; then
    echo "Generating summary..."
    python << 'PYEOF'
import json
import glob

results_files = glob.glob('./results/11_beir_constbert_*.json')
results_files = [f for f in results_files if 'summary' not in f]

print('\n' + '='*100)
print('FINAL SUMMARY')
print('='*100)
print('{:<20} {:>10} {:>12} {:>10} {:>10} {:>12}'.format('Dataset', 'NDCG@10', 'Recall@100', 'MRR@10', 'Time(s)', 'Docs/sec'))
print('-'*100)

total_time = 0
for filepath in sorted(results_files):
    with open(filepath) as f:
        result = json.load(f)
    
    dataset = result['dataset']
    metrics = result['metrics']
    timing = result['timing']['total_time_s']
    num_docs = result['num_docs']
    docs_per_sec = num_docs / result['timing']['encoding_time_s'] if result['timing']['encoding_time_s'] > 0 else 0
    
    total_time += timing
    
    print('{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>10.1f} {:>12.0f}'.format(
        dataset,
        metrics.get('ndcg@10', 0),
        metrics.get('recall@100', 0),
        metrics.get('mrr@10', 0),
        timing,
        docs_per_sec
    ))

print('='*100)
print(f'Total evaluation time: {total_time:.1f}s ({total_time/60:.1f}m)')
print('='*100)
PYEOF
fi

# map@1000................................ 0.0971
# 2026-01-27 23:56:52,947 - __main__ - INFO -   mrr@10.................................. 0.1475
# 2026-01-27 23:56:52,947 - __main__ - INFO -   ndcg@10................................. 0.0785
# 2026-01-27 23:56:52,947 - __main__ - INFO -   ndcg@100................................ 0.1066
# 2026-01-27 23:56:52,947 - __main__ - INFO -   p@10.................................... 0.0383
# 2026-01-27 23:56:52,947 - __main__ - INFO -   recall@1000............................. 0.2679
# 2026-01-27 23:56:52,947 - __main__ - INFO -   recall@200.............................. 0.1885
# 2026-01-27 23:56:52,947 - __main__ - INFO -   recall@50............................... 0.1326