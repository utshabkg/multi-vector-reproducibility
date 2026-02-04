#!/bin/bash
# Run PLAID evaluation on all 5 BEIR datasets
# This demonstrates exact reproduction attempt using PLAID backend

set -e  # Exit on error

echo "================================================================================"
echo "BEIR PLAID Evaluation - Exact Reproduction Attempt"
echo "================================================================================"
echo ""
echo "This script evaluates all 13 BEIR datasets using PLAID indices to test whether"
echo "the architectural incompatibility discovered in RQ2 extends to out-of-domain tasks."
echo ""

# Configuration
SCRIPT="experiments/12-3_beir_plaid_eval.py"
INDEX_ROOT="/media/12TB/shared/datasets/indices/beir_plaid"
DATA_DIR="./data/beir"
OUTPUT_DIR="results"
LOG_DIR="logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/12-3.beir_plaid_eval.log"

echo "Configuration:"
echo "  - Script: $SCRIPT"
echo "  - Index root: $INDEX_ROOT"
echo "  - Data directory: $DATA_DIR"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Log file: $LOG_FILE"
echo ""

# Run evaluation for all datasets
echo "Starting evaluation..."
echo "================================================================================"

python -u "$SCRIPT" \
    --all \
    --index_root "$INDEX_ROOT" \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --k 1000 \
    --batch_size 128 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================================================"
echo "✅ BEIR PLAID Evaluation completed!"
echo "================================================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo "  - 12_beir_plaid_scifact.json"
echo "  - 12_beir_plaid_nfcorpus.json"
echo "  ...
echo "  ...
echo "  - 12_beir_plaid_fever.json"
echo "  - 12_beir_plaid_summary.json (aggregated results)"
echo ""
echo "Log saved to: $LOG_FILE"
echo ""
echo "Next step: Compare with FAISS results to demonstrate backend incompatibility"
echo ""
