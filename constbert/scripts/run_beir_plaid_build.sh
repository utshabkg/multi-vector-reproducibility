#!/bin/bash
# Build PLAID indices for all 13 BEIR datasets

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Create log directory if it doesn't exist
mkdir -p logs

echo "================================================================================"
echo "Building PLAID Indices for All 13 BEIR Datasets"
echo "================================================================================"

python -u experiments/rq3/12-1_build_beir_plaid_indices.py constbert --all --batch-size 512 --num-workers 20 2>&1 | tee logs/12-1.beir_plaid_build.log

echo ""
echo "================================================================================"
echo "Building IVF Structures"
echo "================================================================================"

python -u experiments/rq3/12-2_build_ivf_for_beir.py --all 2>&1 | tee logs/12-2.beir_ivf_build.log

echo ""
echo "================================================================================"
echo "✅ All PLAID indices and IVF structures built!"
echo "================================================================================"
echo "Ready to run: bash scripts/run_beir_plaid_evaluation.sh"
