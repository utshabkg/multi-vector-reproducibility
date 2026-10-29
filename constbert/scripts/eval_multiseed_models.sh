#!/bin/bash
# Evaluate all multi-seed fine-tuned models and compute statistics
# Run this after run_multiseed_finetuning.sh completes

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# ConstBERT seeds 42/123/456 on ToT TEST, then mean ± std
bash "$BASE_DIR/scripts/run_multiseed_eval.sh"

# ColBERT-v2 seeds 42/123/456 on ToT TEST, then mean ± std for both models
bash "$BASE_DIR/scripts/run_colbert_multiseed_eval.sh"

echo ""
echo "Done! Results saved to: $BASE_DIR/results/multiseed"
