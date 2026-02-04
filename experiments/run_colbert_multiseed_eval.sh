#!/bin/bash
# Run ColBERT multi-seed evaluation
# Estimated time: ~6-9 hours (2-3 hours per seed × 3 seeds)

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "ColBERT Multi-Seed Evaluation on ToT TEST"
echo "========================================"
echo "Estimated time: ~6-9 hours"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate colbertv2 2>/dev/null || conda activate constbert

# Fix CUDA architecture for RTX 6000 Ada (compute 8.9 → use 8.6 for CUDA 11.7)
export TORCH_CUDA_ARCH_LIST="8.6"

# Evaluate each seed
for seed in 42 123 456; do
    echo ""
    echo "[$(date)] Starting ColBERT seed=$seed..."
    python -u experiments/eval_multiseed_colbert.py --seed $seed
    echo "[$(date)] Completed ColBERT seed=$seed"
done

# Aggregate results
echo ""
echo "[$(date)] Aggregating results..."
python -u experiments/eval_multiseed_colbert.py --aggregate

# Also aggregate ConstBERT for comparison
python -u experiments/eval_multiseed_tot.py --aggregate

echo ""
echo "========================================"
echo "All ColBERT evaluations complete!"
echo "========================================"
