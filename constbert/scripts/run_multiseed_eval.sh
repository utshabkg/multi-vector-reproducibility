#!/bin/bash
# Run evaluation for all 6 fine-tuned models (3 seeds × 2 models)
# Estimated time: ~15 hours total

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "Multi-Seed Evaluation on ToT TEST"
echo "========================================"
echo "Estimated time: ~15 hours (2.5h per model × 6 models)"
echo ""

# Evaluate ConstBERT seeds (can run on GPU 0)
for seed in 42 123 456; do
    echo "[$(date)] Evaluating ConstBERT seed=$seed..."
    CUDA_VISIBLE_DEVICES=0 python experiments/eval_multiseed_tot.py --model constbert --seed $seed
done

# Aggregate results
echo ""
echo "[$(date)] Aggregating results..."
python experiments/eval_multiseed_tot.py --aggregate

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results in: results/multiseed/"
echo "========================================"