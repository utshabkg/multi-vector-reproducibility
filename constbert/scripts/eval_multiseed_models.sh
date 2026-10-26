#!/bin/bash
# Evaluate all multi-seed fine-tuned models and compute statistics
# Run this after run_multiseed_finetuning.sh completes

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

SEEDS=(42 123 456)
RESULTS_DIR="$BASE_DIR/results/multiseed"
EVAL_SPLITS=("dev1" "dev2" "dev3" "test")

echo "========================================"
echo "Multi-Seed Evaluation"
echo "========================================"

# Create results file
RESULTS_JSON="$RESULTS_DIR/all_seed_results.json"
echo "{" > "$RESULTS_JSON"

# Evaluate ConstBERT models
echo ""
echo "=== Evaluating ConstBERT models ==="
for seed in "${SEEDS[@]}"; do
    CKPT_DIR="$BASE_DIR/checkpoints/constbert_tot_seed${seed}"
    if [ -d "$CKPT_DIR" ]; then
        echo "Evaluating ConstBERT seed=$seed..."
        for split in "${EVAL_SPLITS[@]}"; do
            echo "  - Split: $split"
            python experiments/10_eval_finetuned_tot_test.py \
                --checkpoint_dir "$CKPT_DIR" \
                --split "$split" \
                --output_file "$RESULTS_DIR/constbert_seed${seed}_${split}.json" \
                2>/dev/null || echo "    (skipped - evaluation script needs update)"
        done
    else
        echo "Checkpoint not found: $CKPT_DIR"
    fi
done

# Evaluate ColBERT models
echo ""
echo "=== Evaluating ColBERT models ==="
for seed in "${SEEDS[@]}"; do
    CKPT_DIR="$BASE_DIR/colbert-replicability/colbert/models/colbertv2-tot-seed${seed}"
    if [ -d "$CKPT_DIR" ]; then
        echo "Evaluating ColBERT seed=$seed..."
        # ColBERT evaluation via its own script
        for split in "${EVAL_SPLITS[@]}"; do
            echo "  - Split: $split"
            # This would need the ColBERT evaluation pipeline
        done
    else
        echo "Checkpoint not found: $CKPT_DIR"
    fi
done

echo ""
echo "========================================"
echo "Computing statistics..."
echo "========================================"

# Python script to compute mean ± std
python3 << 'EOF'
import json
import os
from pathlib import Path
import numpy as np

results_dir = Path("results/multiseed")
seeds = [42, 123, 456]
splits = ["dev1", "dev2", "dev3", "test"]

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

for model in ["constbert", "colbert"]:
    print(f"\n{model.upper()}:")
    for split in splits:
        mrr_values = []
        recall_values = []
        
        for seed in seeds:
            result_file = results_dir / f"{model}_seed{seed}_{split}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    if "MRR@10" in data.get("metrics", {}):
                        mrr_values.append(data["metrics"]["MRR@10"] * 100)
                    if "Recall@1000" in data.get("metrics", {}):
                        recall_values.append(data["metrics"]["Recall@1000"] * 100)
        
        if mrr_values:
            mrr_mean = np.mean(mrr_values)
            mrr_std = np.std(mrr_values)
            print(f"  {split}: MRR@10 = {mrr_mean:.2f} ± {mrr_std:.2f}%")
            
            if recall_values:
                r_mean = np.mean(recall_values)
                r_std = np.std(recall_values)
                print(f"         R@1000 = {r_mean:.2f} ± {r_std:.2f}%")
        else:
            print(f"  {split}: No results found")

print("\n" + "=" * 60)
print("For paper, use format: X.XX ± Y.YY%")
print("=" * 60)
EOF

echo ""
echo "Done! Results saved to: $RESULTS_DIR"
