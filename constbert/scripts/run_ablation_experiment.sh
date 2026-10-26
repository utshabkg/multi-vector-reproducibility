#!/bin/bash
#
# Ablation Fine-tuning Experiment Runner
# 
# This script runs the complete ablation experiment:
# 1. Generates combined triples (TRAIN+DEV1+DEV2)
# 2. Runs ConstBERT with 3 seeds (early stopping on DEV3)
# 3. Runs ColBERT with 3 seeds (early stopping on DEV3)  
# 4. Evaluates all models on TEST
# 5. Generates comparison table for paper
#
# Usage:
#   ./run_ablation_experiment.sh --test     # Quick test (5 min)
#   ./run_ablation_experiment.sh --full     # Full experiment (3-4 hours)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

# Suppress tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

echo "========================================"
echo "ABLATION FINE-TUNING EXPERIMENT"
echo "========================================"
echo "Date: $(date)"
echo "Base dir: $BASE_DIR"
echo ""

# Parse arguments
TEST_MODE=false
FULL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test|--full]"
            exit 1
            ;;
    esac
done

if [ "$TEST_MODE" = true ]; then
    echo "MODE: Test (quick pipeline verification)"
    echo ""
    
    # Test ConstBERT
    echo "[1/2] Testing ConstBERT pipeline..."
    echo "========================================"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate constbert
    python -u experiments/run_ablation_finetuning.py --test-mode
    
    # Test ColBERT
    echo ""
    echo "[2/2] Testing ColBERT pipeline..."
    echo "========================================"
    conda activate colbertv2
    python -u experiments/run_ablation_colbert.py --test-mode
    
    echo ""
    echo "========================================"
    echo "TEST COMPLETE!"
    echo "========================================"
    echo "Check output above for both ConstBERT and ColBERT results."
    
elif [ "$FULL_MODE" = true ]; then
    echo "MODE: Full multi-seed experiment"
    echo "Expected time: 4-6 hours (ConstBERT + ColBERT)"
    echo ""
    
    # Step 1: Generate triples (if not exists) - using constbert env
    echo "[Step 1/5] Checking/generating triples..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate constbert
    
    TRIPLES_FILE="data/tot/finetuning/combined_train_dev12.triples"
    if [ ! -f "$TRIPLES_FILE" ]; then
        echo "Generating combined triples..."
        python -u experiments/run_ablation_constbert.py --generate-triples-only
    else
        echo "Triples already exist: $TRIPLES_FILE"
    fi
    
    # Step 2: Run ConstBERT (3 seeds)
    echo ""
    echo "[Step 2/5] Running ConstBERT fine-tuning (3 seeds)..."
    echo "========================================"
    python -u experiments/run_ablation_constbert.py --full
    
    # Step 3: Run ColBERT (3 seeds) - switch to colbertv2 env
    echo ""
    echo "[Step 3/5] Running ColBERT fine-tuning (3 seeds)..."
    echo "========================================"
    conda activate colbertv2
    python -u experiments/run_ablation_colbert.py --full
    
    # Step 4: Generate comparison table
    echo ""
    echo "[Step 4/5] Generating comparison table..."
    conda activate constbert  # Back to constbert for pandas etc
    python -c "
import json
from pathlib import Path

# Load ConstBERT ablation results
constbert_file = Path('results/ablation_constbert_summary.json')
colbert_file = Path('results/colbert_ablation_summary.json')

print('='*70)
print('ABLATION RESULTS: TRAIN+DEV1+DEV2 (428q) with Early Stopping')
print('='*70)
print()

if constbert_file.exists():
    with open(constbert_file) as f:
        constbert = json.load(f)
    print('ConstBERT:')
    print(f'  Baseline MRR@10:    {constbert.get(\"baseline_mrr@10\", 0):.4f}')
    print(f'  Fine-tuned MRR@10:  {constbert[\"mean_mrr@10\"]:.4f} ± {constbert[\"std_mrr@10\"]:.4f}')
    print(f'  Change:             {constbert.get(\"delta_pct\", 0):+.1f}%')
else:
    print('ConstBERT results not found')

print()

if colbert_file.exists():
    with open(colbert_file) as f:
        colbert = json.load(f)
    print('ColBERT-v2:')
    print(f'  Baseline MRR@10:    {colbert.get(\"baseline_mrr@10\", 0):.4f}')
    print(f'  Fine-tuned MRR@10:  {colbert[\"mean_mrr@10\"]:.4f} ± {colbert[\"std_mrr@10\"]:.4f}')
    print(f'  Change:             {colbert.get(\"delta_pct\", 0):+.1f}%')
else:
    print('ColBERT results not found')

print()
print('='*70)
print('COMPARISON TABLE (for paper)')
print('='*70)
print()
print('| Model                    | Pretrained | 428q + Early Stop | Δ      |')
print('|--------------------------|------------|-------------------|--------|')

if constbert_file.exists():
    with open(constbert_file) as f:
        c = json.load(f)
    baseline = c.get('baseline_mrr@10', 0) * 100
    finetuned = c['mean_mrr@10'] * 100
    std = c['std_mrr@10'] * 100
    delta = c.get('delta_pct', 0)
    print(f'| ConstBERT              | {baseline:.2f}%      | {finetuned:.2f}% ± {std:.2f}%     | {delta:+.1f}%  |')

if colbert_file.exists():
    with open(colbert_file) as f:
        c = json.load(f)
    baseline = c.get('baseline_mrr@10', 0) * 100
    finetuned = c['mean_mrr@10'] * 100
    std = c['std_mrr@10'] * 100
    delta = c.get('delta_pct', 0)
    print(f'| ColBERT-v2             | {baseline:.2f}%      | {finetuned:.2f}% ± {std:.2f}%     | {delta:+.1f}%  |')

print()
"
    
    echo ""
    echo "========================================"
    echo "EXPERIMENT COMPLETE"
    echo "========================================"
    echo ""
    echo "Results saved to:"
    echo "  - results/ablation_constbert_summary.json (ConstBERT)"
    echo "  - results/colbert_ablation_summary.json (ColBERT)"
    echo "  - checkpoints/constbert_ablation_seed*/results.json"
    echo "  - checkpoints/colbert_ablation_seed*/results.json"
    echo ""
    
else
    echo "Usage: $0 [--test|--full]"
    echo ""
    echo "Options:"
    echo "  --test    Quick pipeline verification (5 min)"
    echo "  --full    Full multi-seed experiment (3-4 hours)"
    echo ""
fi
