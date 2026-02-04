#!/bin/bash
# ColBERT-v2 BEIR Evaluation Script
# Evaluates ColBERT-v2 on all 13 BEIR datasets for paper comparison

set -e  # Exit on error

source ~/anaconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use constbert conda environment
CONDA_ENV="constbert"

echo "========================================"
echo "ColBERT-v2 BEIR Evaluation"
echo "========================================"
echo "Start time: $(date)"
echo "Using conda environment: $CONDA_ENV"
echo ""

# Configuration
DATA_DIR="./data/beir"
INDEX_DIR="/media/12TB/shared/datasets/indices/colbertv2-beir"
OUTPUT_DIR="./results"
CHECKPOINT="colbert-ir/colbertv2.0"
NBITS=2  # ColBERT-v2 paper uses nbits=2 (ConstBERT PLAID uses nbits=1)
DOC_MAXLEN=180

# Create directories
mkdir -p "$INDEX_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Run evaluation on all datasets
# Order: small datasets first, then medium, then large
# This allows early results while large datasets process

# Small datasets (< 100K docs) - faster [COMPLETED - COMMENTED OUT]
# SMALL_DATASETS="nfcorpus scifact arguana scidocs fiqa"

# Medium datasets (100K - 1M docs) [COMPLETED - COMMENTED OUT]
# MEDIUM_DATASETS="trec-covid quora webis-touche2020"

# Large datasets (> 1M docs) - slower [IN PROGRESS]
LARGE_DATASETS="nq hotpotqa dbpedia-entity fever climate-fever"

# echo "Evaluating small datasets first..."
# for dataset in $SMALL_DATASETS; do
#     echo ""
#     echo "========================================"
#     echo "Dataset: $dataset"
#     echo "========================================"
#     conda run -n $CONDA_ENV python -u experiments/13_beir_colbertv2_evaluation.py \
#         --dataset "$dataset" \
#         --data-dir "$DATA_DIR" \
#         --index-dir "$INDEX_DIR" \
#         --output-dir "$OUTPUT_DIR" \
#         --checkpoint "$CHECKPOINT" \
#         --nbits $NBITS \
#         --doc-maxlen $DOC_MAXLEN
# done

# echo ""
# echo "Evaluating medium datasets..."
# for dataset in $MEDIUM_DATASETS; do
#     echo ""
#     echo "========================================"
#     echo "Dataset: $dataset"
#     echo "========================================"
#     conda run -n $CONDA_ENV python -u experiments/13_beir_colbertv2_evaluation.py \
#         --dataset "$dataset" \
#         --data-dir "$DATA_DIR" \
#         --index-dir "$INDEX_DIR" \
#         --output-dir "$OUTPUT_DIR" \
#         --checkpoint "$CHECKPOINT" \
#         --nbits $NBITS \
#         --doc-maxlen $DOC_MAXLEN
# done

echo ""
echo "Evaluating large datasets..."
conda activate $CONDA_ENV 
for dataset in $LARGE_DATASETS; do
    echo ""
    echo "========================================"
    echo "Dataset: $dataset"
    echo "========================================"
    python -u experiments/13_beir_colbertv2_evaluation.py \
        --dataset "$dataset" \
        --data-dir "$DATA_DIR" \
        --index-dir "$INDEX_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint "$CHECKPOINT" \
        --nbits $NBITS \
        --doc-maxlen $DOC_MAXLEN
done

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR/13_beir_colbertv2_*.json"
echo "========================================"

# Generate summary
echo ""
echo "Generating summary..."
python -c "
import json
from pathlib import Path

results_dir = Path('$OUTPUT_DIR')
all_results = {}

for f in results_dir.glob('13_beir_colbertv2_*.json'):
    if 'summary' not in f.name:
        with open(f) as fp:
            data = json.load(fp)
            dataset = data['dataset']
            all_results[dataset] = data

# Save summary
with open(results_dir / '13_beir_colbertv2_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Print table
print()
print('='*80)
print('COLBERT-V2 BEIR RESULTS SUMMARY')
print('='*80)
print(f'{\"Dataset\":<20} {\"Our NDCG@10\":>12} {\"Paper NDCG@10\":>14} {\"Delta\":>10}')
print('-'*80)

total_our = 0
total_paper = 0
count = 0

for name in sorted(all_results.keys()):
    r = all_results[name]
    our = r.get('our_ndcg10', 0)
    paper = r.get('paper_ndcg10')
    delta = r.get('delta_ndcg10')

    paper_str = f'{paper:.4f}' if paper else 'N/A'
    delta_str = f'{delta:+.2f}pp' if delta is not None else 'N/A'

    print(f'{name:<20} {our:>12.4f} {paper_str:>14} {delta_str:>10}')

    total_our += our
    if paper:
        total_paper += paper
        count += 1

print('-'*80)
mean_our = total_our / len(all_results) if all_results else 0
mean_paper = total_paper / count if count > 0 else 0
mean_delta = (mean_our - mean_paper) * 100 if count > 0 else 0
print(f'{\"MEAN\":<20} {mean_our:>12.4f} {mean_paper:>14.4f} {mean_delta:>+10.2f}pp')
print('='*80)
"
