# ColBERT-v2 Experiments

This directory contains scripts for evaluating ColBERT-v2 (`colbert-ir/colbertv2.0`) as a comparison baseline. ColBERT-v2 uses variable-length (~100-200) token-level vectors per document.

## Environment Setup

ColBERT-v2 experiments require a separate conda environment:

```bash
conda env create -f environment.yml
conda activate colbertv2
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/index_colbertv2_msmarco.sh` | Index MS MARCO with ColBERT-v2 |
| `scripts/index_colbertv2_tot.sh` | Index TREC ToT with ColBERT-v2 |
| `scripts/run_eval_msmarco.sh` | Evaluate on MS MARCO |
| `scripts/run_eval_tot.sh` | Evaluate on TREC ToT |
| `scripts/run_search.py` | Generic ColBERT-v2 search script |
| `scripts/compute_trec_metrics.py` | Compute TREC-style metrics |
| `scripts/generate_triples_bm25.py` | Generate BM25-negative training triples |
| `scripts/train_colbert_tot_triples.py` | Fine-tune ColBERT-v2 on ToT |
| `scripts/run_train_colbertv2_tot.sh` | Training runner script |
| `scripts/env_setup.sh` | Environment setup helper |

## Results

Pre-computed evaluation results are in `results/`:
- `msmarco_eval_metrics.json` — MS MARCO evaluation metrics
- `tot_dev*_eval_metrics.json` — ToT dev split metrics
- `tot_*_finetuned_eval.json` — Fine-tuned model evaluation
