# Reproducibility Study of Multi-Vector Retrieval Models

Code and results for our SIGIR 2026 Reproducibility Track submission examining ConstBERT and ColBERT-v2 multi-vector dense retrieval models.

## Repository Structure

```
├── constbert/                    # ConstBERT experiments (main focus)
│   ├── models/                   # Core model wrappers
│   │   ├── constbert_wrapper.py  # ConstBERT model wrapper
│   │   └── faiss_index.py        # FAISS-IVF index implementation
│   ├── data/                     # Data loading utilities
│   │   ├── loaders.py            # MS MARCO & TREC DL data loaders
│   │   └── tot_loader.py         # TREC ToT data loader
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py            # MRR, Recall, nDCG, MAP computation
│   ├── experiments/              # Experiment scripts organized by RQ
│   │   ├── rq1/                  # RQ1: Implementation correctness (FAISS)
│   │   ├── rq2/                  # RQ2: PLAID engine compatibility
│   │   ├── rq3/                  # RQ3: Cross-domain generalization (BEIR)
│   │   ├── rq4/                  # RQ4: Long-query retrieval (TREC ToT)
│   │   └── rq5/                  # RQ5: Domain-specific fine-tuning
│   ├── results/                  # All experiment results (JSON)
│   │   └── multiseed/            # Multi-seed fine-tuning results
│   └── scripts/                  # Shell scripts for running experiments
│
├── colbertv2/                    # ColBERT-v2 experiments (comparison baseline)
│   ├── experiments/              # Experiment scripts organized by RQ
│   │   ├── rq1/                  # RQ1: MS MARCO evaluation
│   │   ├── rq3/                  # RQ3: BEIR evaluation
│   │   ├── rq4/                  # RQ4: TREC ToT evaluation
│   │   └── rq5/                  # RQ5: Fine-tuning on ToT
│   ├── scripts/                  # Indexing, search, training scripts
│   ├── results/                  # ColBERT-v2 evaluation results
│   └── environment.yml           # Conda environment specification
│
├── colbertv1/                    # ColBERT-v1 (supplementary, not in paper)
│   ├── src/                      # ColBERT-v1 pipeline implementation
│   └── results/                  # PyTerrier-based evaluation results
│
├── scripts/                      # Data download scripts
│   ├── download_msmarco.sh       # Download MS MARCO Passage Ranking
│   ├── download_beir.sh          # Download BEIR benchmark (13 datasets)
│   └── download_tot.sh           # TREC ToT 2025 download instructions
│
├── requirements.txt              # Python dependencies
└── .gitignore
```

## Research Questions → Experiment Mapping

| RQ | Description | ConstBERT Scripts | ColBERT-v2 Scripts |
|----|------------|-------------------|-------------------|
| **RQ1** | Implementation correctness on standard benchmarks | `constbert/experiments/rq1/01_msmarco_faiss_eval.py` | `colbertv2/scripts/run_eval_msmarco.sh` |
| | | `constbert/experiments/rq1/02_trec_dl_faiss_eval.py` | |
| | | `constbert/experiments/rq1/03_storage_analysis.py` | |
| **RQ2** | PLAID engine compatibility | `constbert/experiments/rq2/05_msmarco_plaid_eval.py` | — |
| | | `constbert/experiments/rq2/06_msmarco_plaid_ncells.py` | |
| | | `constbert/experiments/rq2/07_exact_maxsim_verify.py` | |
| | | `constbert/experiments/rq2/08_trec_dl_plaid_eval.py` | |
| | | `constbert/experiments/rq2/17_centroid_coverage_analysis.py` | |
| | | `constbert/experiments/rq2/18_plaid_parameter_sweep.py` | |
| | | `constbert/experiments/rq2/build_plaid_index_msmarco.py` | |
| | | `constbert/experiments/rq2/build_plaid_index_tot.py` | |
| **RQ3** | Cross-domain generalization (BEIR) | `constbert/experiments/rq3/11_beir_evaluation.py` | `colbertv2/scripts/run_eval_msmarco.sh` (BEIR) |
| | | `constbert/experiments/rq3/12-1_build_beir_plaid_indices.py` | |
| | | `constbert/experiments/rq3/12-2_build_ivf_for_beir.py` | |
| | | `constbert/experiments/rq3/12-3_beir_plaid_eval.py` | |
| | | `constbert/experiments/rq3/13_beir_colbertv2_evaluation.py` | |
| | | `constbert/experiments/rq3/15_verify_faiss_hypothesis.py` | |
| | | `constbert/experiments/rq3/compare_beir_backends.py` | |
| **RQ4** | Long-query retrieval (TREC ToT) | `constbert/experiments/rq4/04_tot_faiss_eval.py` | `colbertv2/scripts/run_eval_tot.sh` |
| | | `constbert/experiments/rq4/09_tot_plaid_eval.py` | |
| | | `constbert/experiments/rq4/15_query_length_ablation.py` | |
| | | `constbert/experiments/rq4/16_exact_maxsim_tot.py` | |
| **RQ5** | Domain-specific fine-tuning | `constbert/experiments/rq5/10_finetune_constbert_tot.py` | `colbertv2/scripts/train_colbert_tot_triples.py` |
| | | `constbert/experiments/rq5/10_generate_triples_simple.py` | `colbertv2/scripts/run_train_colbertv2_tot.sh` |
| | | `constbert/experiments/rq5/10_eval_finetuned_tot_test.py` | |
| | | `constbert/experiments/rq5/run_multiseed_finetuning.py` | |
| | | `constbert/experiments/rq5/eval_multiseed_tot.py` | |
| | | `constbert/experiments/rq5/eval_multiseed_colbert.py` | |
| | | `constbert/experiments/rq5/run_ablation_constbert.py` | |
| | | `constbert/experiments/rq5/run_ablation_colbert.py` | |

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- ~50GB disk space for datasets

### Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt

# For ColBERT-v2 experiments (separate environment recommended)
conda env create -f colbertv2/environment.yml
conda activate colbertv2
```

### Data Download

```bash
# MS MARCO Passage Ranking (required for RQ1, RQ2, RQ3)
bash scripts/download_msmarco.sh

# BEIR benchmark (required for RQ3)
bash scripts/download_beir.sh

# TREC ToT 2025 (required for RQ4, RQ5) — requires registration
bash scripts/download_tot.sh
```

## Running Experiments

### RQ1: Implementation Correctness

```bash
# ConstBERT on MS MARCO (FAISS-IVF backend)
python constbert/experiments/rq1/01_msmarco_faiss_eval.py

# ConstBERT on TREC DL 2019/2020
python constbert/experiments/rq1/02_trec_dl_faiss_eval.py

# Storage analysis
python constbert/experiments/rq1/03_storage_analysis.py
```

### RQ2: PLAID Engine Compatibility

```bash
# Build PLAID index
python constbert/experiments/rq2/build_plaid_index_msmarco.py

# Evaluate with PLAID
python constbert/experiments/rq2/05_msmarco_plaid_eval.py

# Verify exact MaxSim equivalence
python constbert/experiments/rq2/07_exact_maxsim_verify.py
```

### RQ3: Cross-Domain Generalization (BEIR)

```bash
# ConstBERT FAISS evaluation on BEIR
python constbert/experiments/rq3/11_beir_evaluation.py

# ColBERT-v2 BEIR evaluation
python constbert/experiments/rq3/13_beir_colbertv2_evaluation.py
```

### RQ4: Long-Query Retrieval (TREC ToT)

```bash
# ConstBERT on ToT (FAISS)
python constbert/experiments/rq4/04_tot_faiss_eval.py

# Query length ablation
python constbert/experiments/rq4/15_query_length_ablation.py
```

### RQ5: Domain-Specific Fine-Tuning

```bash
# Generate training triples
python constbert/experiments/rq5/10_generate_triples_simple.py

# Fine-tune ConstBERT
python constbert/experiments/rq5/10_finetune_constbert_tot.py

# Evaluate fine-tuned model
python constbert/experiments/rq5/10_eval_finetuned_tot_test.py

# Multi-seed experiments
python constbert/experiments/rq5/run_multiseed_finetuning.py
```

## Pre-computed Results

All experiment results are included as JSON files in:
- `constbert/results/` — ConstBERT results (72 files)
- `colbertv2/results/` — ColBERT-v2 results (6 files)
- `colbertv1/results/` — ColBERT-v1 supplementary results (6 files)

These enable verification of all numbers reported in the paper without re-running experiments.

## Models

| Model | Source | Embedding Dim | Vectors/Doc |
|-------|--------|:---:|:---:|
| ConstBERT | `pinecone/ConstBERT` (HuggingFace) | 128 | 32 (fixed) |
| ColBERT-v2 | `colbert-ir/colbertv2.0` (HuggingFace) | 128 | Variable (~100-200) |

## Datasets

| Dataset | Queries | Documents | Used In |
|---------|:---:|:---:|---------|
| MS MARCO Passage | 6,980 (dev.small) | 8.8M | RQ1, RQ2, RQ3 |
| TREC DL 2019 | 43 | 8.8M | RQ1 |
| TREC DL 2020 | 54 | 8.8M | RQ1 |
| BEIR (13 datasets) | Varies | Varies | RQ3 |
| TREC ToT 2025 | 150 (dev) / 150 (test) | 24.8K | RQ4, RQ5 |

## Key Findings

- **RQ1**: ConstBERT matches original paper metrics on MS MARCO (MRR@10: 0.3440) and TREC DL benchmarks
- **RQ2**: ConstBERT's fixed-length 32-vector design is compatible with ColBERT-v2's PLAID engine, achieving near-identical MRR@10 with 47-50% storage reduction
- **RQ3**: ConstBERT underperforms ColBERT-v2 on 8/13 BEIR datasets, with particularly large gaps on knowledge-intensive tasks
- **RQ4**: ConstBERT's 32-token truncation critically limits long-query retrieval on TREC ToT (MRR@10: 0.230 vs ColBERT-v2's 0.414)
- **RQ5**: Fine-tuning improves ConstBERT's ToT performance (+75% MRR@10) but does not close the gap with ColBERT-v2

## License

This code is released for academic research purposes in connection with the accompanying paper submission.
