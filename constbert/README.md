# ConstBERT Experiments

This directory contains all experiment code for evaluating ConstBERT (`pinecone/ConstBERT`), a multi-vector dense retrieval model with a fixed 32-vector document representation.

## Core Library

| File | Description |
|------|-------------|
| `models/constbert_wrapper.py` | ConstBERT model wrapper for encoding queries and documents |
| `models/faiss_index.py` | FAISS-IVF index for approximate nearest neighbor search |
| `data/loaders.py` | MS MARCO and TREC DL data loaders |
| `data/tot_loader.py` | TREC ToT 2025 data loader |
| `evaluation/metrics.py` | MRR@k, Recall@k, nDCG@k, MAP computation |

## Experiments by Research Question

### RQ1: Implementation Correctness (`experiments/rq1/`)

Validates ConstBERT's implementation by comparing against original paper metrics on standard benchmarks.

- `01_msmarco_faiss_eval.py` — MS MARCO Passage Ranking evaluation (FAISS-IVF backend)
- `02_trec_dl_faiss_eval.py` — TREC DL 2019 & 2020 evaluation
- `03_storage_analysis.py` — Embedding storage size analysis

### RQ2: PLAID Engine Compatibility (`experiments/rq2/`)

Tests whether ConstBERT's fixed 32-vector design is compatible with ColBERT-v2's PLAID retrieval engine.

- `build_plaid_index_msmarco.py` — Build PLAID index from ConstBERT embeddings
- `build_plaid_index_tot.py` — Build PLAID index for ToT corpus
- `05_msmarco_plaid_eval.py` — MS MARCO evaluation with PLAID backend
- `06_msmarco_plaid_ncells.py` — PLAID nprobe/ncells parameter analysis
- `07_exact_maxsim_verify.py` — Verify exact MaxSim matches PLAID scores
- `08_trec_dl_plaid_eval.py` — TREC DL evaluation with PLAID backend
- `17_centroid_coverage_analysis.py` — Centroid assignment coverage analysis
- `18_plaid_parameter_sweep.py` — PLAID configuration parameter sweep

### RQ3: Cross-Domain Generalization (`experiments/rq3/`)

Evaluates zero-shot transfer performance across 13 BEIR benchmark datasets.

- `11_beir_evaluation.py` — ConstBERT FAISS evaluation on all BEIR datasets
- `12-1_build_beir_plaid_indices.py` — Build PLAID indices for BEIR
- `12-2_build_ivf_for_beir.py` — Build IVF indices for BEIR PLAID
- `12-3_beir_plaid_eval.py` — ConstBERT PLAID evaluation on BEIR
- `13_beir_colbertv2_evaluation.py` — ColBERT-v2 BEIR evaluation (comparison)
- `15_verify_faiss_hypothesis.py` — FAISS nlist sensitivity hypothesis test
- `compare_beir_backends.py` — Compare FAISS vs PLAID backends on BEIR

### RQ4: Long-Query Retrieval (`experiments/rq4/`)

Examines ConstBERT's behavior on verbose, descriptive queries from TREC ToT 2025.

- `04_tot_faiss_eval.py` — ToT evaluation with FAISS backend
- `09_tot_plaid_eval.py` — ToT evaluation with PLAID backend
- `15_query_length_ablation.py` — Query length truncation ablation study
- `16_exact_maxsim_tot.py` — Exact MaxSim verification on ToT

### RQ5: Domain-Specific Fine-Tuning (`experiments/rq5/`)

Fine-tunes both ConstBERT and ColBERT-v2 on TREC ToT training data.

- `10_generate_triples_simple.py` — Generate BM25-negative training triples
- `10_finetune_constbert_tot.py` — Fine-tune ConstBERT on ToT
- `10_eval_finetuned_tot_test.py` — Evaluate fine-tuned ConstBERT
- `run_multiseed_finetuning.py` — Multi-seed fine-tuning runner
- `eval_multiseed_tot.py` — Evaluate multi-seed ConstBERT models
- `eval_multiseed_colbert.py` — Evaluate multi-seed ColBERT-v2 models
- `run_ablation_constbert.py` — ConstBERT fine-tuning ablation
- `run_ablation_colbert.py` — ColBERT-v2 fine-tuning ablation
- `eval_colbert_base_test.py` — ColBERT-v2 base (unfinetuned) evaluation on ToT
