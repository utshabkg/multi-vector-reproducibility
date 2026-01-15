# ConstBERT Reproducibility Study

Systematic reproducibility study of **"Efficient Constant-Space Multi-Vector Retrieval"** (ECIR 2025) for SIGIR A* reproducibility track.

## Quick Results

| Dataset | Metric | Paper | Our Result | Status |
|---------|--------|-------|------------|--------|
| MS-MARCO Dev | MRR@10 | 39.04% | **38.99%** | ✅ |
| TREC DL 2019 | NDCG@10 | 73.14% | 68.29% | ⚠️ |
| TREC DL 2020 | NDCG@10 | 73.29% | 69.30% | ⚠️ |
| ToT 2025 | MRR@10 | — | 4.27% | ❌ OOD |

**Key Finding**: PLAID retrieval is incompatible with ConstBERT's fixed 32-vector representation, causing 8-78% degradation.

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n constbert python=3.10 -y
conda activate constbert

# Install dependencies
pip install -r requirements.txt

# Install ColBERT for PLAID support
pip install -e external/ColBERT
```

### 2. Verify Installation

```bash
python tests/test_setup.py
```

### 3. Run Experiments

```bash
# MS-MARCO with FAISS IVF (main experiment)
python experiments/01_msmarco_faiss_eval.py

# TREC DL 2019/2020
python experiments/02_trec_dl_faiss_eval.py

# ToT 2025 Zero-Shot
python experiments/04_tot_faiss_eval.py

# With PLAID (requires PLAID index)
python experiments/05_msmarco_plaid_eval.py
```

---

## Project Structure

```
constbert-reproduce/
├── data/                           # Data loading utilities
│   ├── loaders.py                  # MS-MARCO data loader
│   └── tot_loader.py               # TREC ToT data loader
├── models/                         # Model wrappers
│   ├── constbert_wrapper.py        # ConstBERT with MaxSim
│   └── faiss_index.py              # FAISS IVF index
├── evaluation/                     # Evaluation metrics
│   └── metrics.py                  # MRR, NDCG, Recall
├── experiments/                    # Experiment scripts
│   ├── 01_msmarco_faiss_eval.py    # MS-MARCO + FAISS IVF
│   ├── 02_trec_dl_faiss_eval.py    # TREC DL + FAISS IVF
│   ├── 03_storage_analysis.py      # Storage comparison
│   ├── 04_tot_faiss_eval.py        # ToT + FAISS IVF
│   ├── 05_msmarco_plaid_eval.py    # MS-MARCO + PLAID
│   ├── 06_msmarco_plaid_ncells.py  # PLAID parameter sweep
│   ├── 07_exact_maxsim_verify.py   # Exact MaxSim verification
│   ├── 08_trec_dl_plaid_eval.py    # TREC DL + PLAID
│   ├── 09_tot_plaid_eval.py        # ToT + PLAID
│   ├── build_plaid_index_msmarco.py
│   └── build_plaid_index_tot.py
├── tests/                          # Validation tests
│   ├── test_setup.py               # Environment verification
│   ├── test_small_scale.py         # Pipeline validation
│   ├── test_paths.py               # Path configuration
│   └── verify_maxsim.py            # MaxSim correctness
├── results/                        # Experiment results (JSON)
├── logs/                           # Experiment logs
├── external/                       # External dependencies
│   └── ColBERT/                    # ColBERT with PLAID
├── PLAN.md                         # Experiment plan & results
├── REPRODUCIBILITY_REPORT.md       # Detailed analysis
└── requirements.txt                # Python dependencies
```

---

## Data Requirements

### MS-MARCO Passage

```
/media/12TB/shared/datasets/raw/msmarco-passage/
├── collection.tsv          # 8.8M passages (2.9 GB)
├── queries.dev.small.tsv   # 6,980 queries
└── qrels.dev.small.tsv     # Relevance judgments
```

### TREC DL 2019/2020

```
/media/12TB/shared/datasets/raw/msmarco-passage/
├── queries.dl2019.tsv      # 43 queries
├── queries.dl2020.tsv      # 54 queries
├── qrels.dl2019.txt        # Graded relevance (0-3)
└── qrels.dl2020.txt
```

### TREC ToT 2025

```
/media/12TB/shared/datasets/raw/trec-tot-2025/
├── trec-tot-2025-corpus.jsonl    # 6.4M Wikipedia articles
├── queries/test-2025-queries.jsonl
└── qrel/test-2025-qrel.txt
```

---

## Generated Artifacts

### Embeddings & Indices

```
/media/12TB/shared/datasets/
├── processed/msmarco-passage/
│   └── constbert_msmarco_embeddings.npy   # 67.5 GB (float16)
└── indices/
    ├── msmarco-passage/
    │   ├── constbert_faiss_index/          # FAISS IVF
    │   └── constbert_plaid_index/          # 5.6 GB PLAID
    └── trec-tot-2025/
        ├── constbert_tot_faiss_index/
        └── constbert_tot_plaid_index/
```

---

## Experiment Descriptions

| # | Experiment | Description | Runtime |
|---|------------|-------------|---------|
| 01 | MS-MARCO FAISS | Primary reproduction (MRR@10) | ~6.5h |
| 02 | TREC DL FAISS | DL 2019/2020 evaluation | ~2 min |
| 03 | Storage | Compare storage methods | ~1 min |
| 04 | ToT FAISS | Zero-shot evaluation | ~2h |
| 05 | MS-MARCO PLAID | PLAID retrieval | ~4h |
| 06 | PLAID ncells | Parameter sensitivity | ~1h |
| 07 | Exact MaxSim | Ground truth verification | ~2h |
| 08 | TREC DL PLAID | DL with PLAID | ~5 min |
| 09 | ToT PLAID | ToT with PLAID | ~3 min |

---

## Key Configuration

### FAISS IVF Parameters
```python
nlist = 4096          # Number of clusters
nprobe = 128          # Clusters to search
candidate_mult = 10   # Candidates per query token
```

### PLAID Parameters
```python
ncells = 16           # Centroids per query token
ndocs = 8192          # Candidate documents
nbits = 1             # Residual quantization bits
```

---

## Hardware

- **GPU**: NVIDIA RTX 6000 Ada (48 GB VRAM)
- **RAM**: 256 GB
- **Storage**: 12 TB NVMe array

---

## Citation

```bibtex
@inproceedings{macavaney2025constbert,
  title={Efficient Constant-Space Multi-Vector Retrieval},
  author={MacAvaney, Sean and Mallia, Antonio and Tonellotto, Nicola},
  booktitle={ECIR},
  year={2025}
}
```

---

## Documentation

- [PLAN.md](PLAN.md) — Experiment plan and results summary
- [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md) — Detailed analysis
- [ConstBERT.pdf](ConstBERT.pdf) — Original paper

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ujson` | `pip install ujson` |
| CUDA OOM | Reduce batch size: `--batch-size 16` |
| Data not found | Update paths in experiment scripts |
| PLAID index missing | Run `build_plaid_index_*.py` first |

---

## License

This reproducibility study is for academic purposes. See original paper for model license.
