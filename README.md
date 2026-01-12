# ConstBERT Reproducibility Study

Systematic reproduction of "Efficient Constant-Space Multi-Vector Retrieval" (ECIR 2025) for SIGIR A* reproducibility paper.

**Paper**: Sean MacAvaney, Antonio Mallia, Nicola Tonellotto. *Efficient Constant-Space Multi-Vector Retrieval*. ECIR 2025.  
**Model**: `pinecone/ConstBERT` on HuggingFace  
**Target**: Table 1 results (MS-MARCO Dev Set, ConstBERT32)

## Quick Start

```bash
# 1. Setup environment
conda create -n constbert python=3.10 -y
conda activate constbert
pip install -r requirements.txt

# 2. Verify setup
python tests/test_setup.py

# 3. Run main experiment
python experiments/exp1_dev_eval.py
```

## Project Structure

```
constbert-reproduce/
├── data/                       # Data loading utilities
│   └── loaders.py              # MS-MARCO dataset loaders
├── evaluation/                 # Evaluation metrics
│   └── metrics.py              # MRR, Recall, NDCG implementations
├── models/                     # Model wrappers
│   └── constbert_wrapper.py    # ConstBERT wrapper with MaxSim
├── experiments/                # Experiment scripts
│   └── exp1_dev_eval.py        # MS-MARCO Dev evaluation
├── tests/                      # Validation tests
│   ├── test_setup.py           # Environment verification
│   ├── test_small_scale.py     # Pipeline validation (10K docs)
│   ├── test_paths.py           # Path configuration test
│   └── verify_maxsim.py        # MaxSim correctness verification
├── results/                    # Evaluation results (local)
├── ConstBERT/                  # Original author's code (reference only)
├── ConstBERT_summary.md        # Paper summary
├── PLAN.md                     # Reproduction plan
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Data & Storage

**MS-MARCO Dataset** (required):
```
/media/12TB/shared/datasets/raw/msmarco-passage/
├── collection.tsv          # 8.8M passages
├── queries.dev.small.tsv   # 6,980 queries
└── qrels.dev.small.tsv     # Relevance judgments
```

**Generated Outputs**:
- Models: `/media/12TB/shared/models/huggingface/` (HuggingFace cache)
- Embeddings: `/media/12TB/shared/datasets/processed/msmarco-passage/constbert_msmarco_embeddings.npy` (~17GB)
- Index: `/media/12TB/shared/datasets/indices/constbert_msmarco_index.pkl` (~17GB)
- Results: `./results/exp1_dev_results.json` (metrics)

## Key Commands

**Environment Setup:**
```bash
python tests/test_setup.py          # Verify all dependencies
```

**Validation (Optional but Recommended):**
```bash
python tests/test_small_scale.py    # Test on 10K docs (~2 min)
python tests/verify_maxsim.py       # Verify MaxSim correctness
```

**Main Experiment:**
```bash
# First run (encode all 8.8M docs, ~3.5 hours)
python experiments/exp1_dev_eval.py

# Subsequent runs (reuse cached embeddings, ~25 min)
python experiments/exp1_dev_eval.py --load-index
```

**Custom Paths:**
```bash
python experiments/exp1_dev_eval.py \
  --data-dir /path/to/msmarco \
  --embeddings-dir /path/to/embeddings \
  --index-dir /path/to/indices \
  --results-dir ./results
```

## Target Metrics (Paper: ConstBERT32)

### MS-MARCO Dev Set (In-Domain)

| Metric | Paper | Our Result | Diff |
|--------|-------|-----------|------|
| MRR@10 | 39.04 | **38.99** | -0.05% ✓ |
| Recall@50 | 85.86 | **85.35** | -0.51% ✓ |
| Recall@200 | 93.72 | **92.08** | -1.64% ⚠️ |
| Recall@1000 | 96.34 | **92.85** | -3.49% ⚠️ |

### TREC Deep Learning Track (In-Domain)

| Dataset | Metric | Paper | Our Result | Diff |
|---------|--------|-------|-----------|------|
| TREC DL 2019 | NDCG@10 | 73.14 | **68.29** | -4.85% ⚠️ |
| TREC DL 2020 | NDCG@10 | 73.29 | **69.30** | -3.99% ⚠️ |

### TREC Tip-of-the-Tongue 2025 (Out-of-Domain - NEW)

| Metric | ToT Result | MS-MARCO Result | Degradation |
|--------|-----------|-----------------|-------------|
| MRR@10 | **4.27%** | 38.99% | **-89.0%** ⚠️⚠️⚠️ |
| Recall@50 | **11.41%** | 85.35% | **-86.6%** ⚠️⚠️⚠️ |
| Recall@1000 | **25.72%** | 92.85% | **-72.3%** ⚠️⚠️⚠️ |

**Status:** 
- ✅ MS-MARCO primary metric (MRR@10) successfully reproduced
- ⚠️ TREC DL gaps due to FAISS IVF vs PLAID approximation (confirmed by authors)
- ⚠️ Storage claims (11 GB) require PLAID quantization (not reproducible with IVF)
- ⚠️ **NEW:** Severe performance drop on ToT reveals domain-specific limitations

**Key Findings:**
1. ConstBERT effectiveness validated on MS-MARCO-style tasks
2. Alternative retrieval backend (FAISS IVF) successfully demonstrated
3. **Out-of-domain generalization is poor** - 9x worse on long descriptive queries
4. Trade-off: Efficiency ↔ Generalization

See [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md) for full analysis and author clarifications.

## Implementation Details

### MaxSim Scoring (Faithful to Paper)

Formula: `s(q, d) = Σᵢ₌₁ᴺ max_{j=1,...,C} qᵢᵀδⱼ`

- For each query token i, find max dot product with all C document vectors
- Sum these maximums over all query tokens
- Our vectorized implementation is mathematically identical (verified: error < 10⁻¹³)

### Key Design Principles

1. **No Model Modification**: Use pre-trained `pinecone/ConstBERT` as-is
2. **Faithful Reproduction**: Match paper's exact metrics and evaluation setup
3. **Optimized but Correct**: Vectorized MaxSim for speed without changing math
4. **Caching**: Save embeddings to avoid redundant computation

## Documents

- [ConstBERT_summary.md](ConstBERT_summary.md): Detailed paper summary
- [PLAN.md](PLAN.md): Complete reproduction plan
- [ConstBERT/](ConstBERT/): Original author's code (reference, not modified)

## Expected Runtime

| Phase | Time | Details |
|-------|------|---------|
| Document Encoding | ~3 hours | 8.8M docs @ 900 docs/sec (GPU) |
| Index Building (IVF) | ~5 minutes | FAISS IVF training + adding vectors |
| Query Encoding | ~6 seconds | 6,980 queries |
| Retrieval | ~3.3 hours | 6,980 queries @ 1.7s/query (IVF) |
| **Total (first run)** | **~6.5 hours** | With FAISS IVF optimization |
| **Subsequent runs** | **~3.3 hours** | Reuse embeddings & index |

**Note**: Paper likely used brute-force MaxSim (~300+ hours). We use FAISS IVF approximation for computational feasibility.

## Troubleshooting

**Missing ujson**: `pip install ujson`  
**CUDA OOM**: Use `--batch-size 16` or `--batch-size 8`  
**Data not found**: Update `--data-dir /path/to/msmarco-passage`  
**HF download fails**: `huggingface-cli login`

## Citation

```bibtex
@inproceedings{macavaney2025constbert,
  title={Efficient Constant-Space Multi-Vector Retrieval},
  author={MacAvaney, Sean and Mallia, Antonio and Tonellotto, Nicola},
  booktitle={ECIR},
  year={2025}
}
```
