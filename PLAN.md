# ConstBERT Reproducibility Study — Research Plan

**Paper**: "Efficient Constant-Space Multi-Vector Retrieval" (ECIR 2025)  
**Authors**: Sean MacAvaney, Antonio Mallia, Nicola Tonellotto  
**Model**: `pinecone/ConstBERT` (C=32, dim=128)  
**Status**: Phase 2 Complete — Phase 3 (Paper Writing) In Progress

---

## Phase 1: Core Reproduction ✅ COMPLETED

### 1.1 Environment & Data Setup ✅
- [x] Conda environment with PyTorch, Transformers, FAISS
- [x] MS-MARCO Passage dataset (8.8M documents, 6,980 dev queries)
- [x] TREC DL 2019/2020 queries and qrels
- [x] Model verification (C=32 vectors, dim=128)

### 1.2 MS-MARCO Reproduction ✅
- [x] Implement MaxSim scoring (verified error < 10⁻¹³)
- [x] FAISS IVF retrieval (nlist=4096, nprobe=128)
- [x] Evaluate on dev set → **MRR@10: 38.99%** (paper: 39.04%)

### 1.3 TREC DL Evaluation ✅
- [x] DL 2019: NDCG@10 = 68.29% (paper: 73.14%)
- [x] DL 2020: NDCG@10 = 69.30% (paper: 73.29%)

### 1.4 Storage Analysis ✅
- [x] Measure embedding sizes (67.5 GB float16)
- [x] Identify 6× gap with paper's 11 GB claim
- [x] Contact authors for clarification → **PLAID is the answer**

### 1.5 PLAID Integration ✅
- [x] Clone and integrate ColBERT/PLAID
- [x] Build PLAID index (5.6 GB, 32K centroids)
- [x] Evaluate MS-MARCO with PLAID → **MRR@10: 31.09%**
- [x] Discover PLAID-ConstBERT incompatibility

---

## Phase 2: Extended Evaluation ✅ COMPLETED

### 2.1 TREC DL with PLAID ✅
- [x] DL 2019 PLAID: NDCG@10 = 59.53% (FAISS: 68.29%)
- [x] DL 2020 PLAID: NDCG@10 = 60.90% (FAISS: 69.30%)
- [x] Confirmed PLAID degradation is worse on TREC DL

### 2.2 ToT 2025 Zero-Shot ✅
- [x] Build ToT corpus embeddings (6.4M Wikipedia articles)
- [x] FAISS IVF evaluation → **MRR@10: 4.27%**
- [x] PLAID evaluation → **MRR@10: 0.94%**
- [x] Discovered catastrophic OOD generalization failure

### 2.3 Exact MaxSim Verification ✅
- [x] Brute-force on 200-query sample → MRR@10: 37.44%
- [x] Confirmed paper claims are achievable with exact search

---

## Phase 3: Analysis & Paper Writing ⏳ IN PROGRESS

### 3.1 Result Analysis
- [ ] Create publication-ready tables and figures
- [ ] Generate effectiveness-storage trade-off plots
- [ ] Error analysis: query-level failure modes
- [ ] Case studies: why specific queries fail on ToT

### 3.2 Statistical Significance
- [ ] Multiple runs with different random seeds
- [ ] Compute confidence intervals
- [ ] Statistical tests (paired t-test, Wilcoxon)

### 3.3 Additional Experiments (Optional)
- [ ] C parameter sensitivity (C ∈ {8, 16, 32, 64, 128})
- [ ] Reranking pipeline (ESPLADE + ConstBERT)
- [ ] Comparison with 2024-2025 SOTA models

### 3.4 Paper Structure
1. Introduction: Reproducibility importance in IR
2. Related Work: Multi-vector retrieval, PLAID
3. Methodology: Reproduction protocol
4. Results: 2×4 experiment matrix
5. Analysis: PLAID incompatibility, OOD generalization
6. Discussion: Lessons learned, recommendations
7. Conclusion: Impact and future work

### 3.5 Reproducibility Package
- [ ] Docker container
- [ ] Exact software versions (pip freeze)
- [ ] Random seeds and hyperparameters
- [ ] CI/CD for automated validation

---

## Experiment Matrix

### Completed (2×4 Grid)

| Dataset | FAISS IVF | PLAID ncells=16 | Status |
|---------|-----------|-----------------|--------|
| MS-MARCO Dev | 38.99% MRR | 31.09% MRR | ✅ |
| TREC DL 2019 | 68.29% NDCG | 59.53% NDCG | ✅ |
| TREC DL 2020 | 69.30% NDCG | 60.90% NDCG | ✅ |
| ToT 2025 | 4.27% MRR | 0.94% MRR | ✅ |

### Pending

| Experiment | Priority | Estimated Time |
|------------|----------|----------------|
| Statistical significance (5 seeds) | High | ~30 hours |
| C parameter sweep | Medium | ~20 hours |
| ESPLADE + ConstBERT reranking | Low | ~10 hours |

---

## Data Paths

```
/media/12TB/shared/datasets/
├── raw/
│   ├── msmarco-passage/           # 8.8M passages
│   └── trec-tot-2025/             # 6.4M Wikipedia
├── processed/
│   └── msmarco-passage/constbert_msmarco_embeddings.npy (67.5 GB)
└── indices/
    ├── msmarco-passage/
    │   ├── constbert_faiss_index/
    │   └── constbert_plaid_index/ (5.6 GB)
    └── trec-tot-2025/
        ├── constbert_tot_faiss_index/
        └── constbert_tot_plaid_index/
```

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Core Reproduction | 2 weeks | ✅ Done |
| Phase 2: Extended Evaluation | 1 week | ✅ Done |
| Phase 3: Paper Writing | 2 weeks | ⏳ In Progress |
| **Total** | **5 weeks** | |

---

## Hardware

- **GPU**: NVIDIA RTX 6000 Ada (48 GB VRAM)
- **RAM**: 256 GB
- **Storage**: 12 TB NVMe array
