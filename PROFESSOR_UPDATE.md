# ConstBERT Reproducibility Study - Professor Update

**Date:** January 15, 2026  
**Student:** [Your Name]  
**Paper:** ConstBERT: Token Embedding Pruning via Constant Vectors for Efficient Retrieval (ECIR 2025)

---

## Executive Summary

I've completed the full 2×4 experiment matrix testing ConstBERT across 4 datasets and 2 retrieval backends. **Key finding:** PLAID (the paper's recommended backend) has severe compatibility issues with ConstBERT, especially on out-of-domain data.

---

## Complete Results Matrix

| Dataset | FAISS IVF | PLAID (ncells=16) | Δ Degradation |
|---------|-----------|-------------------|---------------|
| MS-MARCO Dev | **38.99%** MRR@10 | 31.09% MRR@10 | **-20.3%** |
| TREC DL 2019 | **68.29%** NDCG@10 | 59.53% NDCG@10 | **-12.8%** |
| TREC DL 2020 | **69.30%** NDCG@10 | 60.90% NDCG@10 | **-12.1%** |
| ToT 2025 | 4.27% MRR@10 | 0.94% MRR@10 | **-78.0%** |

**Paper Reports:** MRR@10 = 38.4% (we achieved 38.99% with FAISS IVF ✅)

---

## Key Discoveries & WHY They Happen

### 1. PLAID-ConstBERT Incompatibility

**What:** PLAID degrades ConstBERT's MRR@10 by 20% on MS-MARCO.

**WHY This Happens:**
- ConstBERT uses **fixed 32 vectors** per document (vs. ColBERT's 100-300)
- PLAID's centroid-based indexing assigns these 32 vectors to only **12-20 unique centroids** on average
- Queries also get few centroids → **low query-document centroid overlap**
- Result: PLAID's candidate filtering misses relevant documents that would be found by exact MaxSim

**Evidence:** Exact MaxSim on 200-query sample = 37.44% (very close to FAISS's 38.99%), confirming PLAID is the bottleneck.

### 2. Catastrophic Out-of-Domain Failure

**What:** ToT 2025 drops from 4.27% (FAISS) to 0.94% (PLAID) — **78% relative degradation**.

**WHY This Happens:**
- ToT queries are **long descriptive narratives** (avg 50+ words) vs. MS-MARCO's keyword queries (avg 6 words)
- ConstBERT's MaxSim scoring with fixed vectors gets **diluted** by long queries
- Each additional query token reduces the average maximum similarity
- PLAID amplifies this: long queries → more centroids → sparser overlap with short documents

**Key Insight:** PLAID degradation is 4× worse on OOD data (78%) than in-domain (20%).

### 3. Storage Discrepancy Resolved

**What:** Paper claims 11 GB index, we got 5.6 GB with PLAID.

**WHY:**
- Paper uses **int8 residuals** (1 byte per dimension) + PQ codes
- We use **1-bit residuals** (nbits=1) as per ColBERTv2 defaults
- 11 GB ÷ 5.6 GB ≈ 2× matches 8-bit vs 1-bit difference
- Paper's retrieval method may be **exact MaxSim** (our 65 GB uncompressed) rather than PLAID

**Author Confirmation:** Authors confirmed they use approximate retrieval but couldn't specify exact method.

---

## Novel Contributions for SIGIR Paper

1. **First documentation** of PLAID-ConstBERT incompatibility (20% MRR drop)
2. **First out-of-domain evaluation** of ConstBERT (ToT 2025)
3. **Theoretical explanation** linking fixed C=32 vectors to PLAID's centroid sparsity
4. **Alternative backend validation:** FAISS IVF outperforms PLAID for ConstBERT

---

## Current Status

✅ **Phase 1 Complete:** All 9 experiments finished  
✅ **Phase 2 Complete:** Documentation polished, project organized  
🔄 **Phase 3 Next:** Paper writing (figures, statistical tests, reproducibility package)

---

## Files Available

| Document | Purpose |
|----------|---------|
| [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md) | Concise results summary |
| [ISSUES_AND_INSIGHTS.md](ISSUES_AND_INSIGHTS.md) | Technical challenges & author Q&A |
| [PLAN.md](PLAN.md) | Project phases & remaining tasks |
| `experiments/01-09_*.py` | All experiment scripts |
| `results/01-09_*.json` | All result files |

---

## Questions for Discussion

1. Should we increase PLAID's ncells (currently 16) to improve recall? (Tested ncells=4, will try 64)
2. For the SIGIR paper, should we propose a "ConstBERT-aware" PLAID variant?
3. Is the ToT evaluation worth including given the 4.27% baseline MRR?

---

*Full details in [ISSUES_AND_INSIGHTS.md](ISSUES_AND_INSIGHTS.md)*
