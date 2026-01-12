# ToT Evaluation Results - Analysis

## Performance Summary

### TREC Tip-of-the-Tongue 2025 (Zero-Shot)
**Dataset:** 6.4M Wikipedia articles, 622 test queries

| Metric | ToT Result | MS-MARCO Result | Gap |
|--------|-----------|-----------------|-----|
| **MRR@10** | **4.27%** | 38.99% | **-34.72%** ⚠️⚠️⚠️ |
| **Recall@50** | **11.41%** | 85.35% | **-73.94%** ⚠️⚠️⚠️ |
| **Recall@200** | 17.20% | 92.08% | -74.88% ⚠️⚠️⚠️ |
| **Recall@1000** | 25.72% | 92.85% | -67.13% ⚠️⚠️⚠️ |
| **NDCG@10** | 4.82% | N/A | - |
| **MAP@1000** | 17.80% | N/A | - |
| **Response Time** | 716 ms | 1,658 ms | +2.3x faster ✅ |

## Key Findings

### 🔴 Critical Performance Drop

**ConstBERT shows SEVERE degradation on ToT dataset:**
- MRR@10 drops from 39% → 4% (9x worse!)
- Recall@50 drops from 85% → 11% (7.5x worse!)
- Only 25.7% recall at top-1000 (vs 92.8% on MS-MARCO)

### 🤔 Why Such Poor Generalization?

#### 1. **Query Style Mismatch**
- **MS-MARCO queries:** Short, factoid, keyword-based
  - Example: "what is the capital of france"
  - Average length: ~10-20 words
  
- **ToT queries:** Long, descriptive, narrative "tip-of-tongue" style
  - Example: "I remember this certain type of cheese from way back when I visited a region that bordered Switzerland, famous for its rolling hills and pastoral beauty..."
  - Average length: ~100-200 words

#### 2. **Corpus Mismatch**
- **MS-MARCO:** Web passages (short, ~100-150 words)
- **ToT:** Wikipedia articles (long, ~3000 chars average)

#### 3. **Training Domain**
- ConstBERT was trained on MS-MARCO passages
- No exposure to long descriptive queries or Wikipedia-style documents
- **Zero-shot transfer fails dramatically**

#### 4. **Multi-Vector Interaction Pattern**
- MaxSim may not handle long, rambling queries well
- Many query tokens might be "filler" words in ToT queries
- MaxSim assumes all query tokens are meaningful

### ⚡ Positive Note: Speed

ToT retrieval is **2.3x faster** than MS-MARCO despite:
- Similar corpus size (6.4M vs 8.8M)
- Same retrieval infrastructure (FAISS IVF)

This is likely because:
- Index is slightly smaller (fewer documents)
- Same batch processing optimizations apply

## Implications for Reproducibility Study

### ✅ What This Tells Us

1. **ConstBERT is domain-specific**, not general-purpose
2. **Zero-shot transfer is poor** when query style differs significantly
3. **Training on MS-MARCO ≠ works on all retrieval tasks**
4. This is actually a **valuable reproducibility finding**!

### 📊 Comparison with Other Models

**Expected behavior for domain-specific models:**
- Dense retrievers trained on MS-MARCO typically see 30-50% drops on out-of-domain tasks
- **ConstBERT sees ~90% drop (MRR@10: 39% → 4%)**
- This suggests **extreme overfitting to MS-MARCO query patterns**

### 🎯 Research Contribution

This is an **important negative result** for reproducibility:
- Shows ConstBERT's limitations beyond reported benchmarks
- Highlights importance of testing on diverse datasets
- Demonstrates that efficiency gains (storage, speed) come at cost of generalization

## Recommendations for Paper

### Frame as Extended Evaluation

**Section Title:** "Generalization to Out-of-Domain Queries"

**Key Points:**
1. We tested ConstBERT on TREC ToT 2025 (zero-shot)
2. Observed severe performance degradation (MRR@10: 4% vs 39% on MS-MARCO)
3. Attributes to:
   - Different query style (descriptive vs factoid)
   - Different corpus (Wikipedia vs web passages)
   - No fine-tuning on target domain
4. Conclusion: ConstBERT is **optimized for MS-MARCO-style retrieval**, not general-purpose

**Contrast with:**
- ColBERT's generalization performance (if available)
- Traditional dense retrievers like DPR
- Cross-encoder rerankers

### Positive Framing

"While ConstBERT achieves strong effectiveness on MS-MARCO, our evaluation reveals that its compression strategy may trade off out-of-domain generalization for efficiency. This represents an important design trade-off that practitioners should consider when deploying multi-vector models in production."

## Next Steps

1. ✅ Document these findings in REPRODUCIBILITY_REPORT.md
2. ⏳ Create comparison table: MS-MARCO vs TREC DL vs ToT
3. ⏳ Update PLAN.md status
4. ⏳ Prepare visualizations (if needed)
5. ⏳ Begin paper draft with all findings

## Data for Paper

### Table: Cross-Dataset Performance

| Dataset | Corpus | Query Style | MRR@10 | Recall@1000 | Notes |
|---------|--------|-------------|--------|-------------|-------|
| MS-MARCO Dev | 8.8M web | Short, factoid | 38.99% | 92.85% | In-domain |
| TREC DL 2019 | 8.8M web | Short, factoid | - | - | NDCG@10: 68.29% |
| TREC DL 2020 | 8.8M web | Short, factoid | - | - | NDCG@10: 69.30% |
| **ToT 2025** | **6.4M wiki** | **Long, descriptive** | **4.27%** ⚠️ | **25.72%** ⚠️ | **Zero-shot** |

**Conclusion:** ConstBERT's effectiveness is highly dependent on query and corpus characteristics matching its training distribution (MS-MARCO).
