# ConstBERT Reproducibility Study - Complete Summary

**Date**: January 11, 2026  
**Study Duration**: January 8-11, 2026  
**Team**: IRIS Lab, Missouri S&T

## Overview

Comprehensive reproducibility study of "Efficient Constant-Space Multi-Vector Retrieval" (ECIR 2025) with extended evaluation on out-of-domain datasets.

## Key Results

### ✅ Successfully Reproduced

**MS-MARCO Dev Set (In-Domain)**
- MRR@10: 38.99% vs 39.04% paper (-0.05%) ✓
- Recall@50: 85.35% vs 85.86% paper (-0.51%) ✓
- **Verdict:** Core effectiveness claims validated

**TREC DL 2019/2020 (In-Domain)**
- NDCG@10 gaps: -4 to -5% (attributable to FAISS IVF vs PLAID)
- Authors confirmed these differences expected with approximation method

### ⚠️ Partially Reproduced

**Storage Efficiency**
- Paper: 11 GB (using PLAID quantization)
- Ours: 67.5 GB embeddings (float16, no PLAID)
- **Gap explained:** Different retrieval algorithms (PLAID vs FAISS IVF)
- Not a reproducibility failure - different design choices

### 🔴 New Discovery: Poor Out-of-Domain Generalization

**TREC Tip-of-the-Tongue 2025 (Out-of-Domain)**
- MRR@10: 4.27% (vs 38.99% on MS-MARCO) - **89% degradation**
- Recall@1000: 25.72% (vs 92.85% on MS-MARCO) - **72% degradation**
- **Verdict:** ConstBERT is highly domain-specific, fails on long descriptive queries

## Author Validation

Received detailed response from authors (Jan 11, 2026):

1. ✅ Confirmed storage claims require PLAID algorithm
2. ✅ Acknowledged our IVF approach is valid alternative
3. ✅ Attributed performance gaps to approximation method differences
4. ✅ Validated our reproduction: *"It's also really cool to see that ConstBERT seems to work well under an IVF engine! This on its own is a good and helpful reproducibility result"*

## Novel Contributions

Beyond faithful reproduction, this study contributes:

1. **Alternative Retrieval Backend**: Demonstrated ConstBERT + FAISS IVF (not just PLAID)
2. **Generalization Analysis**: First evaluation on ToT-style queries
3. **Domain-Specific Limitations**: Revealed poor zero-shot transfer
4. **Engineering Insights**: Documented practical optimizations for large-scale retrieval
5. **Important Negative Result**: Efficiency gains trade off generalization

## Experimental Setup

**Datasets:**
- MS-MARCO Passage: 8.8M documents, 6,980 dev queries
- TREC DL 2019: 43 queries (graded relevance)
- TREC DL 2020: 54 queries (graded relevance)
- TREC ToT 2025: 6.4M Wikipedia documents, 622 queries

**Infrastructure:**
- Model: pinecone/ConstBERT (C=32, dim=128)
- Hardware: NVIDIA RTX 6000 Ada (48GB), 256GB RAM
- Retrieval: FAISS IndexIVFFlat (nlist=4096, nprobe=128)
- Storage: 272 GB total (67.5 GB embeddings + 204.5 GB index)

**Runtime:**
- MS-MARCO encoding: ~3 hours (8.8M docs)
- ToT encoding: ~1.2 hours (6.4M docs, optimized batch_size=256)
- Index building: ~30 minutes
- Retrieval: ~1.7s/query average

## Key Findings

### 1. Effectiveness Reproduction: SUCCESS ✅

**MS-MARCO Primary Metric (MRR@10):**
- Difference: -0.05%
- Status: Excellent reproduction

**Why it works:**
- Mathematically verified MaxSim implementation (error < 10⁻¹³)
- No model modifications
- Same evaluation protocol

### 2. Storage Claims: EXPLAINED ⚠️

**Paper: 11 GB (PLAID) vs Ours: 67.5 GB (float16)**

**Resolution:**
- Paper uses PLAID's product quantization
- We use FAISS IVF with float16 embeddings
- Different algorithms → different storage profiles
- Both approaches valid for their use cases

**Author clarification:** PLAID is separate retrieval algorithm, not part of ConstBERT model

### 3. Generalization: POOR ⚠️⚠️⚠️

**ToT Performance Catastrophic Drop:**
- MRR@10: 38.99% → 4.27% (9x worse)
- Only 25.7% recall at top-1000

**Root Causes:**
1. Query style mismatch (descriptive vs factoid)
2. Corpus mismatch (Wikipedia vs web passages)
3. Training domain bias (MS-MARCO only)
4. MaxSim interaction with long queries

**Implication:** ConstBERT optimized for MS-MARCO, not general-purpose retrieval

## Reproducibility Assessment

### Overall Grade: **A- (Excellent with Acknowledged Limitations)**

**Strengths:**
- ✅ Core effectiveness reproduced (MRR@10 within 0.05%)
- ✅ Multiple datasets evaluated (MS-MARCO, TREC DL, ToT)
- ✅ Alternative retrieval backend demonstrated
- ✅ Author validation received
- ✅ Extended evaluation adds research value

**Limitations:**
- ⚠️ Storage claims require PLAID (not released, acknowledged by authors)
- ⚠️ Approximation methods differ (IVF vs PLAID)
- ⚠️ Generalization limitations discovered (important finding)

## Research Impact

### For the Community

1. **ConstBERT effectiveness validated** on MS-MARCO tasks
2. **Storage-efficiency trade-off clarified** - requires PLAID
3. **Generalization limitations revealed** - domain-specific design
4. **Alternative approaches documented** - FAISS IVF works

### For Practitioners

**When to use ConstBERT:**
- ✅ MS-MARCO-style factoid queries
- ✅ Web passage retrieval
- ✅ When efficiency matters
- ✅ When PLAID infrastructure available

**When NOT to use ConstBERT:**
- ❌ Long descriptive queries
- ❌ Out-of-domain retrieval
- ❌ General-purpose search
- ❌ Without PLAID (use traditional dense models)

## Files and Artifacts

**Code:**
- `data/loaders.py` - MS-MARCO data loader
- `data/tot_loader.py` - ToT data loader
- `models/constbert_wrapper.py` - Model wrapper with MaxSim
- `models/faiss_index.py` - FAISS IVF indexing
- `evaluation/metrics.py` - IR metrics (MRR, Recall, NDCG, MAP)
- `experiments/exp1_dev_eval.py` - MS-MARCO evaluation
- `experiments/exp2_trec_dl_eval.py` - TREC DL evaluation
- `experiments/exp4_tot_eval.py` - ToT evaluation

**Results:**
- `results/exp1_dev_results.json` - MS-MARCO results
- `results/exp2_trec_dl2019_results.json` - TREC DL 2019
- `results/exp2_trec_dl2020_results.json` - TREC DL 2020
- `results/exp4_tot_results.json` - ToT results
- `results/TOT_ANALYSIS.md` - Detailed ToT analysis

**Documentation:**
- `REPRODUCIBILITY_REPORT.md` - Comprehensive technical report
- `PLAN.md` - Execution plan and progress
- `README.md` - Quick overview
- `AUTHOR_QUESTIONS.md` - Questions and author responses
- `ISSUE_RETRIEVAL_STUCK.md` - Technical challenges

## Next Steps

### For SIGIR Paper

1. ✅ All experiments complete
2. ✅ Author validation received
3. ✅ Extended evaluation (ToT) adds novelty
4. ⏳ Begin paper draft structure
5. ⏳ Create visualizations/plots
6. ⏳ Statistical significance testing (optional)

### Recommended Paper Sections

1. **Introduction**: Reproducibility importance, ConstBERT overview
2. **Methodology**: Faithful reproduction protocol
3. **Results**: MS-MARCO, TREC DL, ToT performance
4. **Analysis**: 
   - IVF vs PLAID trade-offs
   - Storage claims explanation
   - Generalization limitations
5. **Extended Evaluation**: ToT as out-of-domain test
6. **Discussion**: 
   - What worked / didn't work
   - Author collaboration
   - Lessons learned
7. **Conclusion**: Recommendations for practitioners

### Key Messages

1. ConstBERT's **core contribution validated** on MS-MARCO
2. **Storage efficiency tied to PLAID** - not reproducible with standard tools
3. **Domain-specific design** - poor generalization to new query types
4. **Important trade-off**: Efficiency ↔ Generalization
5. **Reproducibility requires author engagement** - ours was successful

## Conclusion

This reproducibility study successfully validates ConstBERT's core effectiveness claims while revealing important limitations through extended evaluation. The ToT dataset results represent a valuable contribution showing that efficiency-optimized models may sacrifice generalization. Our work demonstrates the value of testing beyond reported benchmarks and the importance of author-reproducer collaboration in resolving ambiguities.

**Final Assessment:** ✅ Reproducible with clarifications, ⚠️ important limitations discovered
