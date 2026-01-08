# Issues and Challenges in ConstBERT Reproducibility

This document tracks significant challenges encountered during reproduction that provide insights for the SIGIR A* reproducibility paper.

---

## Issue 1: Computational Feasibility - Brute-Force MaxSim ✅ RESOLVED

**Challenge**: Initial brute-force MaxSim implementation on 8.8M documents resulted in 15+ hour hang with estimated 300+ hours total runtime.

**Root Cause**: 
- Computing MaxSim for each query requires comparing against all 8.8M documents
- Each comparison: Query (N tokens) × Document (32 vectors) × 128 dimensions
- 6,980 queries × 8.8M docs = 61.4 billion MaxSim computations
- Small-scale test (10K docs): 167ms/query → Full scale (8.8M): ~2.5 min/query → ~12 days total

**Solution Implemented**:
- FAISS IVF (Inverted File Index) with nlist=4096, nprobe=128
- Two-stage retrieval: IVF finds candidates → exact MaxSim reranks
- Reduced runtime from 300+ hours to 3.3 hours (90x speedup)

**Impact on Results**:
- MRR@10: -0.05% difference (negligible) ✅
- Recall@50: -0.51% difference (acceptable) ✅
- Recall@1000: -3.49% difference (IVF approximation trade-off) ⚠️

**Insight for Paper**:
Original paper does not specify retrieval implementation details. For large-scale reproducibility (8.8M+ documents), approximation methods like FAISS IVF are necessary without specialized infrastructure. The paper should document whether they used:
1. Brute-force MaxSim (requires extensive compute)
2. Approximate nearest neighbor methods (likely PLAID)
3. Specialized hardware/optimizations

**Recommendation**: Accept FAISS IVF as reasonable engineering tradeoff, document clearly in reproducibility report.

---

## Issue 2: Storage Requirements - 24.7x Larger Than Claimed ⚠️ MAJOR GAP

**Challenge**: Our storage (272 GB total) is 24.7x larger than paper's claimed 11 GB for ConstBERT32.

### Detailed Analysis:

**Theoretical Storage** (8,841,823 documents, C=32, dim=128):
- Float32: 134.92 GB
- Float16: 67.46 GB ← **We use this**
- Int8: 33.73 GB
- **Paper claims: 11 GB**

**Our Actual Storage**:
- Raw embeddings (float16): 67.46 GB
- FAISS IVF index (float32): 137.03 GB
- Metadata (doc IDs, config): 67.54 GB
- **Total: 272.02 GB**

**Storage Breakdown**:
1. **Embeddings (67.5 GB)**: Already optimized to float16, matches theoretical minimum
2. **FAISS Index (137 GB)**: Stores all 283M vectors (8.8M docs × 32 vectors) in float32 for fast retrieval
3. **Metadata (67.5 GB)**: Pickle file includes redundant embedding copy

### Paper's 11 GB - Possible Explanations:

**Hypothesis 1: Aggressive Quantization (Most Likely)**
- Int8 quantization: 33.73 GB → still 3x larger than paper
- 4-bit quantization: 16.87 GB → closer but still 1.5x larger
- 3-bit or product quantization: Could reach ~11 GB
- **Likely**: Paper uses quantization not available in public HuggingFace model

**Hypothesis 2: Index-Only Reporting**
- Paper reports only compressed retrieval index, not source embeddings
- Excludes metadata, doc IDs, query encoder

**Hypothesis 3: PLAID-Specific Format**
- Custom binary format optimized for PLAID system
- Not reproducible with public model

### Fair Comparison (Embeddings Only):

- Our float16: **67.5 GB**
- Paper's claimed: **11 GB**
- Ratio: **6.1x larger**

Even with theoretical int8 (33.7 GB), we'd be **3.1x larger** than paper.

### Impact on Reproducibility:

**Storage Efficiency Claim**: Paper's main contribution is "50% storage vs ColBERT (11 GB vs 22 GB)"
- ❌ **Cannot reproduce this claim** with public model
- ✅ Retrieval effectiveness IS reproduced (MRR@10 -0.05%)
- ⚠️ **6.1x storage gap** is a significant reproducibility barrier

### Recommendations:

**For Reproducibility Paper**:
1. Document this storage discrepancy prominently
2. Note that retrieval effectiveness claims ARE validated
3. Storage claims require artifacts not in public release
4. Contact authors for clarification (see [AUTHOR_QUESTIONS.md](AUTHOR_QUESTIONS.md))

**Model Investigation Results** (see [logs/model_investigation.log](logs/model_investigation.log)):
- Model parameters: 109,590,592 (all float32)
- Model config: No quantization parameters found
- Embeddings dtype: float16 (converted during inference, not in model)
- Conclusion: Public model has NO quantization - explains 6.1x gap

**For Authors**:
1. Release quantized model versions on HuggingFace
2. Document exact storage format and calculation method
3. Provide complete reproduction artifacts

**For Community**:
Highlights importance of:
- Releasing complete artifacts (not just fp16 models)
- Documenting optimization techniques
- Clarifying measurement methodology

---

## Summary for SIGIR A* Paper

**Key Reproducibility Challenges**:

1. **Computational Feasibility** ✅ RESOLVED
   - Brute-force MaxSim infeasible → FAISS IVF necessary
   - Successfully addressed with minimal impact (-0.05% MRR@10)

2. **Storage Requirements** ❌ NOT REPRODUCED
   - 6.1x-24.7x larger than claimed
   - **Major reproducibility gap** - cannot verify storage efficiency
   - Missing quantization in public model release

**Reproducibility Assessment**:
- ✅ **Effectiveness claims: Reproduced** (MRR@10: -0.05%)
- ❌ **Storage claims: Not reproduced** (6.1x larger minimum)
- ⚠️ **Computational claims: Cannot compare** (different systems)

**Lessons Learned**:
1. Engineering optimizations are necessary and must be documented
2. Model releases should include ALL optimization artifacts from paper
3. Storage measurements need precise definitions
4. Reproducibility papers provide valuable feedback loop
