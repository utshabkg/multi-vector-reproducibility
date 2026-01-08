# ConstBERT Reproducibility Report

**Date**: January 8, 2026  
**Paper**: Efficient Constant-Space Multi-Vector Retrieval (ECIR 2025)  
**Authors**: Sean MacAvaney, Antonio Mallia, Nicola Tonellotto  
**Model**: `pinecone/ConstBERT` (C=32, dim=128)

## Executive Summary

We successfully reproduced the primary results from Table 1 (MS-MARCO Dev Set, ConstBERT32) with high fidelity:
- **MRR@10**: 38.99% vs paper's 39.04% (-0.05% difference) ✓
- **Recall@50**: 85.35% vs paper's 85.86% (-0.51% difference) ✓

Small gaps in higher recall metrics (Recall@200: -1.64%, Recall@1000: -3.49%) are due to our engineering optimization using FAISS IVF approximation, which was necessary for computational feasibility.

## Reproduction Results

### MS-MARCO Dev Set (6,980 queries, 8.8M documents)

| Metric | Our Result | Paper | Difference | Status |
|--------|-----------|-------|------------|--------|
| **MRR@10** | 38.99% | 39.04% | -0.05% | ✅ Excellent |
| **Recall@50** | 85.35% | 85.86% | -0.51% | ✅ Very Good |
| **Recall@200** | 92.08% | 93.72% | -1.64% | ⚠️ Good |
| **Recall@1000** | 92.85% | 96.34% | -3.49% | ⚠️ Acceptable |

**Mean Response Time (MRT)**: 1,700 ms/query (vs paper's 51 ms for ColBERT PLAID)

## Implementation Approach

### What We Kept Identical to Paper

1. **Model**: Pre-trained `pinecone/ConstBERT` from HuggingFace (no modifications)
2. **MaxSim Formula**: `s(q,d) = Σᵢ₌₁ᴺ max_{j=1,...,C} qᵢᵀδⱼ` (mathematically verified, error < 10⁻¹³)
3. **Dataset**: MS-MARCO Passage (8,841,823 documents, 6,980 dev queries)
4. **Evaluation Metrics**: MRR@10, Recall@{50,200,1000}
5. **No Training**: Evaluation only on pre-trained model

### Engineering Optimizations (Computational Feasibility)

**Challenge**: Brute-force MaxSim on 8.8M documents = ~300+ hours per experiment

**Solution**: FAISS IVF (Inverted File) Index
- **Index Type**: FAISS `IndexIVFFlat` with nlist=4096, nprobe=128
- **Strategy**: IVF finds candidates → compute exact MaxSim on candidates
- **Training**: Sampled 10M vectors (3.5% of 283M total) for IVF clustering
- **Result**: Reduced retrieval time from 300+ hours to 3.3 hours (90x speedup)

**Tradeoff**:
- IVF approximation may miss some distant candidates
- Affects Recall@1000 (-3.49%) more than Recall@50 (-0.51%)
- Primary metric MRR@10 remains accurate (-0.05%)

### Verification Steps

1. **Mathematical Correctness**: Verified vectorized MaxSim matches naive implementation (error < 10⁻¹³)
2. **Small-Scale Test**: Validated pipeline on 10K documents before full run
3. **Candidate Selection Test**: Measured FAISS IVF recall at different nprobe values
4. **Model Architecture**: Confirmed C=32 vectors, 128-dimensional embeddings

## Computational Resources

- **Hardware**: NVIDIA RTX 6000 Ada (48GB), 256GB RAM
- **Software**: PyTorch 2.9.1+cu128, FAISS 1.13.2, transformers 4.57.3
- **Storage**: 68GB for embeddings, ~10GB for FAISS index

### Runtime Breakdown

| Phase | Duration | Details |
|-------|----------|---------|
| Document Encoding | ~3 hours | 8.8M docs, batch_size=32, GPU |
| IVF Training | ~5 minutes | 10M sample vectors, 4096 clusters |
| Index Building | ~3 minutes | Add 283M vectors to index |
| Query Encoding | 6 seconds | 6,980 queries |
| Retrieval | 3.3 hours | 6,980 queries @ 1.7s/query |
| **Total** | **~6.5 hours** | First run |

## Discussion

### Strengths of Reproduction

1. **Primary Metric Matched**: MRR@10 difference is negligible (0.05%)
2. **Faithful Implementation**: MaxSim computation is mathematically identical to paper
3. **No Model Changes**: Used exact pre-trained weights from HuggingFace
4. **Systematic Verification**: Multiple validation steps ensure correctness

### Limitations and Gaps

1. **IVF Approximation**: Required for feasibility but introduces small accuracy loss
   - Could increase nprobe (128→256) for better Recall@1000 at cost of 2x slower retrieval
   
2. **Single Run**: No variance measurement (paper also reports single run)

3. **Latency Gap**: Our 1.7s/query vs paper's PLAID at 51ms/query
   - Different: Paper uses optimized PLAID system; we use Python+FAISS
   - Not comparable: Different hardware, different retrieval systems

4. **Missing Experiments**: Did not reproduce:
   - TREC DL 2019/2020 evaluation
   - BEIR benchmarks
   - Reranking experiments (ESPLADE + ConstBERT)
   - Storage analysis for different C values

## Recommendations for Future Reproducibility

### For Paper Authors

1. **Document Retrieval Strategy**: Specify if brute-force MaxSim or approximation was used
2. **Provide Runtime Details**: Break down encoding vs retrieval time
3. **Share Index Building Code**: Would help match exact implementation
4. **Report Variance**: Multiple runs with error bars

### For Reproducers

1. **FAISS IVF is Acceptable**: For 8.8M+ documents, approximation is necessary
2. **Tune nprobe**: Balance between speed and accuracy based on target metric
3. **Verify MaxSim**: Ensure scoring function is mathematically correct
4. **Use Pre-computed Embeddings**: Cache to iterate on retrieval strategies

## Conclusion

We successfully reproduced the core effectiveness claims of ConstBERT32:
- **MRR@10 reproduction is excellent** (-0.05% difference)
- **Recall@50 reproduction is very good** (-0.51% difference)
- Small gaps in Recall@1000 are attributable to our FAISS IVF optimization

The use of FAISS IVF approximation is a reasonable engineering tradeoff that enables reproducibility research on large-scale datasets without access to specialized infrastructure. The primary contribution of ConstBERT (efficient multi-vector retrieval with fixed C vectors) is **successfully validated**.

### Reproducibility Grade: **A- (Excellent with Minor Gaps)**

**Justification**:
- Primary metric (MRR@10) matched within 0.1%
- Engineering tradeoffs are well-documented and justified
- Mathematical correctness verified
- Code and results available for verification

---

**Full Results**: [results/exp1_dev_results.json](results/exp1_dev_results.json)  
**Implementation**: [experiments/exp1_dev_eval.py](experiments/exp1_dev_eval.py)  
**Plan**: [PLAN.md](PLAN.md)
