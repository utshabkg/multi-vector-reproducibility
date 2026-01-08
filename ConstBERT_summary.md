# ConstBERT Paper Summary (ECIR 2025)

**Paper**: Efficient Constant-Space Multi-Vector Retrieval  
**Authors**: Sean MacAvaney, Antonio Mallia, Nicola Tonellotto  
**Award**: Best Short Paper Honourable Mention at ECIR 2025

## Core Innovation

ConstBERT addresses the storage overhead of multi-vector retrieval models (like ColBERT) by encoding documents into a **fixed number (C) of learned embeddings** instead of one vector per token.

### Key Method: Learned Pooling

- **ColBERT**: Uses M vectors (one per token) for each document
- **ConstBERT**: Uses C vectors (fixed, C < M) via learned linear projection:
  - `[δ₁ | ... | δC] = Wᵀ[d₁ | ... | dM]`
  - W ∈ ℝ^(Mk×Ck) is learned end-to-end
  
### Scoring Function

MaxSim late interaction:
```
s(q, d) = Σᵢ₌₁ᴺ max_{j=1,...,C} qᵢᵀδⱼ
```

## Claimed Results on MS-MARCO Passage (8.8M docs)

### Main Findings (Table 1):

| Model | Index Size | MRR@10 | Recall@50 | Recall@200 | Recall@1000 | TREC19 NDCG@10 | TREC20 NDCG@10 |
|-------|------------|--------|-----------|------------|-------------|----------------|----------------|
| **ColBERT** | 22GB | 39.99 | 86.52 | 94.47 | 97.34 | 74.64 | 73.99 |
| ColBERTSP | 14GB | 39.12 | 85.81 | 93.80 | 97.00 | 74.42 | 72.36 |
| **ConstBERT16** | 5GB | 37.84 | 84.04 | 91.74 | 94.11 | 71.15 | 73.75 |
| **ConstBERT32** | 11GB | 39.04 | 85.86 | 93.72 | 96.34 | 73.14 | 73.29 |
| **ConstBERT64** | 20GB | 39.15 | 86.27 | 94.06 | 96.90 | 74.29 | 73.47 |
| **ConstBERT128** | 40GB | 39.53 | 86.46 | 94.39 | 97.29 | 74.37 | 73.31 |

### Critical Claims to Verify:

1. **ConstBERT32 achieves ~98% of ColBERT's MRR@10** (39.04 vs 39.99) with **50% storage** (11GB vs 22GB)
2. **ConstBERT32 maintains competitive TREC NDCG@10** scores (73.14 vs 74.64 on TREC19)
3. Storage reduction is linear with C: ~5GB for C=16, ~11GB for C=32

### Reranking Performance (Table 3):

| Method | Dev MRR | MRT(ms) | TREC19 NDCG@10 | MRT(ms) | TREC20 NDCG@10 | MRT(ms) |
|--------|---------|---------|----------------|---------|----------------|---------|
| ColBERT (PLAID) | 39.99 | 51.25 | 74.26 | 51.46 | 73.99 | 50.21 |
| ESPLADE | 38.75 | 3.07 | 71.33 | 3.13 | 71.14 | 3.20 |
| ESPLADE + ConstBERT32 | 39.52 | 4.95 | 74.38 | 5.50 | 74.33 | 5.23 |

**Claim**: Two-stage with ConstBERT32 achieves similar effectiveness as PLAID with **~10x faster MRT** (4.95ms vs 51.25ms)

## BEIR Benchmark Results (Table 2)

- ConstBERT32/64 evaluated on 13 BEIR datasets
- Generally maintains competitive NDCG@10 scores vs ColBERT
- Storage reduction is consistent across datasets

## Experimental Setup

- **Platform**: Dual 2.8 GHz Intel Xeon CPUs, 256 GB RAM, single-threaded
- **Training**: Follows ColBERTv2 approach (Santhanam et al.)
- **Code**: Uses PLAID codebase for end-to-end retrieval
- **Metrics**: 
  - MRR@10 for MS-MARCO Dev
  - NDCG@10 for TREC/BEIR
  - MRT (Mean Response Time) in milliseconds
  - Recall@k for various cutoffs

## Implementation Details

- **Model Available**: HuggingFace `pinecone/ConstBERT`
- **Architecture**: BERT-based with learned pooling layer
- **Embedding size**: 128 dimensions
- **Default C**: 32 vectors per document
- **Training**: End-to-end with ColBERTv2 training procedure

## Key Advantages Claimed

1. **Storage efficiency**: 50-77% reduction vs ColBERT
2. **OS paging**: Fixed-size embeddings improve memory alignment
3. **Flexible C parameter**: Trade-off between storage and effectiveness
4. **Orthogonal to other compressions**: Can combine with dimensionality reduction
5. **Reranking friendly**: Fixed vectors simplify implementation

## Research Questions for Reproducibility

### Primary (MS-MARCO):
1. Can we reproduce the MRR@10, Recall@k metrics on MS-MARCO Dev?
2. Can we reproduce TREC DL 2019/2020 NDCG@10 results?
3. Does ConstBERT32 truly achieve ~98% effectiveness with 50% storage?
4. What is the actual retrieval latency (MRT) on our hardware?
5. How does training stability/variance affect results?

### Secondary (For ToT dataset later):
6. Does ConstBERT generalize to domain-specific datasets?
7. How does zero-shot performance compare?

## Potential Concerns

1. **No variance/error bars**: Single run reported
2. **Hardware-specific**: Results on specific CPU config
3. **Training reproducibility**: Need exact hyperparameters
4. **Index building time**: Not reported
5. **Memory usage during retrieval**: Only index size reported
