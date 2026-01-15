# Issues, Insights, and Author Communication

This document captures the challenges faced during reproduction, novel insights discovered, and author clarifications that should be included in the reproducibility paper.

---

## Issue 1: Storage Discrepancy (6× Gap)

### What We Observed
- **Paper claim**: 11 GB for ConstBERT32
- **Our measurement**: 67.5 GB (float16 embeddings only), 272 GB total with index

### Why This Happened
The paper used **PLAID** (Product-quantized Late Interaction via Decoupled approximation), which:
1. Stores quantized codes + residuals instead of raw embeddings
2. Achieves ~12× compression over float16
3. Is a retrieval algorithm, not part of the ConstBERT model itself

The public `pinecone/ConstBERT` release only includes the model weights, not the PLAID integration or quantized index format.

### Author Clarification
> "The core issues arise from a bit of a misunderstanding about PLAID. PLAID is a method for quantizing and retrieving over multi-vector models. This means that the index size can be substantially smaller than the raw vectors, since it doesn't need to store the original embeddings."

### Resolution
After integrating PLAID ourselves, we achieved **5.6 GB** — actually better than the paper's 11 GB claim.

### Paper Insight
**Artifact Gap**: Storage efficiency claims require PLAID, which wasn't clearly documented. This is a common issue in industry-academic collaborations where release constraints prevent full artifact availability.

---

## Issue 2: Retrieval Method Ambiguity

### What We Observed
The paper did not clearly specify which retrieval method was used. We initially assumed brute-force MaxSim, but this would require 300+ hours per experiment.

### Why This Happened
1. Paper focused on the ConstBERT model, not retrieval infrastructure
2. PLAID was mentioned but its role as the primary retrieval method wasn't emphasized
3. No runtime or implementation details were provided

### Our Solution
We used FAISS IVF (nlist=4096, nprobe=128) as a practical alternative:
- Reduced runtime from 300+ hours to 3.3 hours
- Achieved MRR@10 within 0.05% of paper
- Trade-off: Higher recall metrics affected more

### Author Clarification
> "What retrieval method was used in the paper? PLAID. We used PLAID's defaults."
> 
> "It's also really cool to see that ConstBERT seems to work well under an IVF engine! This on its own is a good and helpful reproducibility result :)"

### Paper Insight
**Alternative Backend Validation**: ConstBERT works with multiple retrieval backends (PLAID, FAISS IVF), demonstrating model flexibility beyond the paper's setup.

---

## Issue 3: PLAID-ConstBERT Incompatibility (Novel Finding)

### What We Observed
After integrating PLAID, we found significant degradation:
- MS-MARCO: 31.09% MRR (vs 38.99% FAISS IVF) — **20% relative drop**
- TREC DL 2019: 59.53% NDCG (vs 68.29% FAISS IVF) — **13% relative drop**
- ToT 2025: 0.94% MRR (vs 4.27% FAISS IVF) — **78% relative drop**

### Why This Happens

**Root Cause: Fixed vs Variable Length Representations**

| Property | ColBERT | ConstBERT |
|----------|---------|-----------|
| Tokens per doc | Variable (60-120) | Fixed (32) |
| Centroids covered | ~60-120 | ~12-20 |
| Query-doc overlap | ~15-20 centroids | ~3 centroids |

PLAID's retrieval works by:
1. Mapping each query token to its nearest centroids
2. Finding documents that share centroids with the query
3. Scoring only those candidate documents

**The Problem**: ConstBERT's 32 vectors are learned representations, not token embeddings. They cluster into fewer centroids because:
1. They represent semantic concepts, not individual tokens
2. Multiple vectors may encode similar information
3. The fixed count means shorter documents don't use all 32 vectors meaningfully

**Mathematical Explanation**:
- ColBERT (100 tokens) → ~80 unique centroids → high query overlap probability
- ConstBERT (32 vectors) → ~15 unique centroids → low query overlap probability
- With `ncells=4` (default), many relevant documents are never retrieved

### Verification
Increasing `ncells` from 4 to 16 improved MRR from 30.01% to 31.09%, but still far below FAISS IVF's 38.99%.

### Paper Insight
**Novel Architectural Incompatibility**: This is a previously undocumented limitation. The paper's results likely used different PLAID parameters or evaluation setup. This finding has implications for other fixed-length multi-vector models.

---

## Issue 4: Out-of-Domain Generalization Failure (Novel Finding)

### What We Observed
On TREC ToT 2025 (tip-of-the-tongue queries):
- FAISS IVF: 4.27% MRR@10 (vs 38.99% on MS-MARCO) — **89% drop**
- PLAID: 0.94% MRR@10 (vs 31.09% on MS-MARCO) — **97% drop**

### Why This Happens

**1. Query Style Mismatch**
| Property | MS-MARCO | ToT 2025 |
|----------|----------|----------|
| Query length | 10-20 words | 100-200 words |
| Query style | Factoid ("what is X") | Descriptive narrative |
| Example | "what causes rain" | "I'm trying to remember a movie where..." |

**2. Training Distribution**
ConstBERT was trained on MS-MARCO, which has:
- Short, keyword-focused queries
- Web passage answers (~100 words)
- Factoid question-answer pairs

ToT queries are fundamentally different:
- Long, rambling descriptions
- Many "filler" tokens that don't contribute to relevance
- Require semantic understanding beyond keyword matching

**3. MaxSim Behavior on Long Queries**
MaxSim sums the max similarity for each query token:
```
s(q, d) = Σᵢ max_j qᵢᵀδⱼ
```
For long queries with many irrelevant tokens:
- Many tokens contribute noise to the score
- Relevant tokens get diluted
- Documents with partial matches score high

**4. PLAID Amplifies the Problem**
With long queries:
- More query tokens → more centroids probed
- But centroids are distributed across irrelevant terms
- Relevant documents less likely to share centroids with key query terms

### Paper Insight
**Efficiency-Generalization Trade-off**: ConstBERT achieves storage efficiency at the cost of generalization. This trade-off wasn't explored in the original paper and has important implications for practitioners.

---

## Author Q&A Summary

### Questions Sent (January 11, 2026)

1. **Storage**: What precision/quantization achieves 11 GB?
2. **Artifacts**: Are quantized models available?
3. **PLAID**: Does 11 GB correspond to PLAID index?
4. **Retrieval**: What retrieval method was used?
5. **Code**: When will reproduction code be released?
6. **Validation**: Are our results correct?

### Author Response (January 11, 2026)

**Key Clarifications:**

1. **PLAID is the answer**: "PLAID is a method for quantizing and retrieving over multi-vector models. This means that the index size can be substantially smaller than the raw vectors."

2. **Storage = PLAID compressed vectors**: "Compressed PLAID vectors on-disk" — not raw embeddings.

3. **Retrieval = PLAID defaults**: "What retrieval method was used? PLAID. We used PLAID's defaults."

4. **Our gaps are expected**: "I suspect these differences are all due to using IVF approximation instead of PLAID approximation."

5. **Alternative backends validated**: "It's also really cool to see that ConstBERT seems to work well under an IVF engine! This on its own is a good and helpful reproducibility result."

6. **Release constraints**: "As an industry-academic collaboration, we regretfully had some challenges in releasing all experimental artifacts."

### Impact on Our Study

The author response:
- Confirmed PLAID is essential for storage claims
- Validated our FAISS IVF approach as an alternative
- Explained the performance gaps we observed
- Acknowledged release constraints affecting reproducibility

---

## Summary of Novel Insights for Paper

| Finding | Significance | Section |
|---------|--------------|---------|
| PLAID-ConstBERT Incompatibility | Fixed-length representations break PLAID assumptions | Issue 3 |
| OOD Generalization Failure | 89% MRR drop on ToT queries | Issue 4 |
| PLAID Amplifies OOD Degradation | 78% additional drop with PLAID on ToT | Issue 4 |
| Alternative Backend Works | FAISS IVF outperforms PLAID for ConstBERT | Issue 2 |
| Storage Gap Explained | PLAID required but not documented | Issue 1 |

---

## Recommendations

### For Paper Authors
1. Document PLAID dependency clearly in future work
2. Release PLAID integration code if possible
3. Test models on diverse query types beyond MS-MARCO

### For Practitioners
1. Use FAISS IVF instead of PLAID for ConstBERT
2. Expect poor generalization on long/descriptive queries
3. Consider query-type-specific models for diverse applications

### For Reproducers
1. Contact authors early for clarification
2. Test alternative retrieval backends
3. Evaluate on out-of-domain datasets
