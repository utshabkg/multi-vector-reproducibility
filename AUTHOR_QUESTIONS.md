# Questions for ConstBERT Authors

**Context**: We are conducting a SIGIR A* reproducibility study on "Efficient Constant-Space Multi-Vector Retrieval" (ECIR 2025). We have successfully reproduced the effectiveness results (MRR@10: 38.99% vs paper's 39.04%), but encountered a significant storage discrepancy.

---

## Storage Requirements Questions

### Q1: Storage Measurement Methodology

**Our Finding**: 
- Paper claims: 11 GB for ConstBERT32
- Our measurement: 67.5 GB (embeddings only, float16) or 272 GB (total with retrieval infrastructure)
- Ratio: 6.1x larger (embeddings only)

**Theoretical Calculations**:
- Float32: 134.92 GB
- Float16: 67.46 GB (our embeddings)
- Int8: 33.73 GB
- 4-bit: 16.87 GB
- Paper's 11 GB: Requires ~3-bit or product quantization

**Questions**:
1. What precision/quantization was used for the 11 GB measurement?
   - Int8, 4-bit, 3-bit, product quantization, or other?
2. What exactly is included in the 11 GB?
   - Only document embeddings?
   - Document embeddings + query encoder?
   - Complete retrieval index?
   - Doc ID mappings and metadata?
3. Is 11 GB the raw embedding size or compressed/on-disk size?

### Q2: Public Model Availability

**Our Finding**:
- Public `pinecone/ConstBERT` on HuggingFace uses float32 parameters
- Embeddings generated during inference are float16
- No quantization methods included in model release
- Model config shows: `"dtype": "float32"`, no quantization parameters

**Questions**:
1. Are quantized versions of ConstBERT32 available?
   - If yes, where can we access them?
   - If no, can you release them to enable full reproducibility?
2. What quantization method should be applied to achieve 11 GB?
3. Is there a post-processing step we're missing?

### Q3: PLAID Index Format

**Questions**:
1. Does the 11 GB refer to PLAID's internal index format?
2. If so, is PLAID's index format documented or available?
3. Can you clarify if 11 GB is specific to PLAID or applies to the general ConstBERT model?

---

## Retrieval Implementation Questions

### Q4: MaxSim Computation at Scale

**Our Implementation**:
- Used FAISS IVF approximation (nlist=4096, nprobe=128)
- Necessary because brute-force MaxSim on 8.8M documents = ~300+ hours
- Achieved 3.3 hours with FAISS IVF
- Results: MRR@10 (-0.05%), Recall@50 (-0.51%), Recall@1000 (-3.49%)

**Questions**:
1. What retrieval method was used in the paper?
   - Brute-force MaxSim on all 8.8M documents?
   - Approximate nearest neighbor (ANN) methods?
   - PLAID's inverted index?
2. If approximation was used, what were the parameters?
3. What was the actual runtime for 6,980 queries?

---

## Reproducibility Artifacts

### Q5: Code and Model Release

**Current Status**:
- Paper states "Code coming soon!" in repository README
- Public model available but without quantization
- PLAID integration details not documented

**Questions**:
1. When will the complete reproduction code be released?
2. Can you provide:
   - Quantization scripts used for 11 GB index?
   - PLAID index building code?
   - Complete evaluation pipeline?
3. Would you consider releasing:
   - Quantized model checkpoints (int8, 4-bit)?
   - Exact index format used in experiments?

---

## Results Validation

### Q6: Our Reproduction Results

**Effectiveness (Successfully Reproduced)**:
| Metric | Paper | Ours | Diff |
|--------|-------|------|------|
| MRR@10 | 39.04% | 38.99% | -0.05% ✓ |
| Recall@50 | 85.86% | 85.35% | -0.51% ✓ |
| Recall@200 | 93.72% | 92.08% | -1.64% ⚠️ |
| Recall@1000 | 96.34% | 92.85% | -3.49% ⚠️ |

**Questions**:
1. Do our MRR@10 and Recall@50 results look correct?
2. Is the Recall@1000 gap (-3.49%) expected with FAISS IVF approximation?
3. Did the paper use any approximation methods that might affect these metrics?

---

## Summary

**What We Successfully Reproduced**:
✅ Model loading from HuggingFace
✅ MaxSim scoring (mathematically verified)
✅ MRR@10 metric (-0.05% difference)
✅ Recall@50 metric (-0.51% difference)

**What We Cannot Reproduce**:
❌ Storage efficiency (6.1x larger than claimed)
⚠️ Exact retrieval speed (different infrastructure)
⚠️ Recall@1000 (3.49% gap due to approximation)

**Main Blockers**:
1. Missing quantization artifacts in public model
2. Unclear storage measurement methodology
3. No documentation of retrieval implementation details

---

## Contact Information

**Our Team**: [Your team name/affiliation]
**Purpose**: SIGIR A* reproducibility paper
**Timeline**: Planning to submit findings in [timeline]

**We would greatly appreciate**:
1. Clarification on the questions above
2. Access to quantized models if available
3. Any additional details that would help complete reproduction

We want to emphasize that your work is impressive and our effectiveness reproduction was successful. The storage and implementation questions are simply to ensure accurate and complete reproducibility documentation.

Thank you for your time and contribution to the community!

---

**Related Documents**:
- Full reproducibility report: [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md)
- Issues encountered: [ISSUE_RETRIEVAL_STUCK.md](ISSUE_RETRIEVAL_STUCK.md)
- Our implementation: [GitHub link when available]
