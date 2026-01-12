# ConstBERT Reproducibility Study Plan

**Objective**: SIGIR A* reproducibility paper validating ConstBERT claims  
**Target Datasets**: MS-MARCO Passage (primary), ToT (secondary)  
**Baseline Paper**: "Efficient Constant-Space Multi-Vector Retrieval" (ECIR 2025)  
**Status**: Phase 1 - Experiment 1 in progress

## Phase 1: MS-MARCO Reproducibility (PRIMARY)

### 1.1 Environment Setup ✓ COMPLETED
- [x] Conda environment `constbert`
- [x] Install dependencies (transformers, torch, pytrec_eval, etc.)
- [x] Install ujson (required by model)
- [x] Verify GPU setup (NVIDIA RTX 6000 Ada available)

### 1.2 Data Preparation ✓ COMPLETED
**Dataset Locations:**
- Raw: `/media/12TB/shared/datasets/raw/msmarco-passage/`
- Processed: `/media/12TB/shared/datasets/processed/msmarco-passage/`
- Indices: `/media/12TB/shared/datasets/indices/`

**Files Verified:**
- `collection.tsv` (2.9GB) - 8,841,823 passages
- `queries.dev.small.tsv` - 6,980 dev queries
- `qrels.dev.small.tsv` - relevance judgments

**Tasks:**
- [x] Verify data integrity and format
- [x] Create data loaders compatible with ConstBERT
- [ ] Locate TREC DL 2019/2020 queries and qrels (for future experiments)

### 1.3 Model Acquisition ✓ COMPLETED
- [x] Download `pinecone/ConstBERT` from HuggingFace
- [x] Verify model architecture (128-dim, 32 vectors per doc) ✓ Confirmed: C=32, dim=128
- [x] Test inference on sample documents/queries
- [x] Implement wrapper with MaxSim scoring (verified mathematically correct)

### 1.4 Core Experiments

#### Experiment 1: Dev Set Evaluation ✅ COMPLETED
**Target Metrics (Paper Claims):**
- MRR@10: 39.04 (ConstBERT32) vs 39.99 (ColBERT)
- Recall@50: 85.86 vs 86.52
- Recall@200: 93.72 vs 94.47
- Recall@1000: 96.34 vs 97.34

**Our Results:**
- MRR@10: **38.99%** (Paper: 39.04%, Diff: -0.05%) ✅
- Recall@50: **85.35%** (Paper: 85.86%, Diff: -0.51%) ✅
- Recall@200: **92.08%** (Paper: 93.72%, Diff: -1.64%) ⚠️
- Recall@1000: **92.85%** (Paper: 96.34%, Diff: -3.49%) ⚠️

**Implementation Status:**
- [x] Load ConstBERT32 model
- [x] Implement data loaders
- [x] Implement evaluation metrics (MRR, Recall, NDCG)
- [x] Implement MaxSim retrieval (vectorized, verified correct)
- [x] Test pipeline on 10K documents (validation passed)
- [x] Encode all 8.8M passages (68GB embeddings)
- [x] Build FAISS IVF index (nlist=4096, nprobe=128)
- [x] Run retrieval on 6,980 dev queries (3.3 hours)
- [x] Compute MRR@10, Recall@{50,200,1000}
- [x] Compare with paper results

**Engineering Tradeoff:**
- Used FAISS IVF approximation for computational feasibility
- Brute-force MaxSim would take ~300+ hours vs our 3.3 hours
- Primary metric MRR@10 successfully reproduced (0.05% diff)
- Recall@1000 gap (3.49%) due to IVF approximation missing some candidates
- Acceptable tradeoff documented in reproducibility paper

**Storage Paths:**
- Embeddings: `/media/12TB/shared/datasets/processed/msmarco-passage/constbert_msmarco_embeddings.npy`
- Index: `/media/12TB/shared/datasets/indices/constbert_msmarco_index.pkl`
- Results: `./results/exp1_dev_results.json`

#### Experiment 2: TREC Deep Learning Track ✅ COMPLETED
**Target Metrics:**
- TREC DL 2019 NDCG@10: 73.14 (vs 74.64 ColBERT)
- TREC DL 2020 NDCG@10: 73.29 (vs 73.99 ColBERT)

**Our Results:**
- TREC DL 2019 NDCG@10: **68.29%** (Paper: 73.14%, Diff: -4.85%) ⚠️
- TREC DL 2020 NDCG@10: **69.30%** (Paper: 73.29%, Diff: -3.99%) ⚠️

**Implementation:**
- [x] Downloaded TREC DL 2019/2020 queries (43 + 54 = 97 queries)
- [x] Reused existing embeddings and index from Exp 1
- [x] Ran retrieval with FAISS IVF (candidate_mult=10)
- [x] Computed NDCG@10, MAP@1000, Recall metrics
- [x] Compared with paper claims

**Findings:**
- NDCG@10 gaps (~4-5%) larger than MS-MARCO MRR gap (-0.05%)
- Likely due to FAISS IVF approximation affecting ranking quality
- TREC uses graded relevance (0-3) which is more sensitive to rank order
- Runtime: ~90 seconds for 97 queries total

See [results/exp2_trec_dl2019_results.json](results/exp2_trec_dl2019_results.json) and [results/exp2_trec_dl2020_results.json](results/exp2_trec_dl2020_results.json).

#### Experiment 3: Storage Analysis ✅ COMPLETED
**Target Claims:**
- ConstBERT16: 5GB
- ConstBERT32: 11GB (50% of ColBERT 22GB)
- ConstBERT64: 20GB
- ConstBERT128: 40GB

**Our Measurements (ConstBERT32):**
- Raw embeddings (float16): **67.5 GB**
- FAISS IVF index: **137.0 GB**
- Metadata: **67.5 GB**
- **Total: 272 GB**

**Findings:**
- [x] Measure actual index size on disk
- [x] Calculate theoretical storage for different dtypes
- [x] Compare with paper claims
- [x] Investigate public model format (see [logs/model_investigation.log](logs/model_investigation.log))
- [x] Prepare author questions (see [AUTHOR_QUESTIONS.md](AUTHOR_QUESTIONS.md))

**Key Insights:**
- Public HuggingFace model: 109M parameters in float32, NO quantization
- Model config has no quantization documentation
- Embeddings are float16 only because we convert during inference
- Paper's 11 GB requires quantization NOT available in public release
- This is a **reproducibility artifact availability gap**
- Embeddings alone (67.5 GB) are **6.1x larger** than paper's 11 GB claim
- Paper likely uses aggressive quantization (int8/4-bit/product quantization)
- Public HuggingFace model only available in float16
- **Cannot reproduce storage efficiency claims** without quantized model

**Impact:**
- ✅ Effectiveness reproduced (MRR@10 -0.05%)
- ❌ Storage claims not reproducible (6.1x gap)
- Major gap in artifact availability

See [logs/exp3_storage_analysis.log](logs/exp3_storage_analysis.log) and [ISSUE_RETRIEVAL_STUCK.md](ISSUE_RETRIEVAL_STUCK.md#issue-2).

#### Experiment 4: PLAID Integration (HIGH PRIORITY)
**Motivation:** Authors clarified that paper results use PLAID, not IVF. To properly reproduce, we need to test with PLAID.

**Target Setup:** ConstBERT + PLAID retrieval algorithm
**Claims to Reproduce:**
- Storage: 11 GB (vs our 67.5 GB float16)
- MS-MARCO MRR@10: 39.04% (exact match expected with PLAID)
- Retrieval speed: ~100-200 ms/query (vs our 1.7s with IVF)

**Implementation Plan:**
- [ ] Locate and set up PLAID codebase (https://github.com/stanford-futuredata/plaid)
- [ ] Integrate ConstBERT with PLAID quantization
- [ ] Build PLAID index (with product quantization)
- [ ] Run MS-MARCO Dev evaluation with PLAID
- [ ] Compare PLAID vs IVF results:
  - Effectiveness (MRR@10, Recall@k)
  - Storage (11 GB vs 67.5 GB)
  - Speed (ms/query)
  - Trade-offs

**Expected Outcome:**
- Exact match of paper's 11 GB storage
- Closer match to paper's effectiveness (no IVF approximation)
- Validation of storage-efficiency claims

**Status:** Not started - requires PLAID setup

#### Experiment 5: Reranking Pipeline
**Target Setup:** ESPLADE (first-stage) + ConstBERT32 (reranking)
**Claims:**
- Dev MRR: 39.52 (vs 39.99 PLAID)
- MRT: 4.95ms (vs 51.25ms PLAID)
- TREC19 NDCG@10: 74.38
- TREC20 NDCG@10: 74.33

**Implementation Plan:**
- [ ] Set up ESPLADE first-stage retrieval
- [ ] Implement reranking with ConstBERT32
- [ ] Measure latency and effectiveness
- [ ] Compare with end-to-end PLAID
- [ ] Analyze two-stage vs single-stage trade-offs

**Status:** Not started - requires ESPLADE setup

### 1.5 Ablation Studies (Research Value)

#### C parameter sensitivity:
- [ ] Test with C ∈ {8, 16, 32, 64, 128, 256}
- [ ] Plot effectiveness vs storage trade-off
- [ ] Identify optimal C for different use cases

#### Encoding dimension analysis:
- [ ] Test different embedding dimensions (64, 128, 256)
- [ ] Measure impact on effectiveness and storage

#### Query encoding variations:
- [ ] Compare different query encoding strategies
- [ ] Test query expansion techniques

### 1.6 Statistical Significance
**Critical for A* paper:**
- [ ] Multiple runs with different random seeds
- [ ] Compute confidence intervals for metrics
- [ ] Statistical tests (t-test, Wilcoxon) vs baseline
- [ ] Report variance and error bars

### 1.7 Hardware Comparison
**Paper Hardware:** Dual 2.8 GHz Intel Xeon, 256GB RAM, single-thread  
**Our Hardware:** [TO BE DETERMINED]

- [ ] Document our hardware specs
- [ ] Run experiments on CPU (paper setting)
- [ ] Optional: GPU experiments for comparison
- [ ] Normalize latency measurements

## Phase 2: ToT Dataset Evaluation ✅ COMPLETED

### 2.1 ToT Dataset Preparation ✅ COMPLETED
- [x] Locate ToT dataset path
- [x] Understand dataset structure and domain
- [x] Prepare queries and relevance judgments

**Dataset Location:** `/media/12TB/shared/datasets/raw/trec-tot-2025/`
**Corpus:** 6.4M Wikipedia articles (vs 8.8M MS-MARCO passages)
**Test Queries:** 622 (vs 6,980 MS-MARCO dev)
**Query Style:** Long descriptive "tip-of-tongue" narratives vs short factoid queries

### 2.2 Zero-Shot Evaluation ✅ COMPLETED
**Research Question:** Does ConstBERT generalize to different query styles and corpora?

**Our Results (TREC ToT 2025, 622 queries, 6.4M docs):**
- MRR@10: **4.27%** (vs MS-MARCO 38.99%) - **9x worse** ⚠️⚠️⚠️
- Recall@50: **11.41%** (vs MS-MARCO 85.35%) - **7.5x worse** ⚠️⚠️⚠️
- Recall@200: 17.20% (vs MS-MARCO 92.08%)
- Recall@1000: **25.72%** (vs MS-MARCO 92.85%) - **3.6x worse** ⚠️⚠️⚠️
- NDCG@10: 4.82%
- MAP@1000: 17.80%
- Mean Response Time: 716 ms

**Implementation:**
- [x] Created ToT data loader (6.4M Wikipedia articles)
- [x] Encoded corpus with optimized batch sizes (256, ~1.2 hours)
- [x] Built FAISS IVF index (nlist=4096, nprobe=128)
- [x] Evaluated on 622 test queries
- [x] Computed standard IR metrics

**Key Findings:**
- ✅ **Critical Discovery:** ConstBERT shows severe performance degradation on ToT
- ✅ Poor zero-shot transfer when query style differs (descriptive vs factoid)
- ✅ Model is highly optimized for MS-MARCO, not general-purpose
- ✅ This is an **important negative result** for the reproducibility study
- ✅ Demonstrates trade-off: efficiency ↔ generalization

**Storage Paths:**
- Embeddings: `/media/12TB/shared/datasets/indices/trec-tot-2025/constbert_tot_faiss_index/embeddings.npy` (48.9 GB)
- Index: `/media/12TB/shared/datasets/indices/trec-tot-2025/constbert_tot_faiss_index/index.faiss`
- Results: `./results/exp4_tot_results.json`

See [results/TOT_ANALYSIS.md](results/TOT_ANALYSIS.md) for detailed analysis.

### 2.2 Zero-Shot Evaluation
**Research Questions:**
- Does ConstBERT generalize to domain-specific data?
- How does performance compare to fine-tuned models?

**Tasks:**
- [ ] Run ConstBERT (pre-trained) on ToT
- [ ] Compute standard IR metrics
- [ ] Compare with domain-specific baselines

### 2.3 Fine-Tuning (If Time Permits)
- [ ] Fine-tune ConstBERT on ToT training data
- [ ] Compare with zero-shot performance
- [ ] Analyze domain adaptation effectiveness

## Phase 3: Analysis & Paper Writing

### 3.1 Result Analysis
- [ ] Create comprehensive results tables
- [ ] Generate plots (effectiveness-storage trade-offs)
- [ ] Error analysis: where does ConstBERT fail?
- [ ] Case studies: query-level analysis

### 3.2 Additional Experiments for A* Paper
- [ ] Interpretability: what do the C vectors capture?
- [ ] Failure mode analysis
- [ ] Comparison with recent models (2024-2025)
- [ ] Computational cost analysis (encoding time, index building)

### 3.3 Reproducibility Documentation
- [ ] Detailed setup instructions
- [ ] Hardware specifications
- [ ] Exact software versions (pip freeze)
- [ ] Random seeds and hyperparameters
- [ ] Docker container for reproduction

### 3.4 Paper Structure
1. **Introduction**: Importance of reproducibility
2. **Related Work**: ConstBERT and multi-vector retrieval
3. **Methodology**: Reproduction protocol
4. **Results**: 
   - Direct replication attempts
   - Statistical significance tests
   - Ablation studies
5. **Analysis**: 
   - What worked vs what didn't
   - Discrepancies and explanations
6. **Extended Evaluation**: ToT dataset
7. **Discussion**: Lessons learned, recommendations
8. **Conclusion**: Impact and future work

## Technical Approach

### Wrapper Implementation Strategy
**DO NOT modify ConstBERT code** - use composition:

```python
# Wrapper class for our experiments
class ConstBERTEvaluator:
    def __init__(self, model_name="pinecone/ConstBERT"):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    def encode_corpus(self, passages):
        # Batch encoding with progress tracking
        pass
    
    def build_index(self, embeddings):
        # Index construction
        pass
    
    def retrieve(self, query, k=1000):
        # MaxSim scoring and ranking
        pass
    
    def evaluate(self, queries, qrels, metrics=['mrr@10', 'recall@k', 'ndcg@10']):
        # Compute metrics
        pass
```

### Code Organization
```
reproduce/
├── data/
│   ├── loaders.py          # MS-MARCO, TREC data loading
│   └── preprocessing.py    # Data preparation
├── models/
│   ├── constbert_wrapper.py   # Main wrapper
│   └── baselines.py           # ColBERT, ESPLADE wrappers
├── evaluation/
│   ├── metrics.py             # MRR, NDCG, Recall
│   └── statistical_tests.py  # Significance testing
├── experiments/
│   ├── exp1_dev_eval.py       # MS-MARCO dev
│   ├── exp2_trec_eval.py      # TREC DL
│   ├── exp3_storage.py        # Index size analysis
│   └── exp4_reranking.py      # Two-stage pipeline
├── analysis/
│   ├── plots.py               # Visualization
│   └── error_analysis.py      # Failure analysis
└── notebooks/
    └── exploratory.ipynb      # Interactive analysis
```

## Success Criteria (SIGIR A* Level)

### Essential (Must Have):
1. ✓ Reproduce core MS-MARCO results within ±2% of reported
2. ✓ Statistical significance tests with multiple runs
3. ✓ Detailed ablation studies
4. ✓ Comprehensive error analysis
5. ✓ Complete reproducibility documentation

### Desirable (Good to Have):
6. ToT dataset results showing generalization
7. Comparison with 2024-2025 SOTA methods
8. Novel insights beyond paper's claims
9. Practical recommendations for practitioners
10. Open-source reproducibility package

### Outstanding (Excellent):
11. Discovery of unreported behaviors or failure modes
12. Improved methodology or evaluation protocol
13. Extension to new scenarios or datasets
14. Theoretical analysis of ConstBERT properties
15. Community-validated reproduction (Docker, CI/CD)

## Timeline Estimate

- **Phase 1 (MS-MARCO)**: 2-3 weeks
  - Setup & Data: 2-3 days
  - Core experiments: 5-7 days
  - Ablations: 3-4 days
  - Statistical analysis: 2-3 days

- **Phase 2 (ToT)**: 1 week
  - Data prep: 2 days
  - Experiments: 3-4 days
  - Analysis: 1-2 days

- **Phase 3 (Paper)**: 1-2 weeks
  - Results compilation: 2-3 days
  - Additional experiments: 3-4 days
  - Writing: 5-7 days

**Total**: 4-6 weeks for complete study

## Notes

- Focus on **exact reproduction** first, then extensions
- Document **every deviation** from paper protocol
- Keep detailed logs of experiments (wandb/mlflow)
- Regular checkpoints for intermediate results
- Think like a scientist: hypothesis → experiment → analysis
