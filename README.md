# Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions

**Utshab Kumar Ghosh, Ashish David and Shubham Chatterjee**, Missouri University of Science and Technology

### Published at SIGIR 2026 (Melbourne, VIC, Australia). Paper: [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3805712.3808561)

This repository holds the code and result files for our SIGIR 2026 reproducibility paper. The paper evaluates two multi-vector retrieval models, ConstBERT and ColBERT-v2, across five dimensions:

| RQ | Dimension | Evaluated on |
|----|-----------|--------------|
| RQ1 | Implementation correctness | MS MARCO passage ranking, TREC DL 2019 and 2020 |
| RQ2 | Backend robustness | FAISS-IVF vs. PLAID |
| RQ3 | Domain generalization | 13 BEIR datasets |
| RQ4 | Structural generalization | Long narrative queries from TREC ToT 2025 |
| RQ5 | Adaptation potential | Fine-tuning on TREC ToT 2025 |

## Key Findings

The paper's central finding is that models can reproduce numerically while failing architecturally.

- **RQ1, implementation correctness.** ConstBERT reproduces MS MARCO MRR@10 within 0.05% (38.99% vs. 39.04%) and Recall@50 within 0.51% (85.35% vs. 85.86%). ColBERT-v2 reproduces MRR@10 within 0.55% (39.15% vs. 39.7%).
- **RQ2, backend robustness.** On MS MARCO, ConstBERT reaches 38.99% MRR@10 with FAISS-IVF. With PLAID on the same data, undocumented parameters create an 8-point reproduction gap (31.09% vs. 39.04%), which the paper traces to ConstBERT's sparse centroid coverage.
- **RQ3, domain generalization.** Reproducibility on BEIR is asymmetric. ColBERT-v2 reproduces with high fidelity, with a mean nDCG@10 of 48.6% over the 13 datasets. ConstBERT shows larger gaps. With PLAID, its mean nDCG@10 is 37.4%, against the 46.8% reported for ConstBERT.
- **RQ4, structural generalization.** On TREC ToT 2025, both models drop by 86% to 97% relative to MS MARCO. ConstBERT reaches 4.27% MRR@10 with FAISS-IVF and 0.94% with PLAID-16, and ColBERT-v2 reaches 5.66%. Performance plateaus at 20 words, which the paper attributes to the MaxSim operator's uniform token weighting. With the index removed, exact MaxSim scoring of ConstBERT on 100 sampled ToT queries reaches only 5.08% MRR@10.
- **RQ5, adaptation potential.** Fine-tuning on the ToT training split with BM25 hard negatives lowers performance. On the TEST set, ColBERT-v2 drops by 29.2% and ConstBERT by 7.0% relative to their zero-shot baselines.

The paper concludes that architectural constraints in multi-vector retrieval cannot be overcome by adaptation alone.

## Models and Datasets

| Model | Checkpoint (Hugging Face) | Dimension | Vectors per document |
|-------|---------------------------|:---------:|----------------------|
| ConstBERT | `pinecone/ConstBERT` | 128 | 32, fixed |
| ColBERT-v2 | `colbert-ir/colbertv2.0` | 128 | One per token, variable |

| Dataset | As described in the paper | Used in |
|---------|----------------------------|---------|
| MS MARCO Passage | 8.8M passages, 6,980 dev queries | RQ1, RQ2 |
| TREC Deep Learning 2019 and 2020 | Graded relevance judgments | RQ1 |
| BEIR v1.0.0 | 13 datasets, corpus sizes from 3.6K to 5.4M | RQ3 |
| TREC ToT 2025 | 6.4M Wikipedia articles, 622 TEST queries | RQ2, RQ4, RQ5 |

Other ToT splits in the paper: TRAIN (143 queries) is the fine-tuning set, and DEV1 (142), DEV2 (143) and DEV3 (536) appear in Table 9.

## Repository Layout

```
├── README.md
├── requirements.txt              # pip packages for the ConstBERT scripts
├── ISSUES_AND_INSIGHTS.md        # working notes: issues met during the project
├── PLAN.md                       # working notes: the research plan
├── REPRODUCIBILITY_REPORT.md     # working notes: an interim ConstBERT report
├── scripts/                      # data download helpers
│   ├── download_msmarco.sh       # MS MARCO collection, queries and dev qrels, TREC DL qrels
│   ├── download_beir.sh          # the 13 BEIR datasets
│   └── download_tot.sh           # creates the TREC ToT 2025 folders and prints a partial file list
├── constbert/                    # ConstBERT code, plus some ColBERT-v2 scripts and results
│   ├── README.md
│   ├── models/                   # constbert_wrapper.py, faiss_index.py
│   ├── data/                     # loaders.py (MS MARCO, TREC DL), tot_loader.py (TREC ToT)
│   ├── evaluation/               # metrics.py (MRR, Recall, nDCG, precision, MAP)
│   ├── experiments/
│   │   ├── rq1/                  # implementation correctness
│   │   ├── rq2/                  # backend robustness
│   │   ├── rq3/                  # domain generalization (BEIR)
│   │   ├── rq4/                  # structural generalization (TREC ToT)
│   │   └── rq5/                  # adaptation potential (fine-tuning on ToT)
│   ├── scripts/                  # bash wrappers, each changes into constbert/ first
│   └── results/                  # result JSON files
│       └── multiseed/            # ToT TEST results of the three fine-tuning runs per model
├── colbertv2/                    # ColBERT-v2 scripts and results
│   ├── README.md
│   ├── environment.yml           # conda packages for the colbertv2 environment
│   ├── scripts/                  # indexing, search, evaluation, triple generation, training
│   └── results/                  # MS MARCO and ToT evaluation JSON files
└── colbertv1/                    # supplementary ColBERT-v1 material
    ├── README.md
    ├── src/                      # ColBERT model, training, indexing and retrieval code (two ColBERT-v2 scripts import it)
    └── results/                  # ColBERT-v1 result JSON files, not reported in the paper
```

Each sub-folder README describes its scripts in more detail.

## Scripts and Result Files for Paper Results

Short script paths such as `rq2/...` are relative to `constbert/experiments/`. Result file paths that do not start with `constbert/` or `colbertv2/`, including `multiseed/...`, are under `constbert/results/`. The tracked file names can differ from the names a script writes by default.

| RQ | Paper result | Script(s) | Result file(s) |
|----|--------------|-----------|----------------|
| RQ1 | Table 1, ConstBERT | `constbert/experiments/rq1/01_msmarco_faiss_eval.py` | `constbert/results/01_msmarco_faiss.json` |
| RQ1 | Table 1, ColBERT-v2 | `colbertv2/scripts/run_eval_msmarco.sh`, then `colbertv2/scripts/eval_msmarco_json.py` | `colbertv2/results/msmarco_eval_metrics.json` |
| RQ1 | TREC DL 2019 and 2020 (Sec. 5.1.4) | `constbert/experiments/rq1/02_trec_dl_faiss_eval.py` | `constbert/results/02_trec_dl2019_faiss.json`, `constbert/results/02_trec_dl2020_faiss.json` |
| RQ2 | Table 2, MS MARCO, PLAID-4 | `rq2/build_plaid_index_msmarco.py`, an IVF step (see [RQ2](#rq2-backend-robustness)), `rq2/05_msmarco_plaid_eval.py` | `05_msmarco_plaid.json` |
| RQ2 | Table 2, MS MARCO, PLAID-16 | `rq2/06_msmarco_plaid_ncells.py` (needs the IVF step, see [RQ2](#rq2-backend-robustness)) | `06_msmarco_plaid_ncells16.json` |
| RQ2 | Table 2, ToT, PLAID-16 | `rq2/build_plaid_index_tot.py`, `rq4/09_tot_plaid_eval.py` (ColBERT cannot load the index this builder writes, see [RQ4](#rq4-structural-generalization-trec-tot)) | `09_tot_plaid_ncells16.json` |
| RQ2 | PLAID parameter search (Sec. 5.2.2) | `rq2/18_plaid_parameter_sweep.py` (needs the IVF step, see [RQ2](#rq2-backend-robustness)) | `18_plaid_parameter_sweep.json` |
| RQ2 | Centroid coverage (Sec. 5.2.3) | `rq2/17_centroid_coverage_analysis.py --num_docs 5000` | `17_centroid_coverage.json` |
| RQ3 | Table 3, ConstBERT with PLAID | `rq3/12-1_build_beir_plaid_indices.py`, `rq3/12-2_build_ivf_for_beir.py`, `rq3/12-3_beir_plaid_eval.py` (see the note below) | `12_beir_plaid_<dataset>.json`, `12_beir_plaid_summary.json` |
| RQ3 | Table 4, ConstBERT with FAISS-IVF | `rq3/11_beir_evaluation.py` | `11_beir_constbert_<dataset>.json`, `11_beir_constbert_summary.json` |
| RQ3 | Means of Tables 3 and 4 | `rq3/compare_beir_backends.py`, run from inside `constbert/` | `12_beir_backend_comparison.json` |
| RQ3 | Table 5, ColBERT-v2 | `rq3/13_beir_colbertv2_evaluation.py` | `13_beir_colbertv2_<dataset>.json`, `13_beir_colbertv2_summary.json` |
| RQ4 | Tables 2 and 6, ConstBERT with FAISS-IVF on ToT | `rq4/04_tot_faiss_eval.py` (does not run to completion as committed, see [RQ4](#rq4-structural-generalization-trec-tot)) | `04_tot_faiss.json` |
| RQ4 | Table 6, ColBERT-v2 on ToT | `rq5/eval_colbert_base_test.py` (needs a ColBERT-v2 ToT index, see [RQ4](#rq4-structural-generalization-trec-tot)) | `colbert_base_test.json` |
| RQ4 | Table 7, query length ablation | `rq4/15_query_length_ablation.py` (needs a saved ToT FAISS index, see [RQ4](#rq4-structural-generalization-trec-tot)) | `15_query_length_ablation.json` |
| RQ4 | Table 8, exact MaxSim | `rq4/16_exact_maxsim_tot.py` (does not run to completion as committed, see [RQ4](#rq4-structural-generalization-trec-tot)) | `16_exact_maxsim_tot.json` |
| RQ5 | Table 9, TEST rows of the fine-tuned models | `constbert/scripts/run_multiseed_finetuning.sh`, `rq5/eval_multiseed_tot.py`, `rq5/eval_multiseed_colbert.py` | `multiseed/constbert_seed*_eval.json`, `multiseed/colbert_seed*_eval.json` |
| RQ5 | Fine-tuning with tripled training data (Sec. 5.5.1) | `rq5/run_ablation_constbert.py`, `rq5/run_ablation_colbert.py` | `14_ablation_constbert_finetune_summary.json`, `14_ablation_colbert_finetune_summary.json` |

**Note on Table 3.** `12-3_beir_plaid_eval.py` picks its PLAID search settings from `k`. At `k = 1000` it uses PLAID's documented defaults (`ncells=4`). The paper's Table 3 uses `ncells = 16` (Sec. 5.3.1). Each per-dataset BEIR PLAID result file records the settings of its run. The script has no command-line flags for these settings, so to use the settings that a result file records, edit the `k`-based settings block in `12-3_beir_plaid_eval.py` before you re-run.

## Setup

### Prerequisites

- Linux with an NVIDIA GPU and CUDA. `requirements.txt` installs `faiss-gpu`, and its comment names `faiss-cpu` as the option without a GPU.
- Python 3.10 for the ConstBERT scripts. `requirements.txt` asks for `faiss-gpu>=1.7.4`, and PyPI only has those wheels for Python 3.10 and newer.
- conda, for the separate Python 3.8 environment that the ColBERT-v2 scripts in `colbertv2/scripts/` use.
- Plenty of free disk space. The MS MARCO embeddings and indices are large.

### Installation

```bash
git clone https://github.com/utshabkg/multi-vector-reproducibility.git
cd multi-vector-reproducibility

# ConstBERT scripts
conda create -n constbert python=3.10
conda activate constbert
pip install -r requirements.txt
# ColBERT engine for the PLAID scripts, at the commit named in requirements.txt. The colbert-ai
# release that the line above installs from PyPI has the same version number (0.2.22), and pip
# keeps an installed package of the same version unless --force-reinstall is given
pip install --force-reinstall --no-deps git+https://github.com/stanford-futuredata/ColBERT.git@cc4f3dc91c0b45d2d08c251d9d95178285c65f1c
pip install rank-bm25   # needed by 10_generate_triples_simple.py, not listed in requirements.txt
```

Some wrappers in `constbert/scripts/` activate conda environments by name, so keep the names `constbert` and `colbertv2`.

**ColBERT-v2 environment.** `colbertv2/environment.yml` lists the conda packages for the `colbertv2` environment. Its last entry asks pip to install from a `requirements.txt` inside `colbertv2/`. That file is not in the repository, so `conda env create -f colbertv2/environment.yml` fails at the pip step. The commands below build the environment from a copy without that entry, then install the ColBERT engine at the commit named in the top-level `requirements.txt`. Pick `pytorch` and `faiss-gpu` builds that match your CUDA version, as the notes at the end of `environment.yml` say.

```bash
sed '/^  - pip:/,/requirements\.txt/d' colbertv2/environment.yml > /tmp/colbertv2-env.yml
conda env create -f /tmp/colbertv2-env.yml
conda activate colbertv2
pip install git+https://github.com/stanford-futuredata/ColBERT.git@cc4f3dc91c0b45d2d08c251d9d95178285c65f1c
```

See `colbertv2/README.md` for what individual ColBERT-v2 scripts need.

### Data

Run these from the repository root. By default they write under `data/`. Each one also accepts another target folder as its first argument.

```bash
bash scripts/download_msmarco.sh   # RQ1, RQ2: data/msmarco-passage/
bash scripts/download_beir.sh      # RQ3: data/beir/
bash scripts/download_tot.sh       # RQ2, RQ4, RQ5: creates data/trec-tot-2025/ and prints a partial file list (see below)
```

The bash wrappers in `constbert/scripts/` change into `constbert/` first, so they read datasets from `constbert/data/` instead. The one exception is the ColBERT-v2 training step in `run_multiseed_finetuning.sh`, which reads from `colbertv2/data/`. If you use the wrappers, pass the matching folder to the download script, for example `bash scripts/download_beir.sh constbert/data/beir`.

**MS MARCO.** `download_msmarco.sh` fetches the collection, the query files in `queries.tar.gz`, the dev qrels and the TREC DL qrels. The ConstBERT scripts also read `queries.dev.small.tsv` (the 6,980 dev queries) and the TREC DL 2019 and 2020 query files, which the script does not fetch:

```bash
cd data/msmarco-passage
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xzf collectionandqueries.tar.gz queries.dev.small.tsv && rm collectionandqueries.tar.gz
wget -qO- https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz | gunzip > msmarco-test2019-queries.tsv
wget -qO- https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz | gunzip > msmarco-test2020-queries.tsv
cd ../..
```

**TREC ToT 2025.** `download_tot.sh` does not download anything. The track's [guidelines page](https://trec-tot.github.io/guidelines) links to the corpus, queries and qrels. Decompress the corpus (`gunzip`) and place the files where `constbert/data/tot_loader.py` looks for them:

```
data/trec-tot-2025/trec-tot-2025-corpus.jsonl
data/trec-tot-2025/queries/{train,dev1,dev2,dev3,test}-2025-queries.jsonl
data/trec-tot-2025/qrel/{train,dev1,dev2,dev3,test}-2025-qrel.txt
```

The DEV file names that `download_tot.sh` prints (such as `dev1-queries.jsonl`) differ from these, and it prints no TRAIN files. Some ColBERT-v2 scripts and `constbert/experiments/rq5/run_multiseed_finetuning.py` default to `data/trec-tot-2025/raw/` instead, so pass the paths or edit the defaults.

## Running the Experiments

How paths work:

- Run the Python commands from the repository root. Most scripts resolve `data/`, `indices/` and `checkpoints/` against the current directory.
- The bash wrappers in `constbert/scripts/` change into `constbert/` first, so their relative paths resolve inside `constbert/`.
- Depending on the script, results go to `results/` under the current directory, to `constbert/results/` or to `colbertv2/results/`.
- Several scripts load embeddings, indices or checkpoints saved by an earlier step and stop if these are missing. Default paths do not always line up between scripts, so check a script's paths (`--help`, or the constants near the top of the file) before a long run.

### RQ1: Implementation Correctness

```bash
# ConstBERT on MS MARCO dev with FAISS-IVF. Encodes the 8.8M passages, then saves the
# embeddings and the index (--load-index reuses them on later runs)
python constbert/experiments/rq1/01_msmarco_faiss_eval.py

# ConstBERT on TREC DL 2019 and 2020, using the index saved by the previous command
python constbert/experiments/rq1/02_trec_dl_faiss_eval.py

# ColBERT-v2 on MS MARCO dev (colbertv2 environment). Searches an existing ColBERT-v2 index,
# indices/msmarco-passage-colbertv2 by default, then scores the ranking. The indexing wrappers
# in colbertv2/scripts/ call index_colbertv2.py, which is not in this repository
bash colbertv2/scripts/run_eval_msmarco.sh
python colbertv2/scripts/eval_msmarco_json.py data/msmarco-passage/qrels.dev.small.tsv colbertv2/results/msmarco_full.ranking.tsv
```

### RQ2: Backend Robustness

```bash
# Build the ConstBERT PLAID index for MS MARCO from saved ConstBERT embeddings. It reads
# indices/msmarco-passage/msmarco-passage-constbert/constbert_msmarco_embeddings.npy and
# constbert_msmarco_faiss_index.pkl in the same folder, so copy or link the files that
# 01_msmarco_faiss_eval.py saved (data/msmarco-passage/ and indices/) to that folder first
python constbert/experiments/rq2/build_plaid_index_msmarco.py

# Add the IVF files and metadata.json that ColBERT needs to load the index (see the note below)
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("build_ivf", "constbert/experiments/rq3/12-2_build_ivf_for_beir.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.build_ivf("indices/msmarco-passage/constbert_plaid_index")
PY

# PLAID with its documented defaults for k = 1000 (Table 2, PLAID-4)
python constbert/experiments/rq2/05_msmarco_plaid_eval.py

# PLAID with ncells = 16 (Table 2, PLAID-16)
python constbert/experiments/rq2/06_msmarco_plaid_ncells.py

# PLAID parameter search
python constbert/experiments/rq2/18_plaid_parameter_sweep.py

# Centroid coverage. Reads saved ConstBERT embeddings from a path set inside the script
# and stops if that file is missing, so check the path before you run it
python constbert/experiments/rq2/17_centroid_coverage_analysis.py --num_docs 5000
```

**Note on RQ2.** `build_plaid_index_msmarco.py` does not write the IVF files (`ivf.pt` and `ivf.pid.pt`) or the `metadata.json` that ColBERT needs to load a PLAID index. Without them, `05_msmarco_plaid_eval.py`, `06_msmarco_plaid_ncells.py` and `18_plaid_parameter_sweep.py` stop with an `AssertionError` when they open the index. The `build_ivf()` function in `constbert/experiments/rq3/12-2_build_ivf_for_beir.py` writes these files. That script's command line accepts only BEIR dataset names, so the step above calls the function directly.

### RQ3: Domain Generalization (BEIR)

```bash
# ConstBERT with FAISS-IVF on one dataset, using the BEIR settings stated in the paper
# (nlist=1024, nprobe=128). constbert/scripts/run_beir_evaluation.sh loops over all 13.
python constbert/experiments/rq3/11_beir_evaluation.py --model constbert --dataset nfcorpus --nlist 1024 --nprobe 128 --gpu

# ConstBERT with PLAID: build the indices, add the IVF files, then evaluate (see the note on Table 3)
python constbert/experiments/rq3/12-1_build_beir_plaid_indices.py constbert --all
python constbert/experiments/rq3/12-2_build_ivf_for_beir.py --all
python constbert/experiments/rq3/12-3_beir_plaid_eval.py --all

# ColBERT-v2 on all 13 datasets
python constbert/experiments/rq3/13_beir_colbertv2_evaluation.py --all
```

### RQ4: Structural Generalization (TREC ToT)

```bash
# ConstBERT with FAISS-IVF on ToT TEST (Tables 2 and 6). Encodes the corpus and saves
# embeddings.npy and metadata.npy in indices/trec-tot-2025/constbert_tot_faiss_index/
# (unless both exist), then stops before it scores anything (see the note below)
python constbert/experiments/rq4/04_tot_faiss_eval.py

# ConstBERT with PLAID on ToT. The builder reads the embeddings saved above. ColBERT cannot load
# the index it writes, so the evaluation stops when it opens the index (see the note below)
python constbert/experiments/rq2/build_plaid_index_tot.py
python constbert/experiments/rq4/09_tot_plaid_eval.py

# Query length ablation (Table 7). The script loads a saved ToT FAISS index and does not build
# one. The inline Python step builds and saves that index from the embeddings saved above
python - <<'PY'
import sys
sys.path.insert(0, "constbert")
import numpy as np, torch
from models.faiss_index import ConstBERTFAISSIndex
d = "indices/trec-tot-2025/constbert_tot_faiss_index/"
index = ConstBERTFAISSIndex(use_gpu=torch.cuda.is_available())
index.add_documents(np.load(d + "metadata.npy", allow_pickle=True), np.load(d + "embeddings.npy"))
index.save(d + "index")
PY
python constbert/experiments/rq4/15_query_length_ablation.py

# Exact MaxSim (Table 8). Stops before it scores anything (see the note below)
python constbert/experiments/rq4/16_exact_maxsim_tot.py

# ColBERT-v2 zero-shot on ToT TEST. Searches an existing ColBERT-v2 ToT index at
# indices/trec-tot-2025/trec-tot-2025-colbertv2, a path set in the script (see the note below)
python constbert/experiments/rq5/eval_colbert_base_test.py
```

**Note on RQ4.**

- `04_tot_faiss_eval.py` and `16_exact_maxsim_tot.py` do not run to completion as committed. The [Known Issues](constbert/README.md#known-issues) section of `constbert/README.md` explains why. Their tracked outputs are `constbert/results/04_tot_faiss.json` and `constbert/results/16_exact_maxsim_tot.json`.
- `build_plaid_index_tot.py` does not write `avg_residual.pt`, `buckets.pt` or `metadata.json`, which ColBERT needs to load a PLAID index. So `09_tot_plaid_eval.py` stops with a `FileNotFoundError` when it opens an index built this way. Its tracked output is `constbert/results/09_tot_plaid_ncells16.json`.
- `15_query_length_ablation.py` loads a saved ToT FAISS index (the files `index` and `index.faiss`) from `indices/trec-tot-2025/constbert_tot_faiss_index/`. `04_tot_faiss_eval.py` stops before it saves an index, and no other script in `constbert/experiments/rq4/` saves one. The inline Python step builds the index from the `embeddings.npy` and `metadata.npy` files that `04_tot_faiss_eval.py` saves in that folder, then writes `index` and `index.faiss`.
- `eval_colbert_base_test.py` needs a ColBERT-v2 index of the ToT corpus and the ID map that `colbertv2/scripts/index_colbertv2_tot.sh` writes to `colbertv2/tmp/`. That wrapper then calls `index_colbertv2.py`, which is not in this repository, so no script here builds the index.

### RQ5: Adaptation Potential (Fine-Tuning on ToT)

```bash
# BM25 hard-negative triples from the ToT training split, written where the ConstBERT runs of
# the multi-seed wrapper below read them
python constbert/experiments/rq5/10_generate_triples_simple.py \
  --corpus_file constbert/data/trec-tot-2025/trec-tot-2025-corpus.jsonl \
  --queries_file constbert/data/trec-tot-2025/queries/train-2025-queries.jsonl \
  --qrels_file constbert/data/trec-tot-2025/qrel/train-2025-qrel.txt \
  --output_file constbert/indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples

# Three fine-tuning runs per model (see the notes below)
bash constbert/scripts/run_multiseed_finetuning.sh
```

The TEST rows of Table 9 come from three fine-tuning runs per model.

- `constbert/scripts/run_multiseed_finetuning.sh` launches the runs. It changes into `constbert/` first, so its ConstBERT runs read the triples written above and the ToT corpus and TRAIN queries under `constbert/data/trec-tot-2025/`. Its ColBERT-v2 runs change into `colbertv2/` and read `indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples` and the ToT files under `data/trec-tot-2025/` there, so check those paths first. The ColBERT-v2 training code is `colbertv2/scripts/train_colbert_tot_triples.py`, and `colbertv2/README.md` covers the ColBERT-v2 triple scripts.
- `eval_multiseed_tot.py --model constbert --seed <seed>` and `eval_multiseed_colbert.py --seed <seed>` (both in `constbert/experiments/rq5/`) evaluate each run on TEST, and `eval_multiseed_tot.py --aggregate` summarizes both models.
- `eval_multiseed_tot.py` skips a run whose result file already exists in `constbert/results/multiseed/`. `eval_multiseed_colbert.py` checks for its result file only after it has converted the checkpoint and found or built the index, and then returns the stored result. The tracked files are in that folder, so move them aside before you re-run.
- `eval_multiseed_colbert.py` also needs a ColBERT-v2 index of each fine-tuned model. When one is missing, it calls `colbertv2/scripts/index_colbertv2_tot.sh` to build it, and that wrapper calls `index_colbertv2.py`, which is not in this repository.

## Result Files

The JSON results of our runs are included:

- `constbert/results/` (72 files, 6 of them in `multiseed/`): ConstBERT results, plus the ColBERT-v2 BEIR, ToT TEST, multi-seed and fine-tuning ablation results.
- `colbertv2/results/` (6 files): ColBERT-v2 MS MARCO and ToT runs.
- `colbertv1/results/` (6 files): supplementary ColBERT-v1 results, not reported in the paper.

Some files come from runs that the paper does not report. The table in [Scripts and Result Files for Paper Results](#scripts-and-result-files-for-paper-results) names the result file for each paper result it lists.

## Citation

```bibtex
@inproceedings{ghosh2026reproduction,
  author    = {Ghosh, Utshab Kumar and David, Ashish and Chatterjee, Shubham},
  title     = {Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series    = {SIGIR '26},
  year      = {2026},
  pages     = {2921--2930},
  publisher = {ACM},
  address   = {New York, NY, USA},
  location  = {Melbourne, VIC, Australia},
  isbn      = {979-8-4007-2599-9},
  doi       = {10.1145/3805712.3808561}
}
```

## License

This code is released for academic research purposes in connection with the accompanying paper. The paper itself is published under a Creative Commons Attribution 4.0 International License.
