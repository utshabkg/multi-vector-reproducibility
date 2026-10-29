# ConstBERT Experiments

This directory holds the experiment code for ConstBERT (`pinecone/ConstBERT`), a multi-vector retrieval model that pools each document into exactly 32 vectors of 128 dimensions. It also holds four ColBERT-v2 experiment scripts (marked below), two ColBERT-v2 wrappers in `scripts/` (`run_colbertv2_beir_evaluation.sh` and `run_colbert_multiseed_eval.sh`) and the result files for both models.

## Core Library

| File | Description |
|------|-------------|
| `models/constbert_wrapper.py` | ConstBERT model wrapper for encoding queries and documents |
| `models/faiss_index.py` | FAISS-IVF index for approximate nearest neighbor search |
| `data/loaders.py` | MS MARCO and TREC DL data loaders |
| `data/tot_loader.py` | TREC ToT 2025 data loader |
| `evaluation/metrics.py` | MRR@k, Recall@k, nDCG@k, precision and MAP computation |

## Experiments by Research Question

The research question names follow the paper.

### RQ1: Implementation Correctness (`experiments/rq1/`)

Asks whether the reported in-domain effectiveness can be reproduced on MS MARCO. This folder covers ConstBERT on MS MARCO and on TREC DL 2019 and 2020. The ColBERT-v2 MS MARCO run is in `colbertv2/`.

- `01_msmarco_faiss_eval.py`: MS MARCO passage ranking with the FAISS-IVF backend
- `02_trec_dl_faiss_eval.py`: TREC DL 2019 and 2020, reusing the index built by `01_msmarco_faiss_eval.py`
- `03_storage_analysis.py`: Embedding storage size analysis. Its input paths default to absolute paths on the authors' machine, so pass your own with `--embeddings`, `--faiss-index`, `--faiss-pkl` and `--plaid-index`.

### RQ2: Backend Robustness (`experiments/rq2/`)

Asks how retrieval backends affect reproducibility when parameters are undocumented. Most scripts here run ConstBERT with PLAID, for comparison with the FAISS-IVF runs.

- `build_plaid_index_msmarco.py`: Build a PLAID index for MS MARCO from precomputed ConstBERT embeddings. It reads `constbert_msmarco_embeddings.npy` and `constbert_msmarco_faiss_index.pkl` from `indices/msmarco-passage/msmarco-passage-constbert/`. With its defaults, `01_msmarco_faiss_eval.py` writes these files to `data/msmarco-passage/` and `indices/`, so copy or link them into that folder first.
- `build_plaid_index_tot.py`: Build a PLAID index for the ToT corpus from precomputed ConstBERT embeddings. Its default inputs are the `embeddings.npy` and `metadata.npy` files that `04_tot_faiss_eval.py` writes to `indices/trec-tot-2025/constbert_tot_faiss_index/`.
- `05_msmarco_plaid_eval.py`: MS MARCO evaluation with PLAID
- `06_msmarco_plaid_ncells.py`: MS MARCO evaluation with PLAID at ncells=16
- `07_exact_maxsim_verify.py`: Exact brute-force MaxSim on a subset of MS MARCO dev queries (not reported in the paper). It reads the same embeddings file as `build_plaid_index_msmarco.py`.
- `08_trec_dl_plaid_eval.py`: TREC DL 2019 and 2020 evaluation with PLAID (not reported in the paper)
- `17_centroid_coverage_analysis.py`: Centroid coverage analysis
- `18_plaid_parameter_sweep.py`: PLAID parameter sweep over ncells and the centroid score threshold

### RQ3: Domain Generalization (`experiments/rq3/`)

Out-of-domain evaluation on 13 BEIR datasets, for ConstBERT with FAISS-IVF and with PLAID, and for ColBERT-v2.

- `11_beir_evaluation.py`: ConstBERT FAISS-IVF evaluation on BEIR. `--model` and `--dataset` are required, even with `--all`, which runs all 13 datasets and writes `11_beir_constbert_summary.json`. `scripts/run_beir_evaluation.sh` runs the 13 datasets one at a time.
- `12-1_build_beir_plaid_indices.py`: Encode the BEIR corpora with ConstBERT and build PLAID indices
- `12-2_build_ivf_for_beir.py`: Add the IVF files that PLAID search needs to the BEIR PLAID indices
- `12-3_beir_plaid_eval.py`: ConstBERT PLAID evaluation on BEIR
- `13_beir_colbertv2_evaluation.py`: ColBERT-v2 BEIR evaluation (ColBERT-v2 script)
- `15_verify_faiss_hypothesis.py`: FAISS-IVF nlist test on FEVER (not reported in the paper)
- `compare_beir_backends.py`: Compare the PLAID and FAISS-IVF results on BEIR. It reads and writes `results/` relative to the working directory, so run it from `constbert/`.

### RQ4: Structural Generalization (`experiments/rq4/`)

Zero-shot evaluation of ConstBERT on the long narrative queries of TREC ToT 2025. The ColBERT-v2 ToT TEST run is `experiments/rq5/eval_colbert_base_test.py`.

- `04_tot_faiss_eval.py`: ToT evaluation with FAISS-IVF. It saves the corpus embeddings (`embeddings.npy`, `metadata.npy`) and the FAISS index (`index`, `index.faiss`) in `indices/trec-tot-2025/constbert_tot_faiss_index/`, where the other ToT scripts read them, and reuses them when they exist.
- `09_tot_plaid_eval.py`: ToT evaluation with PLAID
- `15_query_length_ablation.py`: Query length ablation that truncates ToT queries to fixed word counts. It loads the FAISS index that `04_tot_faiss_eval.py` saves and does not build one.
- `16_exact_maxsim_tot.py`: Exact brute-force MaxSim on 100 sampled ToT queries. It scores the stored embeddings of every document that `04_tot_faiss_eval.py` saves, on CPU, so it needs about 55 GB of RAM and runs for several hours.

### RQ5: Adaptation Potential (`experiments/rq5/`)

Fine-tunes ConstBERT and ColBERT-v2 on TREC ToT and evaluates the fine-tuned models. This folder also holds the zero-shot ColBERT-v2 evaluation on ToT TEST.

- `10_generate_triples_simple.py`: Generate BM25 hard-negative training triples. `--corpus_file`, `--queries_file`, `--qrels_file` and `--output_file` are required. It needs the `rank-bm25` package (`pip install rank-bm25`), which the root `requirements.txt` does not list. `scripts/run_constbert_finetuning.sh` does not call this script, and it stops unless the triples already exist at `indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples` under `constbert/`.
- `10_finetune_constbert_tot.py`: Fine-tune ConstBERT on ToT. `--triples_file`, `--corpus_file` and `--queries_file` are required. `scripts/run_constbert_finetuning.sh` passes them.
- `10_eval_finetuned_tot_test.py`: Evaluate the fine-tuned ConstBERT checkpoint on ToT (`--splits`, default `test`). It reads the checkpoint from `checkpoints/constbert_tot_finetuned/` under the working directory.
- `run_multiseed_finetuning.py`: Launch three fine-tuning runs of each model. It starts every job inside `constbert/` with input paths set in the script: `indices/processed/trec-tot-2025-triple-bm25/train.triples` and files under `data/trec-tot-2025/raw/`. Its ColBERT-v2 jobs also read `data/trec-tot-2025/tot_bm25_top1000.tsv`, which no script here creates. A failed job does not stop the script. `scripts/run_multiseed_finetuning.sh` launches the runs with the wrapper layout described below.
- `eval_multiseed_tot.py`: Evaluate the three fine-tuned ConstBERT models on ToT TEST. `--aggregate` summarizes both models.
- `eval_multiseed_colbert.py`: Evaluate the three fine-tuned ColBERT-v2 models on ToT TEST (ColBERT-v2 script)
- `run_ablation_constbert.py`: ConstBERT fine-tuning ablation
- `run_ablation_colbert.py`: ColBERT-v2 fine-tuning ablation (ColBERT-v2 script, runs in the `colbertv2` conda environment)
- `eval_colbert_base_test.py`: Zero-shot ColBERT-v2 (`colbert-ir/colbertv2.0`) evaluation on ToT TEST (ColBERT-v2 script)

## Wrapper Scripts

`scripts/` holds bash wrappers that call the experiment scripts above. Each wrapper changes into `constbert/` first, so relative paths such as `data/`, `indices/`, `checkpoints/`, `results/` and `logs/` resolve inside `constbert/`. The one exception is the ColBERT-v2 training step in `run_multiseed_finetuning.sh`, which runs from `colbertv2/`.

The multi-seed ToT TEST evaluation runs through `run_multiseed_eval.sh` (ConstBERT) and `run_colbert_multiseed_eval.sh` (ColBERT-v2). Two wrappers need inputs that are not in the repository (see [Wrapper Notes](#wrapper-notes)).

## Data

The wrappers read datasets from `constbert/data/` (for example `constbert/data/beir/` or `constbert/data/trec-tot-2025/`), next to the Python loaders in that folder. The ColBERT-v2 training step of `scripts/run_multiseed_finetuning.sh` is the exception. It reads the ToT corpus and queries from `colbertv2/data/trec-tot-2025/` and the triples from `colbertv2/indices/trec-tot-2025/trec-tot-2025-triple-bm25/`. Most experiment scripts resolve `data/` against the working directory, so when you run them from the repository root they read the root `data/` folder. These scripts use paths under `constbert/` wherever you launch them:

- `experiments/rq5/run_multiseed_finetuning.py` starts every job inside `constbert/`.
- `experiments/rq3/15_verify_faiss_hypothesis.py` reads BEIR from `constbert/data/beir/`.
- `experiments/rq5/run_ablation_constbert.py` and `run_ablation_colbert.py` keep their training triples in `constbert/data/tot/finetuning/`. They read the ToT files from `data/trec-tot-2025/` under the working directory.

The download scripts in the root `scripts/` folder write to `data/` under the current directory by default, so pass the target folder as their first argument, for example `bash scripts/download_beir.sh constbert/data/beir` from the repository root. Two of them leave gaps:

- `scripts/download_msmarco.sh` does not provide `queries.dev.small.tsv`, which `data/loaders.py` reads for the dev queries and `experiments/rq2/18_plaid_parameter_sweep.py` reads directly. It also skips `msmarco-test2019-queries.tsv` and `msmarco-test2020-queries.tsv`, which `02_trec_dl_faiss_eval.py` and `08_trec_dl_plaid_eval.py` read. Add these files to the MS MARCO folder yourself.
- `scripts/download_tot.sh` downloads nothing. It prints a list of expected files, but its DEV file names differ from the ones `data/tot_loader.py` reads, and it lists no TRAIN files. The loader and the RQ5 wrappers read these files from the ToT folder, for the splits `train`, `dev1`, `dev2`, `dev3` and `test`:
  - `trec-tot-2025-corpus.jsonl`
  - `queries/<split>-2025-queries.jsonl`
  - `qrel/<split>-2025-qrel.txt`

See [Setup](../README.md#setup) in the top-level README for how to get these files.

## Results

`results/` holds the JSON outputs for ConstBERT and for the ColBERT-v2 scripts marked above. `results/multiseed/` holds the ToT TEST evaluation of each of the three fine-tuning runs per model. Some files hold results that the paper does not report, such as `08_trec_dl_plaid_*.json`, `10_finetune-results_tot_test.json`, `11_beir_constbert_fever_nlist4096.json`, `12_beir_plaid_summary_nbit-4_threshold-0.4.json` and `14_beir_backend_comparison.json`.

## Wrapper Notes

Two wrappers in `scripts/` need inputs that are not in the repository, such as training triples or fine-tuned checkpoints:

- `scripts/run_constbert_finetuning.sh` fine-tunes ConstBERT, saves the model to `constbert/checkpoints/constbert_tot_finetuned/` and then runs `experiments/rq5/10_eval_finetuned_tot_test.py --splits dev1 dev2 dev3 test`. The evaluator writes `results/finetune-results_tot_<split>.json`, so it leaves the tracked `10_finetune-results_tot_<split>.json` files alone. It loads the document embeddings (`embeddings.npy`, `metadata.npy`) and the FAISS index (`index`, `index.faiss`) from `indices/trec-tot-2025/constbert_tot_faiss_index/` and encodes only the queries with the fine-tuned model. If the embeddings are missing, it first encodes the whole corpus with the fine-tuned model and saves them there. That is the folder where `04_tot_faiss_eval.py` keeps the base-model embeddings for the other ToT scripts, so run `04_tot_faiss_eval.py` first. If the index is missing, it builds one from the embeddings and saves it there. If `checkpoints/constbert_tot_finetuned/model.safetensors` is missing, it encodes the queries with the base model. With `--test` or `--dry-run`, the wrapper still trains, but only for 50 steps on a small sample. It saves that model to `checkpoints/constbert_tot_finetuned_test/` and skips the evaluation, because the evaluator always reads `checkpoints/constbert_tot_finetuned/`.
- `scripts/eval_multiseed_models.sh` runs `run_multiseed_eval.sh` and then `run_colbert_multiseed_eval.sh`. `eval_multiseed_tot.py` skips a run whose result file already exists in `results/multiseed/`, and the tracked result files are there. `eval_multiseed_colbert.py` looks for its result file only after two earlier steps. It converts the fine-tuned checkpoint in `colbertv2/models/`, which is not tracked, unless a converted copy already exists in `indices/trec-tot-2025/`. If neither exists, `run_colbert_multiseed_eval.sh` stops at its first run, even though the tracked result files are there. If the model's ColBERT-v2 index is missing from that folder, it builds one through `colbertv2/scripts/index_colbertv2_tot.sh` (see `colbertv2/README.md`).
