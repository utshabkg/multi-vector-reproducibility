# ColBERT-v2 Experiments

This directory holds scripts for ColBERT-v2 (`colbert-ir/colbertv2.0`), one of the two models the paper reproduces. ColBERT-v2 represents each document as a variable-length sequence of vectors, one 128-dimensional vector per token.

Some ColBERT-v2 code and results live in `constbert/`. See [ColBERT-v2 Code in constbert/](#colbert-v2-code-in-constbert) below.

## Environment Setup

Most ColBERT-v2 scripts run in a separate conda environment named `colbertv2`. `environment.yml` lists its conda packages, but its pip entry installs from a `requirements.txt` that is not in this folder, so `conda env create -f environment.yml` fails at the pip step. The [Installation](../README.md#installation) steps in the top-level README build the environment without that entry and then install the ColBERT engine. `scripts/generate_triples_bm25.py` also needs PyTerrier or Pyserini, which those steps do not install.

The ColBERT-v2 BEIR wrapper `constbert/scripts/run_colbertv2_beir_evaluation.sh` runs in the `constbert` conda environment instead.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/index_colbertv2_msmarco.sh` | Index MS MARCO with ColBERT-v2. Calls `scripts/index_colbertv2.py`. |
| `scripts/index_colbertv2_tot.sh` | Index the TREC ToT corpus with ColBERT-v2. Converts a JSONL corpus to TSV, then calls `scripts/index_colbertv2.py`. |
| `scripts/index_colbertv2.py` | Build a ColBERT-v2 index, then move it to `<indices_root>/<index_name>`. The two wrappers above call it and pass `--output-root` and `--indices-root`. When you run it directly, pass both, since their defaults point to folders on our machine. |
| `scripts/run_eval_msmarco.sh` | Search MS MARCO dev queries with `run_search.py`, then score the ranking with the ColBERT codebase's MS MARCO evaluator (`python -m utility.evaluate.msmarco_passages`) |
| `scripts/eval_msmarco_json.py` | Score an MS MARCO ranking against qrels and write the metrics as JSON. Usage: `eval_msmarco_json.py <qrels> <ranking> [out.json]`. The default output path, `colbertv2/results/msmarco_eval_metrics.json`, is relative to the working directory, so run it from the repository root or pass the output path. |
| `scripts/run_eval_tot.sh` | Search a TREC ToT split with `run_search.py` and compute TREC metrics. Defaults to DEV1 and the base checkpoint. It takes optional positional arguments in the order of its usage line: index, queries, qrels, output ranking, k, checkpoint and collection. |
| `scripts/run_search.py` | Run ColBERT-v2 search over an index and write a TSV ranking |
| `scripts/compute_trec_metrics.py` | Compute TREC-style metrics from qrels and a run file. `run_eval_tot.sh` falls back to it when neither `trec_eval` nor `pytrec_eval` is available. |
| `scripts/convert_jsonl_to_tsv.py` | Convert a JSONL corpus or query file to TSV |
| `scripts/convert_ranking_ids.py` | Map internal passage ids in a ranking back to the original ids |
| `scripts/generate_triples_bm25.py` | Generate BM25 hard-negative training triples from an existing PyTerrier or Pyserini BM25 index (default `indices/trec-tot-2025-bm25`). No script in this repository builds that index. With default arguments it writes `data/processed/trec-tot-2025-triple-bm25/train.triples`. |
| `scripts/run_generate_triples_bm25.sh` | Run `generate_triples_bm25.py` with its default arguments. It prints `indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples` as the output path, but the triples land at the generator's default path above. `run_train_colbertv2_tot.sh` reads the `indices/` path, so move the file there or run the generator with `--out`. |
| `scripts/tot_data_loaders_triples.py` | ToT triples dataset classes used by the training script |
| `scripts/train_colbert_tot_triples.py` | Fine-tune ColBERT-v2 on ToT triples. Imports `ColBERT` and `ColBERTTrainer` from `colbertv1/src/`. |
| `scripts/run_train_colbertv2_tot.sh` | Run `train_colbert_tot_triples.py` with the full training settings, or a short test run with `--test` |
| `scripts/env_setup.sh` | Set CPU thread counts for OpenMP, MKL, OpenBLAS and NumExpr |

With a JSONL query file (the default), `run_eval_tot.sh` converts the queries to TSV with new ids 0, 1, 2 and so on, so its run file uses those ids. Only the `compute_trec_metrics.py` fallback maps them back to the ToT query ids. `trec_eval` and `pytrec_eval` score the run as written, against qrels that use the ToT ids. If either one is available, pass a TSV query file that keeps the ToT ids, which `convert_jsonl_to_tsv.py --use-original-pid` writes.

`run_eval_msmarco.sh`, `run_eval_tot.sh`, `constbert/experiments/rq5/eval_colbert_base_test.py` and `constbert/experiments/rq5/eval_multiseed_colbert.py` need a ColBERT-v2 index, which the two indexing wrappers build. `eval_multiseed_colbert.py` also needs each fine-tuned checkpoint in `colbertv2/models/`, which is not tracked, unless a converted copy already exists. When a model's index is missing, it calls `index_colbertv2_tot.sh` itself to build it. It does both before it looks for its tracked result file in `constbert/results/multiseed/`.

ColBERT compiles its CUDA extensions the first time it needs them. The `colbertv2` environment's own `nvcc` (CUDA 11.7) cannot build them for newer GPUs such as the RTX 6000 Ada, so set `CUDA_HOME` to a newer CUDA toolkit (CUDA 12.6 works). The build also needs `ninja`, which is in the environment's `bin/` folder, so activate the environment or put that folder on `PATH`.

## ToT Data Paths

The ToT scripts in `scripts/` default to a `raw/` subfolder and expect to be run from the repository root:

- `data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl`
- `data/trec-tot-2025/raw/queries/<split>-2025-queries.jsonl`
- `data/trec-tot-2025/raw/qrel/<split>-2025-qrel.txt`

`train_colbert_tot_triples.py` differs in one default. Its `--queries_file` default is `data/trec-tot-2025/tot_bm25_top1000.tsv`, which no script here creates, so pass `--queries_file` when you run it directly. `run_train_colbertv2_tot.sh` passes `data/trec-tot-2025/raw/queries/train-2025-queries.jsonl`.

The root `scripts/download_tot.sh` and the ConstBERT ToT loader (`constbert/data/tot_loader.py`) use `data/trec-tot-2025/` with no `raw/`. To use another location, pass the paths as arguments. Two wrappers do not take paths: `run_train_colbertv2_tot.sh` sets them inside the script, and `run_generate_triples_bm25.sh` calls the generator without arguments.

## Results

Evaluation results in `results/`:
- `msmarco_eval_metrics.json`: MS MARCO dev evaluation
- `tot_dev1_eval_metrics.json`: ToT DEV1 evaluation that the paper does not report
- `tot_dev1_finetuned_eval.json`, `tot_dev2_finetuned_eval.json`, `tot_dev3_finetuned_eval.json`: ToT DEV1, DEV2 and DEV3 evaluations from the fine-tuning experiments
- `tot_test_finetuned_eval.json`: ToT TEST evaluation from a single run that the paper does not report. The paper's ColBERT-v2 TEST results are in `constbert/results/` (see below).

## ColBERT-v2 Code in constbert/

These ColBERT-v2 scripts live under `constbert/experiments/`. Their tracked results are in `constbert/results/`:
- `rq3/13_beir_colbertv2_evaluation.py`: BEIR evaluation. Results are `13_beir_colbertv2_*.json`. The wrapper `constbert/scripts/run_colbertv2_beir_evaluation.sh` has its loops over the smaller datasets commented out.
- `rq5/eval_colbert_base_test.py`: Zero-shot evaluation on ToT TEST. Result is `colbert_base_test.json`.
- `rq5/eval_multiseed_colbert.py`: ToT TEST evaluation of the three fine-tuned models. Results are `multiseed/colbert_seed*_eval.json`.
- `rq5/run_ablation_colbert.py`: Fine-tuning ablation. It writes `colbert_ablation_summary.json`, and the tracked summary is `14_ablation_colbert_finetune_summary.json`.
