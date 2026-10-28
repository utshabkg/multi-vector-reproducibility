# ColBERT-v1

This directory holds ColBERT result files that the paper does not report (`results/`) and a ColBERT implementation in PyTorch (`src/`) that two ColBERT-v2 scripts import.

## Results

The file names mark these as PyTerrier runs. The scripts for those runs are not in this repository.

- `results/pyterrier_colbertV1_msmarco-passage_eval.json`: MS MARCO evaluation
- `results/pyterrier_colbertV1_tot_dev1_eval.json`, `results/pyterrier_colbertV1_tot_dev2_eval.json`, `results/pyterrier_colbertV1_tot_dev3_eval.json`: TREC ToT DEV1, DEV2 and DEV3 evaluations
- `results/pyterrier_colbertV1_tot_dev123_eval.json`: TREC ToT evaluation over DEV1, DEV2 and DEV3 combined
- `results/pyterrier_colbert1.9_msmarco_eval.json`: ColBERT v1.9 MS MARCO evaluation

## Source Code

`src/` holds a separate ColBERT implementation in PyTorch: the model (`colbert_model.py`), training (`colbert_training.py`), FAISS indexing (`colbert_indexing.py`) and retrieval (`colbert_retrieval.py`). It does not use PyTerrier.

Two ColBERT-v2 scripts import from `src/`:
- `colbertv2/scripts/train_colbert_tot_triples.py` imports `ColBERT` and `ColBERTTrainer`
- `constbert/experiments/rq5/run_ablation_colbert.py` imports `ColBERT`
