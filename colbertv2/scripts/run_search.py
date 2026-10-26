#!/usr/bin/env python3
"""Minimal wrapper to run ColBERT-v2 Searcher and save a TSV ranking.

Usage:
  python run_search.py --index INDEX --queries /path/to/queries.tsv --k 100 --output /path/to/out.ranking.tsv

If `--index` is a full index path, the script will derive `index_root` and `index` name.
"""
import os
import sys
from argparse import ArgumentParser

# Ensure ColBERT-v2 is importable (repo root is three levels up)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
COLBERT_ROOT = os.path.join(REPO_ROOT, 'ColBERT-v2')
sys.path.insert(0, COLBERT_ROOT)

from colbert.data import Queries
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig


def main():
    parser = ArgumentParser()
    parser.add_argument('--index', required=True, help='Index name or path')
    parser.add_argument('--index_root', default='indices', help='Optional index root path')
    parser.add_argument('--checkpoint', default='colbert-ir/colbertv2.0', help='Checkpoint path or HF id to use if index metadata lacks it')
    parser.add_argument('--queries', required=True, help='Queries TSV (qid\ttext)')
    parser.add_argument('--k', type=int, default=100, help='k results to retrieve')
    parser.add_argument('--output', required=True, help='Output ranking TSV')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output ranking')
    parser.add_argument('--experiment', default='repro-msmarco', help='Run experiment name')
    parser.add_argument('--nranks', type=int, default=1, help='Number of ranks (for RunConfig)')
    parser.add_argument('--collection', default='data/msmarco-passage/collection.tsv', help='Collection CSV/TSV path to use if index metadata lacks it')

    parser.add_argument('--run_root', default=REPO_ROOT, help='Root folder for ColBERT Run/outputs')

    args = parser.parse_args()

    # If index is a full path, split into index_root and index name
    index_arg = args.index
    index_root = args.index_root

    if os.path.isdir(index_arg) and os.path.exists(os.path.join(index_arg, 'metadata.json')):
        index_root = os.path.dirname(index_arg)
        index_name = os.path.basename(index_arg)
    else:
        index_name = index_arg

    # Prepare the run root
    run_root = os.path.abspath(args.run_root)
    os.makedirs(run_root, exist_ok=True)

    # Do not create a ColBERT Run/experiment directory here; callers expect outputs
    # to be written to the explicit output path provided. Create a direct config
    # and run the search without using Run().context.
    config = ColBERTConfig(root=run_root, index_root=index_root)

    # Prefer checkpoint from args if index metadata does not include it.
    searcher = Searcher(index=index_name, checkpoint=args.checkpoint, collection=args.collection, index_root=index_root, config=config)

    queries = Queries(args.queries)

    ranking = searcher.search_all(queries, k=args.k)

    # Ensure the desired output directory exists and write the ranking to the
    # absolute path the caller supplied so we avoid Run-root relative paths.
    outpath_target = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(outpath_target), exist_ok=True)

    # If requested, remove existing output so Run().open can write it.
    if args.overwrite:
        # remove output file and related metadata/lock files if present
        candidates = [outpath_target, f"{outpath_target}.meta", f"{outpath_target}.lock", f"{outpath_target}.meta.lock"]
        for p in candidates:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    outpath = ranking.save(outpath_target)
    print(f"Saved ranking to {outpath}")


if __name__ == '__main__':
    main()
