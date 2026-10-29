#!/usr/bin/env python3
"""Wrapper to create a ColBERT-v2 index for a collection.

Example:
    python index_colbertv2.py \
        --collection /media/12TB/shared/datasets/raw/msmarco-passage/collection.tsv \
    --index-name msmarco-passage-colbertv2 \
    --checkpoint colbert-ir/colbertv2.0 \
    --output-root ./experiments
"""
import os
from argparse import ArgumentParser

# ColBERT should be installed via: pip install colbert-ai

from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def main():
    parser = ArgumentParser()
    parser.add_argument('--collection', required=True, help='Path to collection TSV (pid\ttext)')
    parser.add_argument('--index-name', required=True, help='Index name to create (e.g. msmarco-passage-colbertv2)')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path or HF repo id (e.g. colbert-ir/colbertv2.0)')
    parser.add_argument('--output-root', default='/media/12TB/shared/datasets/indices', help='Experiments root (defaults to shared indices root)')
    parser.add_argument('--indices-root', default='/media/12TB/shared/datasets/indices', help='Root folder for created indexes')
    parser.add_argument('--nranks', type=int, default=1, help='Number of ranks/processes for indexing')
    parser.add_argument('--overwrite', choices=['false','true','reuse','resume','force_silent_overwrite'], default='false')
    parser.add_argument('--experiment', default='index-msmarco-v2')
    parser.add_argument('--dry-run', action='store_true', help='Create plan.json only (no heavy encoding)')

    args = parser.parse_args()

    overwrite_map = {
        'false': False,
        'true': True,
        'reuse': 'reuse',
        'resume': 'resume',
        'force_silent_overwrite': 'force_silent_overwrite'
    }

    os.makedirs(args.output_root, exist_ok=True)

    # Ensure Run().config.root matches the desired output root so that
    # runtime paths (Run().path_, index_root_) are created under the
    # shared indices directory rather than the local experiments folder.
    with Run().context(RunConfig(root=args.output_root, nranks=args.nranks, experiment=args.experiment)):
        config = ColBERTConfig(root=args.output_root, index_root=args.indices_root)

        index_path = os.path.join(args.indices_root, args.index_name)

        if args.dry_run:
            # Fast approximate plan creation without loading model or encoding
            print(f"Dry-run: Creating plan.json at {index_path} ...")
            os.makedirs(index_path, exist_ok=True)

            # Count passages and sample avg doc length
            num_passages = 0
            total_tokens = 0
            sample_count = 0
            sample_limit = 1000

            with open(args.collection, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    num_passages += 1
                    if sample_count < sample_limit:
                        parts = line.split('\t', 1)
                        text = parts[1] if len(parts) > 1 else parts[0]
                        total_tokens += len(text.split())
                        sample_count += 1

            avg_doclen_est = (total_tokens / sample_count) if sample_count > 0 else 120

            import math
            num_chunks = int(math.ceil(num_passages / min(25000, 1 + (num_passages // args.nranks))))
            num_embeddings_est = int(num_passages * avg_doclen_est)
            num_partitions = int(2 ** math.floor(math.log2(16 * math.sqrt(max(1, num_embeddings_est))))) if num_embeddings_est > 0 else 16

            plan = {
                'config': config.export(),
                'num_chunks': num_chunks,
                'num_partitions': num_partitions,
                'num_embeddings_est': num_embeddings_est,
                'avg_doclen_est': avg_doclen_est,
            }

            import ujson
            plan_path = os.path.join(index_path, 'plan.json')
            with open(plan_path, 'w') as f:
                f.write(ujson.dumps(plan, indent=4) + '\n')

            print(f"Wrote plan.json ({num_passages} passages, avg_doclen={avg_doclen_est:.2f}) to {plan_path}")
        else:
            indexer = Indexer(checkpoint=args.checkpoint, config=config)

            print(f"Indexing collection {args.collection} with checkpoint {args.checkpoint}...")
            index_path = indexer.index(name=args.index_name, collection=args.collection, overwrite=overwrite_map[args.overwrite])

            # If the Indexer created the index under a Run/experiment nested path
            # (e.g. <indices_root>/<experiment>/indexes/<index_name>), relocate it
            # to the flat canonical path <indices_root>/<index_name> to match our
            # repository conventions and avoid future path confusion.
            desired_path = os.path.join(args.indices_root, args.index_name)

            if os.path.abspath(index_path) != os.path.abspath(desired_path):
                import shutil
                print(f"Relocating index from {index_path} to {desired_path} ...")
                os.makedirs(os.path.dirname(desired_path), exist_ok=True)
                # move the produced index folder
                shutil.move(index_path, desired_path)
                # attempt to remove empty parent experiment folders if present
                parent = os.path.dirname(index_path)
                try:
                    os.rmdir(parent)
                except Exception:
                    pass
                index_path = desired_path

            print(f"Index created at: {index_path}")


if __name__ == '__main__':
    main()
