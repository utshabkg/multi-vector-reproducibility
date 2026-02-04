#!/usr/bin/env python3
"""
Build IVF (Inverted File) structure for BEIR PLAID indices.

The IVF maps centroids to document IDs, which is required for PLAID search.
This script adds the missing ivf.pt and ivf.pid.pt files to existing indices.

Usage:
    python experiments/15_build_ivf_for_beir.py --dataset scifact
    python experiments/15_build_ivf_for_beir.py --all
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ujson
from colbert.indexing.codecs.residual import ResidualCodec

# Priority BEIR datasets
BEIR_DATASETS = ['arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact', 'trec-covid', 'webis-touche2020']


def load_doclens(index_path):
    """Load document lengths from index"""
    doclens_files = sorted([f for f in os.listdir(index_path) if f.startswith('doclens.')])

    all_doclens = []
    for doclens_file in doclens_files:
        with open(os.path.join(index_path, doclens_file)) as f:
            doclens = ujson.load(f)
            all_doclens.extend(doclens)

    return all_doclens


def build_ivf(index_path, verbose=True, force=False):
    """
    Build IVF structure for a PLAID index

    Args:
        index_path: Path to PLAID index directory
        verbose: Print progress messages
        force: Force rebuild even if IVF exists

    Returns:
        None (saves ivf.pt and ivf.pid.pt to index_path)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Building IVF for {os.path.basename(index_path)}")
        print(f"{'='*80}\n")

    # Check if IVF already exists
    ivf_path = os.path.join(index_path, 'ivf.pt')
    ivf_pid_path = os.path.join(index_path, 'ivf.pid.pt')

    if os.path.exists(ivf_pid_path) and not force:
        if verbose:
            print(f"✓ IVF already exists at {ivf_pid_path}")
            print(f"  Use --force to rebuild")
        return

    # Load plan
    plan_path = os.path.join(index_path, 'plan.json')
    with open(plan_path) as f:
        plan = ujson.load(f)

    num_chunks = plan['num_chunks']
    num_partitions = plan['num_partitions']
    num_embeddings = plan['num_embeddings']

    if verbose:
        print(f"Index info:")
        print(f"  - Chunks: {num_chunks}")
        print(f"  - Partitions (centroids): {num_partitions:,}")
        print(f"  - Embeddings: {num_embeddings:,}\n")

    # Step 1: Load all codes (centroid assignments)
    if verbose:
        print("Step 1: Loading codes from chunks...")

    codes = torch.zeros(num_embeddings, dtype=torch.long)

    for chunk_idx in tqdm(range(num_chunks), desc="Loading codes", disable=not verbose):
        chunk_codes_path = os.path.join(index_path, f'{chunk_idx}.codes.pt')
        chunk_codes = torch.load(chunk_codes_path)

        # Get offset from metadata
        metadata_path = os.path.join(index_path, f'{chunk_idx}.metadata.json')
        with open(metadata_path) as f:
            metadata = ujson.load(f)

        offset = metadata['passage_offset'] * 32  # ConstBERT has 32 vectors/doc
        chunk_codes = chunk_codes.flatten()  # Flatten to 1D
        codes[offset:offset+len(chunk_codes)] = chunk_codes

    if verbose:
        print(f"  ✓ Loaded {len(codes):,} codes\n")

    # Step 2: Sort codes to group by centroid
    if verbose:
        print("Step 2: Sorting codes by centroid...")

    sorted_codes = codes.sort()
    ivf = sorted_codes.indices  # Indices after sorting (embedding IDs grouped by centroid)
    values = sorted_codes.values  # Sorted centroid IDs

    if verbose:
        print(f"  ✓ Sorted codes\n")

    # Step 3: Compute IVF lengths (how many embeddings per centroid)
    if verbose:
        print("Step 3: Computing IVF lengths...")

    ivf_lengths = torch.bincount(values, minlength=num_partitions)
    assert ivf_lengths.size(0) == num_partitions

    if verbose:
        print(f"  ✓ Computed IVF lengths")
        print(f"    - Min embeddings per centroid: {ivf_lengths.min().item()}")
        print(f"    - Max embeddings per centroid: {ivf_lengths.max().item()}")
        print(f"    - Mean embeddings per centroid: {ivf_lengths.float().mean().item():.1f}\n")

    # Step 4: Save embedding-level IVF
    if verbose:
        print("Step 4: Saving embedding-level IVF...")

    torch.save((ivf, ivf_lengths), ivf_path)

    if verbose:
        print(f"  ✓ Saved to {ivf_path}\n")

    # Step 5: Build passage-level IVF (centroid -> unique passage IDs)
    if verbose:
        print("Step 5: Building passage-level IVF...")

    # Load doclens
    all_doclens = load_doclens(index_path)
    total_num_embeddings = sum(all_doclens)

    assert total_num_embeddings == num_embeddings, f"Mismatch: {total_num_embeddings} vs {num_embeddings}"

    # Build emb2pid mapping
    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

    offset_doclens = 0
    for pid, dlength in enumerate(all_doclens):
        emb2pid[offset_doclens:offset_doclens + dlength] = pid
        offset_doclens += dlength

    if verbose:
        print(f"  - Built emb2pid mapping: {len(emb2pid):,} embeddings")

    # Convert embedding IVF to passage IVF
    ivf_pids = emb2pid[ivf]  # Map embedding indices to passage IDs

    # For each centroid, get unique passage IDs
    unique_pids_per_centroid = []
    ivf_pid_lengths = []

    offset = 0
    for length in tqdm(ivf_lengths.tolist(), desc="Processing centroids", disable=not verbose):
        pids = torch.unique(ivf_pids[offset:offset+length])
        unique_pids_per_centroid.append(pids)
        ivf_pid_lengths.append(pids.shape[0])
        offset += length

    ivf_pid = torch.cat(unique_pids_per_centroid)
    ivf_pid_lengths = torch.tensor(ivf_pid_lengths)

    # Add padding for safe access
    max_stride = ivf_pid_lengths.max().item()
    zero = torch.zeros(1, dtype=torch.long)
    offsets = torch.cat((zero, torch.cumsum(ivf_pid_lengths, dim=0)))

    if offsets[-2] + max_stride > ivf_pid.size(0):
        padding = torch.zeros(max_stride, dtype=ivf_pid.dtype)
        ivf_pid = torch.cat((ivf_pid, padding))

    # Save passage-level IVF
    torch.save((ivf_pid, ivf_pid_lengths), ivf_pid_path)

    if verbose:
        print(f"  ✓ Saved passage-level IVF to {ivf_pid_path}")
        print(f"    - Total unique passage references: {ivf_pid.size(0):,}")
        print(f"    - Min passages per centroid: {ivf_pid_lengths.min().item()}")
        print(f"    - Max passages per centroid: {ivf_pid_lengths.max().item()}")
        print(f"    - Mean passages per centroid: {ivf_pid_lengths.float().mean().item():.1f}\n")

    # Step 6: Update metadata
    if verbose:
        print("Step 6: Updating metadata...")

    metadata_path = os.path.join(index_path, 'metadata.json')
    metadata = {
        'num_chunks': num_chunks,
        'num_partitions': num_partitions,
        'num_embeddings': num_embeddings,
        'avg_doclen': num_embeddings / len(all_doclens),
        'config': {
            'doc_maxlen': 220,
            'nbits': 1,
            'dim': 128,
            'kmeans_niters': 4,
        }
    }

    with open(metadata_path, 'w') as f:
        ujson.dump(metadata, f, indent=2)

    if verbose:
        print(f"  ✓ Updated metadata.json\n")

    print(f"{'='*80}")
    print(f"✅ IVF built successfully!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Build IVF for BEIR PLAID indices')
    parser.add_argument('--dataset', type=str, choices=BEIR_DATASETS,
                       help='BEIR dataset name')
    parser.add_argument('--all', action='store_true',
                       help='Build IVF for all 5 priority datasets')
    parser.add_argument('--index_root', type=str,
                       default='/media/12TB/shared/datasets/indices/beir_plaid',
                       help='Root directory containing PLAID indices')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if IVF already exists (useful for fixing metadata)')

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify either --dataset or --all")

    # Determine which datasets to process
    if args.all:
        datasets = BEIR_DATASETS
    else:
        datasets = [args.dataset]

    print(f"\n{'='*80}")
    print(f"Building IVF for {len(datasets)} BEIR PLAID index(es)")
    print(f"{'='*80}\n")
    print(f"Index root: {args.index_root}\n")

    # Process each dataset
    for dataset_name in datasets:
        index_path = os.path.join(args.index_root, f'{dataset_name}_constbert_plaid')

        if not os.path.exists(index_path):
            print(f"❌ Index not found: {index_path}")
            continue

        try:
            build_ivf(index_path, verbose=True, force=args.force)
        except Exception as e:
            print(f"❌ Error building IVF for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ All IVFs built successfully!")
    print(f"{'='*80}\n")
    print("Indices are now ready for PLAID evaluation.")
    print("Run: ./run_beir_plaid_evaluation.sh\n")


if __name__ == "__main__":
    main()
