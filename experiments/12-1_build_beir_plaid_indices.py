#!/usr/bin/env python3
"""
Build PLAID indices for BEIR datasets using ConstBERT.

This script:
1. Loads BEIR datasets
2. Encodes documents using ConstBERT
3. Saves embeddings (N, 32, 128) to files
4. Builds PLAID index using ColBERT's ResidualCodec
5. Saves PLAID indices for retrieval

Usage:
    python experiments/12-1_build_beir_plaid_indices.py constbert --all
    python experiments/12-1_build_beir_plaid_indices.py constbert --dataset scidocs
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constbert_wrapper import ConstBERTWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BEIRDataset:
    """Loader for BEIR datasets"""

    DATASETS = {
        'trec-covid': {
            'name': 'TREC-COVID',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip',
            'size': '171K docs'
        },
        'nfcorpus': {
            'name': 'NFCorpus',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip',
            'size': '3.6K docs'
        },
        'fiqa': {
            'name': 'FiQA-2018',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip',
            'size': '57K docs'
        },
        'scifact': {
            'name': 'SciFact',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
            'size': '5K docs'
        },
        'arguana': {
            'name': 'ArguAna',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip',
            'size': '8.7K docs'
        },
        'scidocs': {
            'name': 'SCIDOCS',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip',
            'size': '25K docs'
        },
        'quora': {
            'name': 'Quora',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip',
            'size': '523K docs'
        },
        'dbpedia-entity': {
            'name': 'DBPedia-Entity',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip',
            'size': '4.6M docs'
        },
        'fever': {
            'name': 'FEVER',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip',
            'size': '5.4M docs'
        },
        'climate-fever': {
            'name': 'Climate-FEVER',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip',
            'size': '5.4M docs'
        },
        'hotpotqa': {
            'name': 'HotpotQA',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip',
            'size': '5.2M docs'
        },
        'nq': {
            'name': 'Natural Questions',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip',
            'size': '2.7M docs'
        },
        'webis-touche2020': {
            'name': 'Touche-2020',
            'url': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip',
            'size': '382K docs'
        }
    }

    def __init__(self, dataset_name: str, data_dir: str = "./data/beir"):
        """Initialize BEIR dataset loader"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.dataset_path = self.data_dir / dataset_name

        logger.info(f"Initializing BEIR dataset: {self.DATASETS[dataset_name]['name']}")

    def download(self):
        """Download BEIR dataset if not already present"""
        if self.dataset_path.exists():
            logger.info(f"Dataset already exists at {self.dataset_path}")
            return

        logger.info(f"Downloading {self.dataset_name} from BEIR...")

        try:
            from beir import util
            self.data_dir.mkdir(parents=True, exist_ok=True)
            url = self.DATASETS[self.dataset_name]['url']
            util.download_and_unzip(url, str(self.data_dir))
            logger.info(f"Downloaded to {self.dataset_path}")
        except ImportError:
            logger.error("BEIR library not installed. Install with: pip install beir")
            logger.info(f"Alternatively, manually download from: {self.DATASETS[self.dataset_name]['url']}")
            raise

    def load(self) -> Tuple[Dict, Dict, Dict]:
        """Load corpus, queries, and qrels"""
        from beir.datasets.data_loader import GenericDataLoader

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Run download() first.")

        logger.info(f"Loading {self.dataset_name} dataset...")
        corpus, queries, qrels = GenericDataLoader(str(self.dataset_path)).load(split="test")
        logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")

        return corpus, queries, qrels


class DocumentDataset(Dataset):
    """PyTorch Dataset for parallel document loading"""
    def __init__(self, doc_ids: List[str], corpus: Dict):
        self.doc_ids = doc_ids
        self.corpus = corpus

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        doc = self.corpus[doc_id]
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        return text.strip()


def encode_beir_corpus(model, corpus: Dict, batch_size: int = 256,
                                  num_workers: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    Encode corpus with parallel data loading

    Returns:
        embeddings: numpy array of shape (num_docs, 32, 128)
        doc_ids: list of document IDs
    """
    logger.info(f"Encoding {len(corpus)} documents with {num_workers} workers...")

    doc_ids = list(corpus.keys())
    dataset = DocumentDataset(doc_ids, corpus)

    # Use DataLoader for parallel text preprocessing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4  # Prefetch 4 batches per worker
    )

    all_embeddings = []

    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Encoding documents"):
            # Model encode internally handles tokenization and GPU transfer
            batch_emb = model.encode_documents(
                batch_texts,
                batch_size=batch_size,  # Already batched by DataLoader
                show_progress=False
            )
            all_embeddings.append(batch_emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Encoded {len(doc_ids)} documents, shape: {embeddings.shape}")

    return embeddings, doc_ids


def flatten_constbert_embeddings(embeddings):
    """
    Flatten ConstBERT embeddings from (N, 32, 128) to (N*32, 128).

    Also create doclens array indicating how many vectors per document.
    For ConstBERT, this is constant: [32, 32, 32, ...]
    """
    N, C, dim = embeddings.shape
    assert C == 32, f"Expected C=32, got {C}"
    assert dim == 128, f"Expected dim=128, got {dim}"

    print(f"\nFlattening embeddings...")
    print(f"  From: {embeddings.shape} (N, C, dim)")

    # Flatten to (N*C, dim)
    flattened = embeddings.reshape(N * C, dim)

    # Create doclens: [32, 32, ..., 32] repeated N times
    doclens = [C] * N

    print(f"  To: {flattened.shape} (N*C, dim)")
    print(f"  Document lengths: constant {C} vectors/doc")

    return flattened, doclens


def train_plaid_centroids(embeddings_flat, config, sample_size=10_000_000):
    """
    Train PLAID centroids using k-means clustering.

    Args:
        embeddings_flat: (N*C, 128) flattened embeddings
        config: ColBERTConfig with nbits, kmeans_niters, dim
        sample_size: Number of vectors to sample for training

    Returns:
        ResidualCodec initialized with trained centroids
    """
    print(f"\n{'='*80}")
    print("Training PLAID Centroids")
    print(f"{'='*80}")

    # Sample vectors for training
    num_total_vectors = len(embeddings_flat)
    num_sample = min(sample_size, num_total_vectors)

    print(f"Total vectors: {num_total_vectors:,}")
    print(f"Sample size: {num_sample:,}")

    if num_sample < num_total_vectors:
        indices = np.random.choice(num_total_vectors, size=num_sample, replace=False)
        sample = embeddings_flat[indices]
    else:
        sample = embeddings_flat

    # Convert to torch
    sample_torch = torch.from_numpy(sample).float()

    # Determine number of centroids (partitions)
    # ColBERT uses 2^(ceil(log2(sqrt(num_embeddings))))
    # For large datasets, this can be many centroids
    num_partitions = int(2 ** np.ceil(np.log2(np.sqrt(num_total_vectors))))
    num_partitions = max(1024, min(num_partitions, 65536))  # Clamp between 1K-64K

    print(f"Number of centroids (partitions): {num_partitions:,}")
    print(f"K-means iterations: {config.kmeans_niters}")
    print(f"Embedding dim: {config.dim}")
    print(f"Quantization bits: {config.nbits}")

    # Run k-means
    import faiss

    print(f"\nRunning FAISS k-means...")
    start_time = time.time()

    kmeans = faiss.Kmeans(
        d=config.dim,
        k=num_partitions,
        niter=config.kmeans_niters,
        verbose=True,
        gpu=torch.cuda.is_available(),
        seed=123
    )

    kmeans.train(sample_torch.numpy())

    elapsed = time.time() - start_time
    print(f"✅ K-means completed in {elapsed:.1f}s")

    # Get centroids
    centroids = torch.from_numpy(kmeans.centroids)
    print(f"Centroids shape: {centroids.shape}")

    # Train residual codec (bucket quantization)
    print(f"\nTraining residual codec...")
    codec = train_residual_codec(sample_torch, centroids, config)

    return codec


def train_residual_codec(sample, centroids, config):
    """
    Train residual quantization buckets.

    This computes statistics on the residuals to determine
    optimal bucket boundaries for quantization.
    """
    from colbert.infra import ColBERTConfig
    from colbert.indexing.codecs.residual import ResidualCodec

    # Move to GPU if available
    if torch.cuda.is_available():
        sample = sample.cuda().half()
        centroids = centroids.cuda().half()

    # Assign each vector to nearest centroid
    print("  - Finding nearest centroids...")
    with torch.no_grad():
        # Compute distances in batches
        batch_size = 100_000
        assignments = []
        for i in range(0, len(sample), batch_size):
            batch = sample[i:i+batch_size]
            # Cosine similarity (embeddings are normalized)
            scores = batch @ centroids.T
            assignments.append(scores.argmax(dim=1))
        assignments = torch.cat(assignments)

    # Compute residuals in batches to save memory
    print("  - Computing residuals...")
    residuals_list = []
    batch_size = 100_000
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i+batch_size]
        batch_assignments = assignments[i:i+batch_size]
        residuals_batch = batch - centroids[batch_assignments]
        # Move to CPU immediately to free GPU memory
        residuals_list.append(residuals_batch.cpu())
        del residuals_batch
    residuals = torch.cat(residuals_list)
    del residuals_list

    # Compute bucket statistics on CPU
    print("  - Computing bucket boundaries...")
    num_options = 2 ** config.nbits

    # Sample residuals for quantile computation (to save memory)
    # Use 100K samples instead of all 10M (100K × 128 = 12.8M elements, manageable)
    num_residual_samples = min(100_000, len(residuals))
    if num_residual_samples < len(residuals):
        indices = torch.randperm(len(residuals))[:num_residual_samples]
        residuals_sample = residuals[indices]
    else:
        residuals_sample = residuals

    # Flatten residuals across all dimensions and convert to numpy for quantile
    residuals_flat = residuals_sample.flatten().numpy()

    # Compute quantiles using numpy (more memory-efficient)
    quantiles_np = np.linspace(0, 1, num_options + 1)[1:-1]  # Exclude 0 and 1
    bucket_cutoffs = torch.from_numpy(np.quantile(residuals_flat, quantiles_np)).float()

    # Compute bucket weights (midpoints of buckets)
    bucket_weights = torch.zeros(num_options)
    bucket_weights[0] = float(residuals_flat.min())
    for i in range(1, num_options):
        bucket_weights[i] = (bucket_cutoffs[i-1] + bucket_cutoffs[i]) / 2 if i < num_options - 1 else float(residuals_flat.max())

    # Compute average residual (on sample to save memory)
    avg_residual = residuals_sample.mean()

    print(f"  - Bucket cutoffs: {bucket_cutoffs.numpy()[:5]}... ({len(bucket_cutoffs)} buckets)")
    print(f"  - Average residual: {avg_residual.item():.6f}")
    print(f"  - Used {num_residual_samples:,} residual samples for statistics")

    # Create codec (all tensors already on CPU)
    codec = ResidualCodec(
        config=config,
        centroids=centroids.cpu(),
        avg_residual=avg_residual,
        bucket_cutoffs=bucket_cutoffs,
        bucket_weights=bucket_weights
    )

    return codec


def build_plaid_index(embeddings_path, doc_ids_path, index_path, config):
    """
    Build complete PLAID index from ConstBERT embeddings.

    Args:
        embeddings_path: Path to .npy file with (N, 32, 128) embeddings
        doc_ids_path: Path to .npy file with document IDs
        index_path: Output directory for PLAID index
    """
    print(f"\n{'='*80}")
    print("Building PLAID Index from ConstBERT Embeddings")
    print(f"{'='*80}\n")

    # Create index directory
    os.makedirs(index_path, exist_ok=True)

    # Load embeddings and doc_ids
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)  # (N, 32, 128) float16/float32
    print(f"Loading doc_ids from {doc_ids_path}...")
    doc_ids = np.load(doc_ids_path)

    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Num documents: {len(doc_ids)}")
    print(f"  - Dtype: {embeddings.dtype}")

    # Flatten for PLAID
    embeddings_flat, doclens = flatten_constbert_embeddings(embeddings)

    # Train PLAID codec
    codec = train_plaid_centroids(embeddings_flat, config)

    # Save codec
    print(f"\nSaving codec to {index_path}...")
    codec.save(index_path)
    print("  ✅ Codec saved")

    # Compress and save embeddings in chunks
    print(f"\nCompressing and saving embeddings...")
    chunk_size = 50_000  # Documents per chunk
    num_chunks = int(np.ceil(len(doc_ids) / chunk_size))

    print(f"  - Total documents: {len(doc_ids):,}")
    print(f"  - Chunk size: {chunk_size:,} documents")
    print(f"  - Number of chunks: {num_chunks}")

    offset = 0
    for chunk_idx in tqdm(range(num_chunks), desc="Compressing chunks"):
        # Get chunk of documents
        start_doc = chunk_idx * chunk_size
        end_doc = min((chunk_idx + 1) * chunk_size, len(doc_ids))

        # Get corresponding vectors
        start_vec = start_doc * 32
        end_vec = end_doc * 32

        # Extract chunk embeddings
        chunk_embs = torch.from_numpy(embeddings_flat[start_vec:end_vec])
        chunk_doclens = doclens[start_doc:end_doc]

        # Compress with PLAID
        compressed = codec.compress(chunk_embs)

        # Save chunk files
        path_prefix = os.path.join(index_path, str(chunk_idx))
        compressed.save(path_prefix)

        # Save doclens
        import ujson
        doclens_path = os.path.join(index_path, f'doclens.{chunk_idx}.json')
        with open(doclens_path, 'w') as f:
            ujson.dump(chunk_doclens, f)

        # Save metadata
        metadata_path_chunk = os.path.join(index_path, f'{chunk_idx}.metadata.json')
        with open(metadata_path_chunk, 'w') as f:
            metadata = {
                'passage_offset': offset,
                'num_passages': len(chunk_doclens),
                'num_embeddings': len(compressed)
            }
            ujson.dump(metadata, f)

        offset += len(chunk_doclens)

    # Save plan
    print(f"\nSaving index plan...")
    plan_path = os.path.join(index_path, 'plan.json')
    import ujson
    plan = {
        'num_chunks': num_chunks,
        'num_partitions': len(codec.centroids),
        'num_embeddings': len(embeddings_flat),
        'avg_doclen': 32,  # ConstBERT always has 32 vectors/doc
        'num_documents': len(doc_ids)
    }
    with open(plan_path, 'w') as f:
        ujson.dump(plan, f, indent=2)

    print("  ✅ Plan saved")

    # Save doc_ids mapping with index for self-contained retrieval
    print(f"\nSaving document ID mapping...")
    doc_ids_save_path = os.path.join(index_path, 'doc_ids.npy')
    np.save(doc_ids_save_path, doc_ids)
    print(f"  ✅ Document IDs saved ({len(doc_ids):,} documents)")

    # Calculate total index size
    import subprocess
    result = subprocess.run(['du', '-sh', index_path], capture_output=True, text=True)
    index_size = result.stdout.strip().split()[0]

    print(f"\n{'='*80}")
    print(f"✅ PLAID Index Built Successfully!")
    print(f"{'='*80}")
    print(f"Index path: {index_path}")
    print(f"Index size: {index_size}")
    print(f"Documents: {len(doc_ids):,}")
    print(f"Vectors: {len(embeddings_flat):,}")
    print(f"Centroids: {len(codec.centroids):,}")
    print(f"Quantization: {config.nbits}-bit")
    print(f"{'='*80}\n")


def process_dataset(dataset_name, args, config):
    """Process a single BEIR dataset: encode and build PLAID index"""
    logger.info(f"\n{'='*100}")
    logger.info(f"PROCESSING: {dataset_name}")
    logger.info(f"{'='*100}\n")

    try:
        # Load dataset
        beir_dataset = BEIRDataset(dataset_name, args.data_dir)
        beir_dataset.download()
        corpus, _, _ = beir_dataset.load()

        # Create output directory for embeddings
        embeddings_dir = Path(args.output_dir) / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Paths for embeddings
        embeddings_path = embeddings_dir / f"{dataset_name}_constbert_embeddings.npy"
        doc_ids_path = embeddings_dir / f"{dataset_name}_constbert_doc_ids.npy"

        # Check if embeddings already exist
        if embeddings_path.exists() and doc_ids_path.exists():
            logger.info(f"Embeddings already exist for {dataset_name}, skipping encoding")
        else:
            # Load model
            logger.info(f"Loading {args.model} model...")
            if args.model == 'constbert':
                model = ConstBERTWrapper(device='cuda' if torch.cuda.is_available() else 'cpu')
            else:
                raise NotImplementedError(f"Model {args.model} not supported")

            # Encode corpus
            start_time = time.time()
            embeddings, doc_ids = encode_beir_corpus(
                model, corpus, args.batch_size, args.num_workers
            )
            encode_time = time.time() - start_time
            logger.info(f"✓ Document encoding: {encode_time:.2f}s ({len(corpus)/encode_time:.0f} docs/sec)")

            # Save embeddings
            logger.info(f"Saving embeddings to {embeddings_path}")
            np.save(embeddings_path, embeddings)
            np.save(doc_ids_path, doc_ids)
            logger.info(f"✓ Saved embeddings: {embeddings.shape}, doc_ids: {len(doc_ids)}")

        # Build PLAID index
        index_path = Path(args.output_dir) / f"{dataset_name}_constbert_plaid"
        logger.info(f"Building PLAID index at {index_path}")

        build_plaid_index(str(embeddings_path), str(doc_ids_path), str(index_path), config)

        logger.info(f"✅ Successfully processed {dataset_name}")

    except Exception as e:
        logger.error(f"Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Build PLAID indices for BEIR datasets using ConstBERT')
    parser.add_argument('model', type=str, choices=['constbert'],
                       help='Model to use for encoding')
    parser.add_argument('--dataset', type=str,
                       help='Specific BEIR dataset name')
    parser.add_argument('--all', action='store_true',
                       help='Process all 13 BEIR datasets')
    parser.add_argument('--data-dir', type=str, default='./data/beir',
                       help='Directory for BEIR data')
    parser.add_argument('--output-dir', type=str, default='/media/12TB/shared/datasets/indices/beir_plaid',
                       help='Output directory for PLAID indices')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Encoding batch size')
    parser.add_argument('--num-workers', type=int, default=32,
                       help='DataLoader workers')

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Must specify either --dataset or --all")
    if args.all and args.dataset:
        parser.error("Cannot specify both --dataset and --all")

    # Determine datasets
    if args.all:
        datasets = ['arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa',
                    'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact',
                    'trec-covid', 'webis-touche2020']
    else:
        datasets = [args.dataset]

    # Setup ColBERT config for PLAID
    from colbert.infra import ColBERTConfig
    config = ColBERTConfig(
        dim=128,
        nbits=1,  # PLAID 1-bit quantization (as in paper)
        kmeans_niters=4,
        doc_maxlen=220,
        index_bsize=64
    )

    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"CPU cores: {mp.cpu_count()}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Datasets to process: {datasets}")

    # Process each dataset
    for dataset_name in datasets:
        process_dataset(dataset_name, args, config)

    logger.info("\n🎉 All datasets processed successfully!")


if __name__ == "__main__":
    main()