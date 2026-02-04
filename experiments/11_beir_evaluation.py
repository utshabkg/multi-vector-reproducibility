#!/usr/bin/env python3
"""
OPTIMIZED BEIR Evaluation Script for ConstBERT and ColBERT-v2
Maximizes hardware utilization: 120 CPU cores, 700GB RAM, 2x RTX 6000 Ada GPUs

Key Optimizations:
1. Parallel document encoding with DataLoader workers
2. Batched MaxSim computation on GPU
3. Multi-GPU FAISS with query batching
4. Memory-efficient processing
5. Vectorized operations throughout

Usage:
    python experiments/11_beir_evaluation.py --model constbert --dataset scidocs
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

# FAISS imports
import faiss

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constbert_wrapper import ConstBERTWrapper
from evaluation.metrics import evaluate_retrieval

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
    OPTIMIZED: Encode corpus with parallel data loading
    
    Key optimizations:
    - DataLoader with multiple workers for parallel preprocessing
    - Larger batch size to saturate GPU
    - Pin memory for faster GPU transfer
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


def encode_beir_queries(model, queries: Dict, batch_size: int = 256) -> Tuple[np.ndarray, List[str]]:
    """OPTIMIZED: Encode queries with larger batches"""
    logger.info(f"Encoding {len(queries)} queries...")
    
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    embeddings = model.encode_queries(query_texts, batch_size=batch_size, show_progress=True)
    logger.info(f"Encoded {len(query_ids)} queries, shape: {embeddings.shape}")
    
    return embeddings, query_ids


class OptimizedFAISSIndex:
    """
    OPTIMIZED Multi-vector FAISS index with GPU acceleration
    
    Key optimizations:
    1. GPU-based FAISS search
    2. Batched MaxSim computation on GPU
    3. Memory-efficient candidate retrieval
    """
    
    def __init__(self, use_gpu_maxsim: bool = True):
        """
        Use CPU FAISS (stable) but GPU for MaxSim computation (fast)
        """
        self.use_gpu_maxsim = use_gpu_maxsim
        self.index = None
        self.doc_ids = None
        self.doc_embeddings = None
        self.num_vectors_per_doc = None
        
    def build_index(self, doc_embeddings: np.ndarray, doc_ids: List[str], 
                   nlist: int = 256, nprobe: int = 64):
        """
        Build GPU-accelerated FAISS index
        
        Args:
            doc_embeddings: shape (num_docs, num_vectors_per_doc, dim)
            doc_ids: list of document IDs
            nlist: number of IVF cells
            nprobe: number of cells to probe at search time
        """
        num_docs, num_vectors_per_doc, dim = doc_embeddings.shape
        logger.info(f"Building FAISS index: {num_docs} docs, {num_vectors_per_doc} vectors/doc, dim={dim}")
        
        # Flatten embeddings for FAISS: (num_docs * num_vectors_per_doc, dim)
        flat_embeddings = doc_embeddings.reshape(-1, dim).astype('float32')
        
        # Store for MaxSim computation
        self.doc_ids = doc_ids
        self.doc_embeddings = torch.from_numpy(doc_embeddings).cuda()  # Keep on GPU!
        self.num_vectors_per_doc = num_vectors_per_doc
        
        # Build IVF index with GPU
        logger.info(f"Creating IVF index with nlist={nlist}...")
        
        # Build CPU IVF index (stable, no hanging)
        logger.info(f"Creating CPU IVF index with nlist={nlist}...")
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train index
        logger.info("Training index...")
        if len(flat_embeddings) > 100000:
            sample_size = min(100000, len(flat_embeddings))
            sample_indices = np.random.choice(len(flat_embeddings), sample_size, replace=False)
            train_data = flat_embeddings[sample_indices]
        else:
            train_data = flat_embeddings
        
        self.index.train(train_data)
        
        # Add vectors
        logger.info("Adding vectors to index...")
        self.index.add(flat_embeddings)
        self.index.nprobe = nprobe
        
        logger.info(f"✓ Index built with {self.index.ntotal} vectors (CPU)")
        logger.info(f"✓ MaxSim will use GPU: {self.use_gpu_maxsim}")
    
    def search_batched(self, query_embeddings: np.ndarray, k: int = 1000, 
                      candidate_mult: int = 3, batch_size: int = 512) -> Tuple[List[List[str]], List[List[float]]]:
        """
        OPTIMIZED batched search with GPU MaxSim
        
        Key optimization: Process multiple queries in parallel on GPU
        
        Args:
            query_embeddings: shape (num_queries, num_query_tokens, dim)
            k: number of documents to return per query
            candidate_mult: retrieve candidate_mult * k candidates for MaxSim reranking (reduced to 3 for speed)
            batch_size: number of queries to process together (increased to 512)
        
        Returns:
            doc_ids_all: List of lists of document IDs
            scores_all: List of lists of scores
        """
        num_queries = len(query_embeddings)
        logger.info(f"Searching {num_queries} queries with batch_size={batch_size}, retrieving top-{k}...")
        
        # Move query embeddings to GPU
        query_embeddings_gpu = torch.from_numpy(query_embeddings).cuda()
        
        all_doc_ids = []
        all_scores = []
        
        num_batches = (num_queries + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Batched search"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_queries)
            
            batch_queries = query_embeddings_gpu[start_idx:end_idx]  # (batch, num_tokens, dim)
            
            # Step 1: FAISS candidate retrieval (approximate)
            # Flatten query tokens for FAISS: (batch * num_tokens, dim)
            batch_size_actual, num_tokens, dim = batch_queries.shape
            flat_queries = batch_queries.reshape(-1, dim).cpu().numpy().astype('float32')
            
            # Retrieve candidates (we need more than k to account for MaxSim reranking)
            n_candidates = min(candidate_mult * k, self.index.ntotal)
            _, flat_indices = self.index.search(flat_queries, n_candidates)
            
            # Convert vector indices to document indices
            flat_indices_gpu = torch.from_numpy(flat_indices).cuda()  # (batch * num_tokens, n_candidates)
            doc_indices = flat_indices_gpu // self.num_vectors_per_doc  # Integer division
            
            # Get unique documents per query
            doc_indices = doc_indices.reshape(batch_size_actual, num_tokens, n_candidates)
            
            # For each query, get union of candidate documents across all tokens
            batch_doc_ids_list = []
            batch_scores_list = []
            
            for query_idx in range(batch_size_actual):
                # Get unique candidate docs for this query
                query_doc_indices = doc_indices[query_idx].flatten().unique()
                
                # Limit to reasonable number of candidates
                if len(query_doc_indices) > candidate_mult * k:
                    query_doc_indices = query_doc_indices[:candidate_mult * k]
                
                # Step 2: Compute exact MaxSim scores for candidates on GPU
                query_emb = batch_queries[query_idx]  # (num_tokens, dim)
                candidate_doc_embs = self.doc_embeddings[query_doc_indices]  # (num_candidates, num_vec_per_doc, dim)
                
                # Compute MaxSim: for each query token, max similarity with doc tokens
                # query_emb: (num_tokens, dim)
                # candidate_doc_embs: (num_candidates, num_vec_per_doc, dim)
                # We need: (num_tokens, num_candidates, num_vec_per_doc)
                
                # Expand query: (num_tokens, 1, dim) and doc: (1, num_candidates, num_vec_per_doc, dim)
                # Then compute dot product over last dim
                num_candidates = len(query_doc_indices)
                num_vec_per_doc = candidate_doc_embs.shape[1]
                
                # Reshape for batch matmul: (num_tokens, 1, dim) @ (num_candidates, dim, num_vec_per_doc)
                # But we need to broadcast properly
                query_expanded = query_emb.unsqueeze(1).unsqueeze(2)  # (num_tokens, 1, 1, dim)
                doc_expanded = candidate_doc_embs.unsqueeze(0)  # (1, num_candidates, num_vec_per_doc, dim)
                
                # Compute similarities: sum over last dimension (dot product)
                similarities = (query_expanded * doc_expanded).sum(dim=-1)  # (num_tokens, num_candidates, num_vec_per_doc)
                
                # MaxSim: max over document tokens, sum over query tokens
                max_sims, _ = similarities.max(dim=2)  # (num_tokens, num_candidates)
                maxsim_scores = max_sims.sum(dim=0)  # (num_candidates,)
                
                # Get top-k
                if len(maxsim_scores) > k:
                    topk_scores, topk_indices = maxsim_scores.topk(k)
                else:
                    topk_scores = maxsim_scores
                    topk_indices = torch.arange(len(maxsim_scores), device=maxsim_scores.device)
                
                topk_doc_indices = query_doc_indices[topk_indices]
                
                # Convert to document IDs
                doc_ids = [self.doc_ids[idx.item()] for idx in topk_doc_indices]
                scores = topk_scores.cpu().numpy().tolist()
                
                batch_doc_ids_list.append(doc_ids)
                batch_scores_list.append(scores)
            
            all_doc_ids.extend(batch_doc_ids_list)
            all_scores.extend(batch_scores_list)
        
        return all_doc_ids, all_scores


def retrieve_and_evaluate(index: OptimizedFAISSIndex,
                                    query_embeddings: np.ndarray,
                                    query_ids: List[str],
                                    qrels: Dict,
                                    k: int = 1000,
                                    batch_size: int = 512) -> Dict:
    """
    OPTIMIZED retrieval and evaluation
    """
    logger.info(f"Retrieving top-{k} documents for {len(query_ids)} queries (batched)...")
    
    # Batched search with GPU MaxSim (reduced candidate_mult for speed)
    doc_ids_batch, scores_batch = index.search_batched(
        query_embeddings,
        k=k,
        candidate_mult=3,  # Reduced from 5 to 3 for speed
        batch_size=batch_size
    )
    
    # Convert to results dictionary
    results = {}
    for query_idx, qid in enumerate(query_ids):
        doc_ids = doc_ids_batch[query_idx]
        scores = scores_batch[query_idx]
        results[qid] = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores)}
    
    # Evaluate
    logger.info("Computing metrics...")
    metrics = evaluate_retrieval(results, qrels)
    
    # Convert to simpler format
    final_metrics = {}
    for metric_name, value in metrics.items():
        if '_at_' in metric_name:
            base, k_val = metric_name.split('_at_')
            final_metrics[f"{base}@{k_val}"] = value
        else:
            final_metrics[metric_name] = value
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='OPTIMIZED BEIR evaluation')
    parser.add_argument('--model', type=str, required=True, choices=['constbert', 'colbert'],
                       help='Model to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                       help='BEIR dataset name')
    parser.add_argument('--data-dir', type=str, default='./data/beir',
                       help='Directory for BEIR data')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory for results')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Encoding batch size (increased for 48GB GPU)')
    parser.add_argument('--query-batch-size', type=int, default=512,
                       help='Query search batch size (increased for speed)')
    parser.add_argument('--num-workers', type=int, default=32,
                       help='DataLoader workers (use ~30-50% of CPU cores)')
    parser.add_argument('--nlist', type=int, default=256,
                       help='FAISS nlist parameter')
    parser.add_argument('--nprobe', type=int, default=64,
                       help='FAISS nprobe parameter')
    parser.add_argument('--ngpus', type=int, default=2,
                       help='Number of GPUs to use')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for encoding')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate on all datasets')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"CPU cores: {mp.cpu_count()}")
    
    # Determine datasets
    if args.all:
        datasets = ['arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa',
                    'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact',
                    'trec-covid', 'webis-touche2020']
    else:
        datasets = [args.dataset]
    
    # Load model
    logger.info(f"Loading {args.model} model...")
    if args.model == 'constbert':
        model = ConstBERTWrapper(device=device)
    else:
        raise NotImplementedError("ColBERT evaluation not yet implemented")
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING: {dataset_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Load dataset
            beir_dataset = BEIRDataset(dataset_name, args.data_dir)
            beir_dataset.download()
            corpus, queries, qrels = beir_dataset.load()
            
            # Encode corpus (OPTIMIZED)
            start_time = time.time()
            doc_embeddings, doc_ids = encode_beir_corpus(
                model, corpus, args.batch_size, args.num_workers
            )
            encode_time = time.time() - start_time
            logger.info(f"✓ Document encoding: {encode_time:.2f}s ({len(corpus)/encode_time:.0f} docs/sec)")
            
            # Build FAISS index (OPTIMIZED - CPU stable, GPU MaxSim)
            start_time = time.time()
            faiss_index = OptimizedFAISSIndex(use_gpu_maxsim=True)
            faiss_index.build_index(doc_embeddings, doc_ids, args.nlist, args.nprobe)
            index_time = time.time() - start_time
            logger.info(f"✓ Index building: {index_time:.2f}s")
            
            # Encode queries (OPTIMIZED)
            start_time = time.time()
            query_embeddings, query_ids = encode_beir_queries(
                model, queries, args.batch_size
            )
            query_time = time.time() - start_time
            logger.info(f"✓ Query encoding: {query_time:.2f}s")
            
            # Retrieve and evaluate (OPTIMIZED)
            start_time = time.time()
            metrics = retrieve_and_evaluate(
                faiss_index, query_embeddings, query_ids, qrels,
                k=1000, batch_size=args.query_batch_size
            )
            retrieval_time = time.time() - start_time
            logger.info(f"✓ Retrieval: {retrieval_time:.2f}s ({len(queries)/retrieval_time:.1f} queries/sec)")
            
            # Calculate total time
            total_time = encode_time + index_time + query_time + retrieval_time
            logger.info(f"✓ TOTAL TIME: {total_time:.2f}s")
            
            # Store results
            result = {
                'dataset': dataset_name,
                'model': args.model,
                'metrics': metrics,
                'num_docs': len(corpus),
                'num_queries': len(queries),
                'timing': {
                    'encoding_time_s': encode_time,
                    'index_time_s': index_time,
                    'query_time_s': query_time,
                    'retrieval_time_s': retrieval_time,
                    'total_time_s': total_time
                },
                'config': {
                    'batch_size': args.batch_size,
                    'query_batch_size': args.query_batch_size,
                    'num_workers': args.num_workers,
                    'nlist': args.nlist,
                    'nprobe': args.nprobe,
                    'ngpus': args.ngpus,
                    'device': device
                }
            }
            
            all_results[dataset_name] = result
            
            # Print results
            logger.info(f"\n{'='*80}")
            logger.info(f"RESULTS for {dataset_name}:")
            logger.info(f"{'='*80}")
            for metric, value in sorted(metrics.items()):
                logger.info(f"  {metric:.<40} {value:.4f}")
            logger.info(f"{'='*80}\n")
            
            # Save individual result
            output_path = Path(args.output_dir) / f"11_beir_{args.model}_{dataset_name}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved to {output_path}")
            
            # Free memory
            del faiss_index
            del doc_embeddings
            del query_embeddings
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    if args.all:
        summary_path = Path(args.output_dir) / f"11_beir_{args.model}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved summary to {summary_path}")
        
        # Print summary table
        logger.info("\n" + "="*100)
        logger.info("BEIR EVALUATION SUMMARY")
        logger.info("="*100)
        logger.info(f"{'Dataset':<20} {'NDCG@10':>10} {'Recall@100':>12} {'MRR@10':>10} {'Time(s)':>10}")
        logger.info("-"*100)
        for dataset_name, result in all_results.items():
            metrics = result['metrics']
            timing = result['timing']['total_time_s']
            logger.info(f"{dataset_name:<20} {metrics.get('ndcg@10', 0):>10.4f} "
                       f"{metrics.get('recall@100', 0):>12.4f} "
                       f"{metrics.get('mrr@10', 0):>10.4f} "
                       f"{timing:>10.1f}")
        logger.info("="*100)


if __name__ == '__main__':
    main()