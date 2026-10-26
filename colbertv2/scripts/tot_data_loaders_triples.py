#!/usr/bin/env python3
"""
TREC TOT Dataset with Pre-Computed BM25 Triples
Adapted from msmarco_data_loaders_triples.py to use BM25-based triples for ToT fine-tuning.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class TOTTriplesDataset(Dataset):
    """
    Dataset that loads PRE-COMPUTED query-positive-negative triples for ToT.
    
    Format: Each line is "qid \t pos_pid \t neg_pid"
    
    The triples file contains BM25-sampled (query, pos, neg) tuples.
    """
    
    def __init__(
        self,
        triples_file: str,
        corpus_file: str,
        queries_file: str,
        max_samples: int = None
    ):
        """
        Args:
            triples_file: Path to train.triples (qid \t pos_pid \t neg_pid)
            corpus_file: Path to corpus JSONL file
            queries_file: Path to queries JSONL file
            max_samples: Optional limit on number of triples (for testing)
        """
        self.triples_file = Path(triples_file)
        self.corpus_file = Path(corpus_file)
        self.queries_file = Path(queries_file)
        self.max_samples = max_samples
        
        if not self.triples_file.exists():
            raise FileNotFoundError(f"Triples file not found: {self.triples_file}")
        
        print(f"Loading ToT triples from: {self.triples_file}")
        self._load_corpus()
        self._load_queries()
        self._load_triples()
    
    def _load_corpus(self):
        """Load corpus into memory for fast lookup"""
        import json
        self.corpus = {}
        
        print(f"Loading corpus from: {self.corpus_file}")
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                doc_id = str(doc['id'])
                # Combine title and text
                text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                self.corpus[doc_id] = text
                
                if (i + 1) % 1_000_000 == 0:
                    print(f"  Loaded {i + 1:,} documents...")
        
        print(f"✅ Loaded {len(self.corpus):,} documents")
    
    def _load_queries(self):
        """Load queries from JSONL or TSV file"""
        import json
        self.queries = {}
        
        print(f"Loading queries from: {self.queries_file}")
        
        # Try JSONL format first
        if self.queries_file.suffix == '.jsonl':
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    q = json.loads(line)
                    # Handle different key formats
                    qid = q.get('query_id') or q.get('id') or q.get('qid')
                    text = q.get('query') or q.get('text') or q.get('query_text')
                    if qid and text:
                        self.queries[str(qid)] = text
        else:
            # Try TSV format
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], '\t'.join(parts[1:])
                        self.queries[qid] = qtext
        
        print(f"✅ Loaded {len(self.queries):,} queries")
    
    def _load_triples(self):
        """Load triples (qid, pos_pid, neg_pid) and convert to text"""
        self.triples = []
        skipped = 0
        
        with open(self.triples_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    skipped += 1
                    continue
                
                qid, pos_pid, neg_pid = parts
                
                # Look up text
                query_text = self.queries.get(qid)
                pos_text = self.corpus.get(pos_pid)
                neg_text = self.corpus.get(neg_pid)
                
                # Skip if any text is missing
                if not query_text or not pos_text or not neg_text:
                    skipped += 1
                    continue
                
                self.triples.append((query_text, pos_text, neg_text))
                
                if (i + 1) % 1000 == 0:
                    print(f"  Loaded {i + 1:,} triples...")
        
        print(f"✅ Loaded {len(self.triples):,} triples (skipped {skipped} with missing data)")
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Returns:
            query (str): Query text
            positive (str): Positive document text
            negative (str): Negative document text
        """
        return self.triples[idx]
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.
        Matches author's approach: duplicate queries, concatenate documents.
        
        Returns:
            Dictionary with:
              - 'query': duplicated queries [q1, q2, ..., q1, q2, ...]
              - 'documents': concatenated [pos1, pos2, ..., neg1, neg2, ...]
              - 'batch_size': original batch size (before duplication)
        """
        queries, positives, negatives = zip(*batch)
        
        # Duplicate queries (author's approach)
        duplicated_queries = list(queries) + list(queries)
        
        # Concatenate positives and negatives
        documents = list(positives) + list(negatives)
        
        return {
            'query': duplicated_queries,
            'documents': documents,
            'batch_size': len(queries)
        }


class TOTTriplesDatasetValidation(Dataset):
    """
    Validation dataset using query-positive pairs from qrels.
    Negatives are sampled randomly from corpus.
    """
    
    def __init__(
        self,
        queries_file: str,
        qrels_file: str,
        corpus_file: str,
        max_samples: int = 200,
        negatives_per_query: int = 1
    ):
        """
        Args:
            queries_file: Path to dev queries JSONL
            qrels_file: Path to dev qrels txt
            corpus_file: Path to corpus JSONL
            max_samples: Number of validation queries to use
            negatives_per_query: Number of random negatives per query
        """
        self.queries_file = Path(queries_file)
        self.qrels_file = Path(qrels_file)
        self.corpus_file = Path(corpus_file)
        self.max_samples = max_samples
        self.negatives_per_query = negatives_per_query
        
        print(f"Loading validation data...")
        self._load_corpus_sample()
        self._load_queries()
        self._load_qrels()
        self._create_validation_pairs()
    
    def _load_corpus_sample(self):
        """Load a sample of corpus for negative sampling"""
        import json
        self.corpus = {}
        self.corpus_ids = []
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100000:  # Sample first 100K docs
                    break
                doc = json.loads(line)
                doc_id = str(doc['id'])
                text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                self.corpus[doc_id] = text
                self.corpus_ids.append(doc_id)
        
        print(f"  Loaded {len(self.corpus):,} documents for validation")
    
    def _load_queries(self):
        """Load queries from JSONL or TSV"""
        import json
        self.queries = {}
        
        if self.queries_file.suffix == '.jsonl':
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    q = json.loads(line)
                    # Handle different key formats
                    qid = q.get('query_id') or q.get('id') or q.get('qid')
                    text = q.get('query') or q.get('text') or q.get('query_text')
                    if qid and text:
                        self.queries[str(qid)] = text
        else:
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.queries[parts[0]] = '\t'.join(parts[1:])
        
        print(f"  Loaded {len(self.queries):,} validation queries")
    
    def _load_qrels(self):
        """Load qrels"""
        self.qrels = {}
        
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    qid, _, doc_id = parts[0], parts[1], parts[2]
                    if qid not in self.qrels:
                        self.qrels[qid] = []
                    self.qrels[qid].append(doc_id)
        
        print(f"  Loaded qrels for {len(self.qrels):,} queries")
    
    def _create_validation_pairs(self):
        """Create (query, pos, neg) validation pairs"""
        self.validation_pairs = []
        
        for qid, query_text in list(self.queries.items())[:self.max_samples]:
            if qid not in self.qrels:
                continue
            
            relevant_docs = self.qrels[qid]
            
            # Sample negative
            for _ in range(self.negatives_per_query):
                neg_id = random.choice(self.corpus_ids)
                while neg_id in relevant_docs:
                    neg_id = random.choice(self.corpus_ids)
                
                neg_text = self.corpus.get(neg_id, "")
                
                # Use first relevant doc as positive
                pos_id = relevant_docs[0]
                # Load pos text from corpus if needed
                import json
                pos_text = ""
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        doc = json.loads(line)
                        if str(doc['id']) == pos_id:
                            pos_text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                            break
                
                if pos_text and neg_text:
                    self.validation_pairs.append((query_text, pos_text, neg_text))
        
        print(f"  Created {len(self.validation_pairs):,} validation pairs")
    
    def __len__(self):
        return len(self.validation_pairs)
    
    def __getitem__(self, idx):
        return self.validation_pairs[idx]
    
    @staticmethod
    def collate_fn(batch):
        """Same collate function as training dataset"""
        queries, positives, negatives = zip(*batch)
        duplicated_queries = list(queries) + list(queries)
        documents = list(positives) + list(negatives)
        return {
            'query': duplicated_queries,
            'documents': documents,
            'batch_size': len(queries)
        }
