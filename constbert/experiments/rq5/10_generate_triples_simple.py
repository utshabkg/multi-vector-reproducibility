#!/usr/bin/env python3
"""
Simple triple generation using rank_bm25 library (pure Python, no Java dependencies)
Generates training triples in format: qid\tpos_docid\tneg_docid
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    print("rank_bm25 not available. Install with: pip install rank-bm25")


def load_corpus(corpus_file, max_docs=None):
    """Load corpus documents"""
    docs = []
    doc_ids = []
    
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file) as f:
        for i, line in enumerate(tqdm(f, desc="Loading corpus")):
            if max_docs and i >= max_docs:
                break
            doc = json.loads(line)
            doc_id = str(doc['id'])
            title = doc.get('title', '')
            text = doc.get('text', '')
            content = f"{title} {text}".strip()
            
            docs.append(content)
            doc_ids.append(doc_id)
    
    return docs, doc_ids


def load_queries(queries_file):
    """Load queries from JSONL"""
    queries = {}
    with open(queries_file) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q.get('query_id') or q.get('id'))
            text = q.get('query') or q.get('text')
            queries[qid] = text
    return queries


def load_qrels(qrels_file):
    """Load qrels"""
    qrels = defaultdict(set)
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels[qid].add(docid)
    return qrels


def tokenize(text):
    """Simple tokenization"""
    return text.lower().split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', required=True)
    parser.add_argument('--queries_file', required=True)
    parser.add_argument('--qrels_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--topk', type=int, default=100, help='BM25 top-k candidates')
    parser.add_argument('--negs_per_query', type=int, default=8, help='Negatives per positive')
    parser.add_argument('--max_corpus_docs', type=int, default=None, help='Limit corpus size for testing')
    args = parser.parse_args()
    
    if not RANK_BM25_AVAILABLE:
        raise SystemExit("rank_bm25 is required. Install with: pip install rank-bm25")
    
    # Create output directory
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    queries = load_queries(args.queries_file)
    qrels = load_qrels(args.qrels_file)
    
    print(f"Loaded {len(queries)} queries, {len(qrels)} queries with relevance judgments")
    
    # Load corpus
    corpus_docs, corpus_doc_ids = load_corpus(args.corpus_file, args.max_corpus_docs)
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_doc_ids)}
    
    print(f"Loaded {len(corpus_docs)} documents")
    
    # Build BM25 index
    print("Tokenizing corpus...")
    tokenized_corpus = [tokenize(doc) for doc in tqdm(corpus_docs, desc="Tokenizing")]
    
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Generate triples
    print("Generating triples...")
    written = 0
    
    with open(args.output_file, 'w') as f:
        for qid in tqdm(queries, desc="Processing queries"):
            if qid not in qrels or not qrels[qid]:
                continue
            
            query_text = queries[qid]
            tokenized_query = tokenize(query_text)
            
            # Get BM25 scores
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k candidates
            top_indices = np.argsort(scores)[::-1][:args.topk]
            
            # Find negatives (not in qrels)
            negatives = []
            for idx in top_indices:
                doc_id = corpus_doc_ids[idx]
                if doc_id not in qrels[qid]:
                    negatives.append(doc_id)
                    if len(negatives) >= args.negs_per_query:
                        break
            
            if not negatives:
                continue
            
            # Write triples: one line per (query, positive, negative)
            for pos_doc_id in qrels[qid]:
                # Check if positive doc is in our corpus
                if pos_doc_id not in doc_id_to_idx:
                    continue
                
                for neg_doc_id in negatives:
                    f.write(f"{qid}\t{pos_doc_id}\t{neg_doc_id}\n")
                    written += 1
    
    print(f"\n✅ Wrote {written:,} triples to {args.output_file}")
    print(f"   Queries processed: {len([q for q in queries if q in qrels])}")
    print(f"   Average triples per query: {written / max(1, len([q for q in queries if q in qrels])):.1f}")


if __name__ == "__main__":
    main()
