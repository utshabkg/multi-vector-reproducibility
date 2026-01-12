"""
TREC Tip-of-the-Tongue (ToT) 2025 Dataset Loader

The ToT dataset tests retrieval on complex, descriptive queries where users
can't remember exact terms. This tests zero-shot generalization of ConstBERT.

Dataset stats:
- Corpus: 6.4M Wikipedia documents (vs 8.8M MS-MARCO passages)
- Test queries: 622 (vs 6,980 MS-MARCO dev)
- Domain: Open-domain Wikipedia (vs MS-MARCO web passages)
"""

import json
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TRECToTDataLoader:
    """Loader for TREC Tip-of-the-Tongue 2025 dataset."""
    
    def __init__(self, data_dir: str = "/media/12TB/shared/datasets/raw/trec-tot-2025"):
        self.data_dir = Path(data_dir)
        self.corpus_file = self.data_dir / "trec-tot-2025-corpus.jsonl"
        self.queries_dir = self.data_dir / "queries"
        self.qrels_dir = self.data_dir / "qrel"
        
        # Verify paths exist
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_file}")
        if not self.queries_dir.exists():
            raise FileNotFoundError(f"Queries dir not found: {self.queries_dir}")
        if not self.qrels_dir.exists():
            raise FileNotFoundError(f"Qrels dir not found: {self.qrels_dir}")
    
    def load_corpus(self, max_docs: int = None) -> Tuple[List[str], List[str]]:
        """
        Load Wikipedia corpus.
        
        Format: {"id": "12", "url": "...", "title": "Anarchism", "text": "..."}
        
        Returns:
            doc_ids: List of document IDs
            passages: List of passage texts (title + text combined)
        """
        logger.info(f"Loading corpus from {self.corpus_file}...")
        
        doc_ids = []
        passages = []
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                
                if i > 0 and i % 100000 == 0:
                    logger.info(f"  Loaded {i:,} documents...")
                
                doc = json.loads(line.strip())
                doc_id = str(doc['id'])
                # Combine title and text for richer representation
                passage = f"{doc['title']}. {doc['text']}"
                
                doc_ids.append(doc_id)
                passages.append(passage)
        
        logger.info(f"Loaded {len(doc_ids):,} documents from corpus")
        return doc_ids, passages
    
    def load_queries(self, split: str = "test") -> Dict[str, str]:
        """
        Load queries for a specific split.
        
        Args:
            split: One of 'train', 'dev1', 'dev2', 'dev3', 'test'
        
        Format: {"query_id": 3000, "query": "I remember this certain type of cheese..."}
        
        Returns:
            Dict mapping query_id -> query_text
        """
        query_file = self.queries_dir / f"{split}-2025-queries.jsonl"
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")
        
        logger.info(f"Loading {split} queries from {query_file}...")
        
        queries = {}
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                q = json.loads(line.strip())
                query_id = str(q['query_id'])
                query_text = q['query']
                queries[query_id] = query_text
        
        logger.info(f"Loaded {len(queries)} queries for {split} split")
        return queries
    
    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, int]]:
        """
        Load relevance judgments for a specific split.
        
        Args:
            split: One of 'train', 'dev1', 'dev2', 'dev3', 'test'
        
        Format: "3000 0 26092459 1" (query_id, iteration, doc_id, relevance)
        
        Returns:
            Dict mapping query_id -> {doc_id: relevance_score}
        """
        qrel_file = self.qrels_dir / f"{split}-2025-qrel.txt"
        if not qrel_file.exists():
            raise FileNotFoundError(f"Qrel file not found: {qrel_file}")
        
        logger.info(f"Loading {split} qrels from {qrel_file}...")
        
        qrels = {}
        with open(qrel_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                
                query_id, _, doc_id, rel = parts
                query_id = str(query_id)
                doc_id = str(doc_id)
                rel = int(rel)
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = rel
        
        logger.info(f"Loaded qrels for {len(qrels)} queries in {split} split")
        return qrels
    
    def get_corpus_statistics(self) -> Dict:
        """Get basic statistics about the corpus."""
        logger.info("Computing corpus statistics...")
        
        total_docs = 0
        total_chars = 0
        sample_docs = []
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_docs += 1
                doc = json.loads(line.strip())
                passage = f"{doc['title']}. {doc['text']}"
                total_chars += len(passage)
                
                if i < 3:
                    sample_docs.append({
                        'id': doc['id'],
                        'title': doc['title'],
                        'length': len(passage)
                    })
        
        avg_length = total_chars / total_docs if total_docs > 0 else 0
        
        stats = {
            'total_documents': total_docs,
            'avg_passage_length': avg_length,
            'total_characters': total_chars,
            'sample_documents': sample_docs
        }
        
        return stats


if __name__ == "__main__":
    # Test the loader
    loader = TRECToTDataLoader()
    
    # Get statistics
    print("\n=== Corpus Statistics ===")
    stats = loader.get_corpus_statistics()
    print(f"Total documents: {stats['total_documents']:,}")
    print(f"Average passage length: {stats['avg_passage_length']:.0f} characters")
    print(f"\nSample documents:")
    for doc in stats['sample_documents']:
        print(f"  ID {doc['id']}: {doc['title']} ({doc['length']:,} chars)")
    
    # Load test queries
    print("\n=== Test Split ===")
    queries = loader.load_queries("test")
    print(f"Number of test queries: {len(queries)}")
    print(f"\nSample query:")
    sample_qid = list(queries.keys())[0]
    print(f"  Query ID: {sample_qid}")
    print(f"  Query: {queries[sample_qid][:200]}...")
    
    # Load test qrels
    qrels = loader.load_qrels("test")
    print(f"\nNumber of queries with relevance judgments: {len(qrels)}")
    if sample_qid in qrels:
        print(f"  Query {sample_qid} has {len(qrels[sample_qid])} relevant documents")
