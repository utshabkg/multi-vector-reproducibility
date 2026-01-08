"""
Data loaders for MS-MARCO and TREC Deep Learning Track datasets.
"""
import os
from typing import Dict, List, Tuple, Iterator
import pandas as pd
from pathlib import Path


class MSMARCODataLoader:
    """Load MS-MARCO passage ranking dataset."""
    
    def __init__(self, data_dir: str = "/media/12TB/shared/datasets/raw/msmarco-passage"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_collection(self) -> Dict[str, str]:
        """
        Load the document collection.
        Returns: Dict mapping passage_id -> passage_text
        """
        collection_path = self.data_dir / "collection.tsv"
        print(f"Loading collection from {collection_path}...")
        
        collection = {}
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                pid, passage = line.rstrip('\n').split('\t')
                collection[pid] = passage
        
        print(f"Loaded {len(collection)} passages")
        return collection
    
    def load_queries(self, split: str = "dev") -> Dict[str, str]:
        """
        Load queries for a given split.
        Args:
            split: 'dev' or 'train'
        Returns: Dict mapping query_id -> query_text
        """
        if split == "dev":
            queries_path = self.data_dir / "queries.dev.small.tsv"
        elif split == "train":
            queries_path = self.data_dir / "queries.train.tsv"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Loading {split} queries from {queries_path}...")
        
        queries = {}
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, query = line.rstrip('\n').split('\t')
                queries[qid] = query
        
        print(f"Loaded {len(queries)} queries")
        return queries
    
    def load_qrels(self, split: str = "dev") -> Dict[str, Dict[str, int]]:
        """
        Load relevance judgments.
        Args:
            split: 'dev' or 'train'
        Returns: Dict mapping query_id -> {passage_id -> relevance}
        """
        if split == "dev":
            qrels_path = self.data_dir / "qrels.dev.small.tsv"
        elif split == "train":
            qrels_path = self.data_dir / "qrels.train.tsv"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Loading {split} qrels from {qrels_path}...")
        
        qrels = {}
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 4:
                    qid, _, pid, rel = parts
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = int(rel)
        
        print(f"Loaded qrels for {len(qrels)} queries")
        return qrels
    
    def iterate_collection(self, batch_size: int = 1000) -> Iterator[List[Tuple[str, str]]]:
        """
        Iterate over collection in batches.
        Yields: List of (passage_id, passage_text) tuples
        """
        collection_path = self.data_dir / "collection.tsv"
        
        batch = []
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                pid, passage = line.rstrip('\n').split('\t')
                batch.append((pid, passage))
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        if batch:
            yield batch


class TRECDLDataLoader:
    """Load TREC Deep Learning Track queries and qrels."""
    
    def __init__(self, data_dir: str = "/media/12TB/shared/datasets/raw/msmarco-passage"):
        self.data_dir = Path(data_dir)
    
    def load_queries(self, year: int = 2019) -> Dict[str, str]:
        """
        Load TREC DL queries for a given year.
        Args:
            year: 2019 or 2020
        Returns: Dict mapping query_id -> query_text
        """
        # Try different potential locations
        potential_paths = [
            self.data_dir / f"queries.dl{year}.tsv",
            self.data_dir / f"trec{year}-queries.tsv",
            self.data_dir / "trec" / f"{year}-queries.tsv",
        ]
        
        queries_path = None
        for path in potential_paths:
            if path.exists():
                queries_path = path
                break
        
        if queries_path is None:
            raise FileNotFoundError(
                f"TREC DL {year} queries not found. Tried: {potential_paths}"
            )
        
        print(f"Loading TREC DL {year} queries from {queries_path}...")
        
        queries = {}
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, query = line.rstrip('\n').split('\t')
                queries[qid] = query
        
        print(f"Loaded {len(queries)} TREC DL {year} queries")
        return queries
    
    def load_qrels(self, year: int = 2019) -> Dict[str, Dict[str, int]]:
        """
        Load TREC DL qrels for a given year.
        Args:
            year: 2019 or 2020
        Returns: Dict mapping query_id -> {passage_id -> relevance}
        """
        # Try different potential locations
        potential_paths = [
            self.data_dir / f"qrels.dl{year}.tsv",
            self.data_dir / f"trec{year}-qrels.tsv",
            self.data_dir / "trec" / f"{year}-qrels.tsv",
        ]
        
        qrels_path = None
        for path in potential_paths:
            if path.exists():
                qrels_path = path
                break
        
        if qrels_path is None:
            print(f"WARNING: TREC DL {year} qrels not found. Tried: {potential_paths}")
            print("You may need to download them from TREC website.")
            return {}
        
        print(f"Loading TREC DL {year} qrels from {qrels_path}...")
        
        qrels = {}
        with open(qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split()
                if len(parts) >= 4:
                    qid, _, pid, rel = parts[0], parts[1], parts[2], parts[3]
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = int(rel)
        
        print(f"Loaded qrels for {len(qrels)} queries")
        return qrels


if __name__ == "__main__":
    # Test data loading
    print("Testing MS-MARCO data loader...")
    loader = MSMARCODataLoader()
    
    # Load a sample of data
    queries = loader.load_queries("dev")
    print(f"Sample query: {list(queries.items())[0]}")
    
    qrels = loader.load_qrels("dev")
    print(f"Sample qrel: {list(qrels.items())[0]}")
    
    # Test batch iteration
    print("\nTesting batch iteration...")
    for i, batch in enumerate(loader.iterate_collection(batch_size=100)):
        print(f"Batch {i}: {len(batch)} passages")
        if i >= 2:  # Just show first few batches
            break
    
    print("\nData loading test complete!")
