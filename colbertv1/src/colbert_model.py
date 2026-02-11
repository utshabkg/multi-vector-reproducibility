"""
ColBERT Model Implementation (from scratch)
Based on: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (SIGIR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class ColBERT(nn.Module):
    """
    ColBERT: Late interaction model for passage retrieval
    
    Architecture:
    - Shared BERT encoder for queries and documents
    - Linear projection: 768 -> embedding_dim (default 128)
    - Query encoder: adds [Q] token, pads to Nq tokens
    - Doc encoder: adds [D] token, uses all tokens
    - Scoring: MaxSim(Q, D) = sum over q in Q of max over d in D of (q · d)
    """
    
    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        embedding_dim: int = 128,
        query_maxlen: int = 32,  # Nq from paper
        doc_maxlen: int = 180,   # From paper (MS MARCO passages)
        mask_punctuation: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        # BERT encoder (shared for queries and documents)
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # Linear projection layer
        self.linear = nn.Linear(self.bert.config.hidden_size, embedding_dim, bias=False)
        
        # Config
        self.embedding_dim = embedding_dim
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.mask_punctuation = mask_punctuation
        self.device = device
        
        # Use BERT's pre-trained unused tokens for [Q] and [D] markers
        # [unused0] = token ID 1 for queries, [unused1] = token ID 2 for documents
        self.Q_marker_token_id = 1  # [unused0]
        self.D_marker_token_id = 2  # [unused1]
        self.mask_token_id = self.tokenizer.mask_token_id
        
        self.to(device)
        
        # Build punctuation skiplist (author's approach)
        # Include both the symbol and its tokenized ID
        self.skiplist = {}
        if self.mask_punctuation:
            import string
            for symbol in string.punctuation:
                self.skiplist[symbol] = True
                # Get tokenized IDs for the symbol
                token_ids = self.tokenizer.encode(symbol, add_special_tokens=False)
                for tid in token_ids:
                    self.skiplist[tid] = True
    
    def _create_doc_mask(self, input_ids):
        """Create mask for document tokens (filter padding and punctuation)"""
        # Author's approach: filter both padding (0) and punctuation from skiplist
        mask = []
        for doc in input_ids.cpu().tolist():
            doc_mask = [(token_id not in self.skiplist) and (token_id != 0) 
                       for token_id in doc]
            mask.append(doc_mask)
        return mask
    
    def query(self, queries, return_mask=False):
        """
        Encode queries with [Q] marker and [MASK] padding
        
        Args:
            queries: List of query strings or pre-tokenized dict
            return_mask: Whether to return attention mask
            
        Returns:
            Q: (batch_size, query_maxlen, embedding_dim) normalized embeddings
            mask: (batch_size, query_maxlen) attention mask (if return_mask=True)
        """
        # Tokenize if input is strings
        if isinstance(queries, list):
            # Author's approach: add placeholder '.' then replace position 1 with [Q] marker
            queries_with_placeholder = ['. ' + q for q in queries]
            
            encoding = self.tokenizer(
                queries_with_placeholder,
                padding='max_length',
                truncation=True,
                max_length=self.query_maxlen,
                return_tensors='pt'
            )
        else:
            encoding = queries
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Replace position 1 with [Q] marker (author's approach)
        input_ids[:, 1] = self.Q_marker_token_id
        
        # Replace padding tokens with [MASK] for query augmentation
        input_ids[input_ids == self.tokenizer.pad_token_id] = self.mask_token_id
        
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Linear projection
        embeddings = self.linear(hidden_states)  # (batch, seq_len, embedding_dim)
        
        # L2 normalization (crucial for MaxSim)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        if return_mask:
            return embeddings, attention_mask
        return embeddings
    
    def doc(self, documents, return_mask=False):
        """
        Encode documents with [D] marker
        
        Args:
            documents: List of document strings or pre-tokenized dict
            return_mask: Whether to return attention mask
            
        Returns:
            D: (batch_size, doc_maxlen, embedding_dim) normalized embeddings
            mask: (batch_size, doc_maxlen) attention mask (if return_mask=True)
        """
        # Tokenize if input is strings
        if isinstance(documents, list):
            # Author's approach: add placeholder '.' then replace position 1 with [D] marker
            docs_with_placeholder = ['. ' + d for d in documents]
            
            # Use 'longest' padding for efficiency (author's approach)
            encoding = self.tokenizer(
                docs_with_placeholder,
                padding='longest',
                truncation=True,
                max_length=self.doc_maxlen,
                return_tensors='pt'
            )
        else:
            encoding = documents
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Replace position 1 with [D] marker (author's approach)
        input_ids[:, 1] = self.D_marker_token_id
        
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Linear projection
        embeddings = self.linear(hidden_states)
        
        # Apply punctuation mask BEFORE normalization (author's approach)
        if self.mask_punctuation:
            mask = torch.tensor(self._create_doc_mask(input_ids), 
                              device=self.device, dtype=torch.float32).unsqueeze(2)
            embeddings = embeddings * mask
        
        # L2 normalization (after masking)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        if return_mask:
            return embeddings, attention_mask
        return embeddings
    
    def score(self, Q, D, Q_mask=None, D_mask=None):
        """
        Compute MaxSim score between query and document embeddings
        
        MaxSim(Q, D) = sum_{q in Q} max_{d in D} (q · d)
        
        Args:
            Q: (batch_size, query_maxlen, embedding_dim)
            D: (batch_size, doc_maxlen, embedding_dim) or (num_docs, doc_maxlen, embedding_dim)
            Q_mask: (batch_size, query_maxlen) optional attention mask for queries
            D_mask: (batch_size, doc_maxlen) or (num_docs, doc_maxlen) optional mask for docs
            
        Returns:
            scores: (batch_size,) or (batch_size, num_docs) MaxSim scores
        """
        # Compute similarity matrix: Q @ D^T
        # Q: (batch, Nq, dim), D: (batch, Nd, dim)
        # scores: (batch, Nq, Nd)
        
        if len(D.shape) == 2:
            # Single document case
            D = D.unsqueeze(0)
        
        # Handle broadcasting for multiple documents
        if Q.shape[0] == 1 and D.shape[0] > 1:
            Q = Q.expand(D.shape[0], -1, -1)
        elif D.shape[0] == 1 and Q.shape[0] > 1:
            D = D.expand(Q.shape[0], -1, -1)
        
        # Compute dot products: (batch, Nq, Nd)
        similarity_matrix = torch.bmm(Q, D.transpose(1, 2))
        
        # Apply document mask if provided
        if D_mask is not None:
            if len(D_mask.shape) == 2:
                D_mask = D_mask.unsqueeze(1)  # (batch, 1, Nd)
            similarity_matrix = similarity_matrix.masked_fill(~D_mask.bool(), float('-inf'))
        
        # MaxSim: max over document dimension
        max_similarities, _ = similarity_matrix.max(dim=2)  # (batch, Nq)
        
        # Apply query mask if provided
        if Q_mask is not None:
            max_similarities = max_similarities * Q_mask.float()
        
        # Sum over query dimension
        scores = max_similarities.sum(dim=1)  # (batch,)
        
        return scores
    
    def forward(self, queries, documents):
        """
        Forward pass: encode queries and documents, compute scores
        
        Args:
            queries: List of query strings
            documents: List of document strings
            
        Returns:
            scores: (batch_size,) MaxSim scores
        """
        Q, Q_mask = self.query(queries, return_mask=True)
        D, D_mask = self.doc(documents, return_mask=True)
        scores = self.score(Q, D, Q_mask, D_mask)
        return scores


def test_colbert():
    """Test ColBERT implementation"""
    print("="*60)
    print("Testing ColBERT Implementation")
    print("="*60)
    
    # Initialize model
    print("\n1. Initializing ColBERT model...")
    model = ColBERT(embedding_dim=128, query_maxlen=32, doc_maxlen=180)
    print(f"✓ Model initialized on {model.device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Embedding dim: {model.embedding_dim}")
    
    # Test query encoding
    print("\n2. Testing query encoding...")
    queries = [
        "What is information retrieval?",
        "How does BERT work?"
    ]
    Q = model.query(queries)
    print(f"  Query embeddings shape: {Q.shape}")
    print(f"  Expected: (2, 32, 128)")
    assert Q.shape == (2, 32, 128), f"Wrong shape: {Q.shape}"
    print(f"  ✓ Query encoding works")
    
    # Test document encoding
    print("\n3. Testing document encoding...")
    documents = [
        "Information retrieval is the process of obtaining information system resources that are relevant to an information need.",
        "BERT is a transformer-based machine learning technique for natural language processing pre-training."
    ]
    D = model.doc(documents)
    print(f"  Document embeddings shape: {D.shape}")
    print(f"  Expected: (2, 180, 128)")
    assert D.shape == (2, 180, 128), f"Wrong shape: {D.shape}"
    print(f"  ✓ Document encoding works")
    
    # Test scoring
    print("\n4. Testing MaxSim scoring...")
    scores = model.score(Q, D)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Scores: {scores}")
    print(f"  Expected: (2,) with positive values")
    assert scores.shape == (2,), f"Wrong shape: {scores.shape}"
    print(f"  ✓ Scoring works")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    scores = model(queries, documents)
    print(f"  Scores: {scores}")
    print(f"  ✓ Forward pass works")
    
    print("\n" + "="*60)
    print("ColBERT Implementation Test: PASSED ✓")
    print("="*60)
    print("\nNext steps:")
    print("1. Implement training loop")
    print("2. Test on TOT dataset (143 train queries)")
    print("3. Implement indexing and retrieval")


if __name__ == "__main__":
    test_colbert()
