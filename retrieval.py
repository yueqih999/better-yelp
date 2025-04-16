import torch
import faiss
import numpy as np
from typing import Dict, List, Tuple


class FaissRetriever:

    def __init__(self, hidden_dim: int = 128):
        """Initialize Faiss retriever
        
        Args:
            hidden_dim: Dimension of embeddings
        """
        self.hidden_dim = hidden_dim
        # Initialize Faiss index
        self.index = faiss.IndexFlatIP(hidden_dim)  # Inner product similarity
        self.business_ids = []  # Store business IDs in order
        self.business_info = {}  # Store business info
        

    def add_businesses(self, business_embeddings: torch.Tensor, business_ids: List[str], business_info: Dict):
        """Add business embeddings to the index
        
        Args:
            business_embeddings: Business embeddings tensor
            business_ids: List of business IDs corresponding to embeddings
            business_info: Dictionary of business information
        """
        # Convert to numpy and normalize
        embeddings = business_embeddings.detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(embeddings)  # In-place L2 normalization
        
        # Add to index
        self.index.add(embeddings)
        self.business_ids.extend(business_ids)
        self.business_info.update(business_info)
        

    def search(self, user_embedding: torch.Tensor, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for nearest businesses for a user embedding
        
        Args:
            user_embedding: User embedding tensor
            k: Number of recommendations to return
            
        Returns:
            List of tuples (business_id, similarity_score, business_info)
        """
        query = user_embedding.detach().cpu().numpy().astype('float32')
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                business_id = self.business_ids[idx]
                info = self.business_info.get(business_id, {})
                results.append((business_id, float(score), info))
                
        return results
    
    
    def batch_search(self, user_embeddings: torch.Tensor, k: int = 10) -> List[List[Tuple[str, float, Dict]]]:
        """Batch search for nearest businesses for multiple user embeddings
        
        Args:
            user_embeddings: Batch of user embedding tensors
            k: Number of recommendations per user
            
        Returns:
            List of recommendation lists, one per user
        """
        queries = user_embeddings.detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(queries)

        scores, indices = self.index.search(queries, k)
        
        all_results = []
        for user_scores, user_indices in zip(scores, indices):
            user_results = []
            for score, idx in zip(user_scores, user_indices):
                if idx != -1:  # Valid index
                    business_id = self.business_ids[idx]
                    info = self.business_info.get(business_id, {})
                    user_results.append((business_id, float(score), info))
            all_results.append(user_results)
                
        return all_results 