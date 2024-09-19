import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class CodeEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
        self.model = AutoModel.from_pretrained('thenlper/gte-base')
        self.snippets = []

    
    def generate_embeddings(self, snippet: str) -> list[float]:
        inputs = self.tokenizer(snippet, return_tensors='pt',max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        embeddings = CodeEmbedding.average_pool(outputs.last_hidden_state, attention_mask)
        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy().tolist()[0]                       # embeddings.tolist() = [[0.1, 0.2, 0.3, ...]] , hence we take 0th element

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        This method computes a weighted average of the token embeddings, where padding tokens are ignored.
        This creates a fixed-size representation for variable-length input sequences,
        which is useful for many downstream tasks.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Write a method to find the k nearest neighbors of a query embedding,
    # using the cosine similarity between the query embedding and the embeddings in a list
    @staticmethod
    def find_k_nearest_neighbors(query_embedding, embeddings, top_n=3):
        # Convert query_embedding to numpy array if it's not already
        """Search for most similar texts based on cosine similarity."""
        similarities = [CodeEmbedding.cosine_similarity(query_embedding, emb) for emb in embeddings]
        # Create a list of tuples (index, similarity)
        indexed_similarities = list(enumerate(similarities))
        # Sort by similarity in descending order
        sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
        # Return the top n indices
        return [index for index, _ in sorted_similarities[:top_n]]
    
    @staticmethod
    def cosine_similarity(v1, v2):
        # Normalize the vectors
        # Convert to numpy arrays if they're lists
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        
        # Check if shapes are the same and 1-dimensional
        if v1_np.shape == v2_np.shape and v1_np.ndim == 1:
            dot_product = np.dot(v1_np, v2_np)
        else:
            # If shapes differ or are not 1-dimensional, use matrix multiplication
            dot_product = np.dot(v1_np, v2_np.T)
        
        # Convert back to a Python scalar
        dot_product = dot_product.item()
        magnitude_A = np.linalg.norm(v1)
        magnitude_B = np.linalg.norm(v2)
        # Calculate and return the dot product of normalized vectors
        return dot_product / (magnitude_A * magnitude_B)
