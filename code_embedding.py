import numpy as np
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class CodeEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
        self.model = AutoModel.from_pretrained('thenlper/gte-base')

    def generate_embeddings(self, snippets):
        prefix = "query: "
        input_texts = [prefix + snippet for snippet in snippets]
        batch_dict = self.tokenizer(input_texts, max_length=512,
                                     padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1).detach().numpy()

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        This method computes a weighted average of the token embeddings, where padding tokens are ignored.
        This creates a fixed-size representation for variable-length input sequences,
        which is useful for many downstream tasks.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def find_k_nearest_neighbors(query_embedding, embeddings, k=5):
        similarities = np.dot(embeddings, query_embedding.T)
        sorted_indices = similarities.argsort(axis=0)[-k:][::-1].squeeze()
        filtered_indices = [idx for idx in sorted_indices if similarities[idx] >= 0.8]
        return filtered_indices