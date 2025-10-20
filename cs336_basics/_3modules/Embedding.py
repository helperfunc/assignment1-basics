import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_{model}
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.embedding_mat = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding_mat, mean=0.0, std=1.0, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs
        '''
        token_ids = token_ids.long()
        return self.embedding_mat[token_ids]