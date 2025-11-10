import torch
from .Embedding import Embedding
from .transformer_block import TransformerBlock
from .RMSNorm import RMSNorm
from .Linear import Linear
from .RoPE import RoPE
from .softmax import softmax

class Transformer_lm(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, theta: float = 10000.0, device=None, dtype=None):
        '''
        Transformer Language Model
        
        d_model: int Hidden dimension of the model
        num_heads: int Number of attention heads
        d_ff: int Dimensionality of the feed-forward inner layer
        vocab_size: int Size of the vocabulary
        context_length: int Maximum sequence length (context window)
        num_layers: int Number of Transformer blocks
        theta: float RoPE theta parameter (default 10000)
        device: torch.device | None Device to store parameters on
        dtype: torch.dtype | None Data type of parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        d_k = d_model // num_heads
        self.rope = RoPE(theta=theta, d_k=d_k, max_seq_len=context_length, device=device)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, apply_softmax: bool) -> torch.Tensor:
        '''
        input_ids: (..., seq_len) Token indices
        apply_softmax: bool whether to apply softmax to logits

        returns:
        If apply_softmax=False: (..., seq_len, vocab_size) Logits
        If apply_softmax=True: (..., seq_len, vocab_size) Probabilities
        '''
        # (..., seq_len) -> (..., seq_len, d_model)
        x = self.embedding(input_ids)

        for layer in self.layers:
            # mask is handled inside TransformerBlock (causal)
            x = layer(x, mask=None)
        
        x = self.final_norm(x)

        logits = self.lm_head(x)

        if apply_softmax:
            return softmax(logits, dim=-1)
        else:
            return logits