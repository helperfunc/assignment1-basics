import torch
from .RoPE import RoPE
from .Linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, rope: RoPE | None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.rope = rope
        self.W_qkv = Linear(num_heads * self.d_k, 3*d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        '''
        x: [..., seq_len, d_model]
        returns : [..., seq_len, d_model]
        '''
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        if mask is None:
            # Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, 
            # the other elements of the result tensor out are set to 0.
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # [..., seq_len, 3*d_model]
        qkv = self.W_qkv(x)
        
        # [..., seq_len, d_model] = [..., seq_len, num_heads*d_k]
        Q, K, V = qkv.chunk(3, dim=-1)

        # [..., seq_len, num_heads, d_k]
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v)

        # [..., num_heads, seq_len, d_k]
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        # apply RoPE to Q and K
        if self.rope is not None:
            if token_positions is None:
                # [0, 1, 2, ..., seq_len - 1]
                token_pos = torch.arange(seq_len, device=x.device)
                # [..., seq_len]
                token_pos = token_pos.view(*([1] * len(batch_size)), seq_len)
                token_pos = token_pos.expand(*batch_shape, seq_len)
            else:
                token_pos = token_positions
            # apply RoPE independently for each head
            # reshape [...*num_heads, seq_len, d_k]
            original_Q_shape = Q.shape
            original_K_shape = K.shape
            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)
            token_pos_flat = token_pos.reshape(-1, seq_len)

            Q_flat = self.rope(Q_flat, token_pos_flat)
            K_flat = self.rope(K_flat, token_pos_flat)

            Q = Q_flat.reshape(original_Q_shape)
            K = K_flat.reshape(original_K_shape)
            
        # [..., num_heads, seq_len, d_v]
        atten_output = scaled_dot_product_attention(Q, K, V, mask)

        # [..., seq_len, num_heads, d_v]
        atten_output = atten_output.transpose(-3, -2).contiguous()
        # [..., seq_len, num_heads * d_v]
        atten_output = atten_output.view(*batch_shape, seq_len, self.num_heads * self.d_v)

        # [..., seq_len, d_model]
        return self.W_o(atten_output)
