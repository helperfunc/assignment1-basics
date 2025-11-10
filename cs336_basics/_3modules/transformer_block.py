import torch
from .RMSNorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU
from .RoPE import RoPE

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None, device=None, dtype=None):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multihead_self_attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.point_wise_ff = SwiGLU(d_model, d_ff)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        '''
        x: (..., seq_len, d_model) Input tensor
        mask: Optional attention mask
        Returns: (..., seq_len, d_model) Output tensor
        '''
        atten_output = self.multihead_self_attn(self.rms_norm_1(x), mask=mask)
        atten_res_output = x + atten_output
        ff_output = self.point_wise_ff(self.rms_norm_2(atten_res_output))
        output = atten_res_output + ff_output
        return output