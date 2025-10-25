import torch

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pos = torch.arange(max_seq_len, device=device).float() # [0, max_seq_len - 1]
        dim = torch.arange(d_k // 2, device=device) # [0, d_k//2-1]
        # angle θ_{i,k} =i/Θ^{(2k−2)/d} for k ∈ {1, . . . , d/2}  
        # 2k-2 [0, ..., (d-2)/d]
        # pos[:, None] [max_seq_len] -> [max_seq_len, 1]
        angle = pos[:, None] / (theta ** (2 * dim / d_k)) # [0/d_k, ..., (d_k-2)/d_k]
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        '''
        orig_shape = x.shape # (..., seq_len, d_k)
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        assert d_k == self.d_k

        # token embedding [a0, a1, a2, a3, a4, a5]
        # RoPE rotate the 2 dimention vector 0/1, 2/3, 4/5
        # q = (x_even, x_odd) groups (a0, a1) (a2, a3) (a4, a5)
        x_even = x[..., ::2] # (..., seq_len, d_k//2) 0, 2, 4, ...
        x_odd = x[..., 1::2] # (..., seq_len, d_k//2) 1, 3, 5, ...

        cos = self.cos[token_positions] # (..., seq_len, d_k//2)
        sin = self.sin[token_positions] # (..., seq_len, d_k//2)

        # q' = Rq, R = 
        # [ cosθ  -sinθ ]
        # [ sinθ   cosθ ]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1) # (..., seq_len, d_k//2, 2)
        x_rotated = x_rotated.reshape(*orig_shape)

        return x_rotated



        