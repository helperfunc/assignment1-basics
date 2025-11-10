import torch

class RoPE(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Why use RoPE?
    1. Relative position encoding: RoPE encodes relative distances between tokens rather than 
       absolute positions, making attention scores depend only on token distance (j-i), not 
       their absolute positions i and j.
    2. Length generalization: Because it uses relative positions, the model can handle sequences 
       longer than those seen during training.
    3. Multi-scale position information: Different dimension pairs rotate at different frequencies,
       capturing both fine-grained local patterns (fast-rotating pairs) and coarse-grained 
       long-range dependencies (slow-rotating pairs).
    4. Efficient implementation: No learnable parameters needed; rotation angles are precomputed
       and stored as buffers.
    5. Mathematical elegance: The rotation property R_i^T R_j = R_{j-i} naturally provides 
       relative position information through matrix multiplication.
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers if needed.
        
        theta: float Θ value for the RoPE (typically 10000)
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        
        Implementation details:
        - Precomputes rotation angles θ_{i,k} = i / Θ^{(2k-2)/d_k} for all positions and dimension pairs
        - Stores cos and sin values as non-persistent buffers for efficient lookup
        - Each dimension pair k rotates at a different frequency based on Θ^{(2k-2)/d_k}
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        # pos: Token positions [0, 1, 2, ..., max_seq_len-1]
        # Shape: [max_seq_len]
        pos = torch.arange(max_seq_len, device=device).float()
        
        # dim: Dimension pair indices [0, 1, 2, ..., d_k/2-1], representing k-1 in the formula
        # Shape: [d_k//2]
        # Why d_k//2? Because RoPE treats d_k dimensions as d_k/2 independent 2D rotation pairs
        dim = torch.arange(d_k // 2, device=device)
        
        # angle: Matrix of angles θ_{i,k} = i / Θ^{(2k-2)/d_k}
        # Broadcasting: pos[:, None] becomes [max_seq_len, 1], so angle becomes [max_seq_len, d_k//2]
        # Each (position, dimension_pair) gets its unique rotation angle
        # 
        # Why 2*dim? In the formula θ_{i,k} = i / Θ^{(2k-2)/d_k}, when k ∈ {1, ..., d_k/2},
        # we have dim = k-1 ∈ {0, ..., d_k/2-1}, so 2*dim = 2k-2
        # 
        # Rotation frequency hierarchy:
        # - k=0 (dim=0): θ = i / Θ^0 = i        → Fast rotation (high frequency, ~1 radian/position)
        # - k=1 (dim=1): θ = i / Θ^{2/d_k}      → Medium rotation
        # - Large k:     θ = i / Θ^{large}      → Slow rotation (low frequency)
        # 
        # This creates a multi-scale encoding:
        # - Fast-rotating pairs capture fine-grained local relationships between adjacent tokens
        # - Slow-rotating pairs capture coarse-grained long-range dependencies
        # 2 * (d_k//2-1) = (d_k - 2)/d_k=(2k - 2)/d_k     k = d_k//2
        angle = pos[:, None] / (theta ** (2 * dim / d_k))
        
        # Store precomputed cos/sin values as buffers (not learnable parameters)
        # persistent=False means these won't be saved in state_dict (they can be recomputed)
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Apply RoPE rotation to input tensor.
        
        x: (..., seq_len, d_k) Input tensor (typically query or key vectors)
        token_positions: (..., seq_len) Token positions for each element
        
        Returns: (..., seq_len, d_k) Rotated tensor
        
        How RoPE works:
        1. Split d_k dimensions into d_k/2 pairs: (x0,x1), (x2,x3), ..., (x_{d_k-2}, x_{d_k-1})
        2. Treat each pair as a 2D vector and apply 2D rotation:
           [x_even']   [cosθ  -sinθ] [x_even]
           [x_odd' ] = [sinθ   cosθ] [x_odd ]
        3. Each pair rotates by a different angle θ_{i,k} based on position i and pair index k
        4. Interleave rotated pairs back to original shape
        
        Why this formula?
        - x_rotated_even = x_even * cos - x_odd * sin  (first row of rotation matrix)
        - x_rotated_odd = x_even * sin + x_odd * cos   (second row of rotation matrix)
        
        Key property: When computing attention Q_i^T K_j:
        Q_i^T K_j = (R_i q)^T (R_j k) = q^T R_i^T R_j k = q^T R_{j-i} k
        
        Since rotation matrices satisfy R_i^T R_j = R_{j-i}, the attention score only depends
        on the relative distance (j-i) between tokens, not their absolute positions.
        '''
        orig_shape = x.shape  # (..., seq_len, d_k)
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        assert d_k == self.d_k

        # Split input into even and odd indexed dimensions
        # Example: [a0, a1, a2, a3, a4, a5] → (a0, a2, a4) and (a1, a3, a5)
        # This creates pairs: (a0,a1), (a2,a3), (a4,a5) for independent 2D rotations
        x_even = x[..., ::2]   # (..., seq_len, d_k//2) - dimensions 0, 2, 4, ...
        x_odd = x[..., 1::2]   # (..., seq_len, d_k//2) - dimensions 1, 3, 5, ...

        # Fetch precomputed cos/sin values for the given token positions
        # Uses token_positions to index into [max_seq_len, d_k//2] buffers
        # Result: (..., seq_len, d_k//2) - one cos/sin value per (position, dimension_pair)
        cos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k//2)

        # Apply 2D rotation to each dimension pair
        # This implements the rotation matrix multiplication:
        # [x_even']   [cosθ  -sinθ] [x_even]
        # [x_odd' ] = [sinθ   cosθ] [x_odd ]
        # 
        # Expanding the matrix multiplication:
        # x_even' = x_even * cosθ + x_odd * (-sinθ) = x_even * cos - x_odd * sin
        # x_odd'  = x_even * sinθ + x_odd * cosθ    = x_even * sin + x_odd * cos
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave even and odd dimensions back to original structure
        # Stack creates (..., seq_len, d_k//2, 2), then reshape to (..., seq_len, d_k)
        # This ensures [a0', a1', a2', a3', a4', a5'] maintains the correct pairing order
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)  # (..., seq_len, d_k//2, 2)
        x_rotated = x_rotated.reshape(*orig_shape)  # (..., seq_len, d_k)

        return x_rotated
