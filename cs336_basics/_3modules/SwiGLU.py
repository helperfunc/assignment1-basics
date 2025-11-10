import torch
from .GLU import GLU
from .Linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff_raw = int(8.0/3.0 * d_model)
            d_ff = ((d_ff_raw + 63) // 64) * 64
        self.lin = Linear(d_ff, d_model)
        self.glu = GLU(d_model, d_ff)
        
    def forward(self, x):
        return self.lin.forward(self.glu(x))