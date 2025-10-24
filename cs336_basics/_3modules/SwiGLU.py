import torch
from .GLU import GLU
from .Linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(8.0/3.0) * d_model
        self.lin = Linear(d_ff, d_model)
        self.glu = GLU(d_model)
        
    def forward(self, x):
        return self.lin.forward(self.glu(x))