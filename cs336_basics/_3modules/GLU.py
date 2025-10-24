import torch
from .SiLU import SiLU
from .Linear import Linear

class GLU(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(8.0/3.0) * d_model
        self.lin1 = Linear(d_model, d_ff)
        self.lin2 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor):
        W1x = self.lin1.forward(x)
        W2x = self.lin2.forward(x)
        return torch.mul(SiLU(W1x), W2x)