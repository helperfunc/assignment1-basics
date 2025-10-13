import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device|None = None, dtype: torch.dtype | None = None):
        '''
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.W = torch.nn.parameter.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b = 3 * std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the linear transformation to the input
        '''
        return x @ self.W.t()
