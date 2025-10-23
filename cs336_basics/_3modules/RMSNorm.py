import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.g = torch.nn.parameter.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape 
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        Remember to upcast your input to torch.float32 before performing the normalization 
        (and later downcast to the original dtype), to prevent overflow when you square the input
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = (x / RMS_x) * self.g
        return result.to(in_dtype)
