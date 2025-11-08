import torch

def softmax(x: torch.Tensor, dim: int):
    val_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - val_max
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    