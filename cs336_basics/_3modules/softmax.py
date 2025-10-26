import torch

def softmax(x: torch.Tensor, i: int):
    val_max = torch.max(x, dim=-1, keepdim=True).values
    x = x - val_max
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    