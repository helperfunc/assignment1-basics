from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # get state associated with p
                t = state.get("t", 0) # get iteration number from the state 
                grad = p.grad.data # the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
    
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e3)

for t in range(10):
    opt.zero_grad() # reset the gradients for all learnable parameters
    loss = (weights**2).mean() # compute a scalar loss value
    print(loss.cpu().item())
    loss.backward() # run backward pass, which computes gradients
    opt.step() # run optimizer step

