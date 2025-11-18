import math
import torch

class AdamW(torch.optim.Optimizer):
    '''
    Simple Adam optmizer implmentaion (decoupled weight decay)
    Accepts:
        params: Iterable of parameters to optimize
        lr: learning rate
        betas: tuple
        eps: epsilon for numerical stability
        weight_decay: lambda (decoupled weight decay)
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if len(betas) != 2:
            raise ValueError("betas must be a tuple of two floats")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        "Performs a single optimization step."
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
            
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                
                denom = exp_avg_sq.sqrt().add_(eps)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
    