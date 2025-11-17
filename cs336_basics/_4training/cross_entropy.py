import torch

def cross_entropy(o, x):
    '''
    Compute average cross-entropy loss:
    l_i = -log softmax(o)[x]
    Works with arbitrary leading batch-like dimensions, where the last dimension of `o`
    is the vacabulary dimension.
    o: (..., V)
    x: (...,)
    The implementation:
    - subtracts the pre-example maximum for numerical stability, 
    - cancels out log/exp where possible using the log-sum-exp trick,
    - return the mean across all batch-like dimensions.
    '''
    if o.dim() < 1:
        raise ValueError("o must have at least one dimension (vocab dimension)")
    if x.shape != o.shape[:-1]:
        raise ValueError("Targets `x` must have shape equal to o.shape[:-1]")

    o = o.to(torch.float32)
    x = x.long()

    o_max = o.max(dim=-1, keepdim=True).values # shape (..., 1)
    o_shift = o - o_max # (..., V)

    # torch.sum() (..., V) -> (...,)
    log_sum_exp = torch.log(torch.sum(torch.exp(o_shift), dim=-1))

    # extracts the shifted logit values for the target indices x from o_shift
    # o_shift has shape (..., V)
    # x.unsqueeze(-1) reshape x from (...,) to (...,1)
    # dim = -1, last dimension  (..., 1)
    # squeeze(-1) (...,1) -> (...,)
    o_target_shift = o_shift.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

    loss = log_sum_exp - o_target_shift # (...)
    return loss.mean()