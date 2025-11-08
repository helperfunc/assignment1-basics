import torch
from .softmax import softmax

def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor,  
    values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    '''
    Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v)
    Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
    seq_len). The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    # queries: [..., seq_len_q, d_k]
    # keys: [..., seq_len_k, d_k]
    # keys.transpose(-2, -1): [..., d_k, seq_len_k]
    # pre_softmax: [..., seq_len_q, seq_len_k]
    return [..., seq_len_q, d_v]
    '''
    d_k = queries.shape[-1]
    # [batch_size, ..., seq_len, seq_len]
    pre_softmax = queries @ keys.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=queries.dtype))

    if mask is not None:
        # mask: [seq_len_q, seq_len_k] or [..., seq_len_q, seq_len_k]
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    
    # atten_weights: [..., seq_len_q, seq_len_k]
    atten_weights = softmax(pre_softmax, dim=-1)
    atten_weights = torch.nan_to_num(atten_weights, nan=0.0)
    # values: [..., seq_len_k, d_v]
    # atten_weights @ values: [..., seq_len_q, d_v]
    output = atten_weights @ values
    return output