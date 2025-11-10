import sys
sys.path.insert(0, '/chronos_data/huixu/assignment1-basics')

import torch
from cs336_basics._3modules.transformer_block import TransformerBlock
from cs336_basics._3modules.RoPE import RoPE

def debug_transformer():
    d_model, num_heads, d_ff = 64, 4, 128
    batch_size, seq_len = 1, 16
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建 TransformerBlock
    rope = RoPE(theta=10000, d_k=16, max_seq_len=16)
    block = TransformerBlock(d_model, num_heads, d_ff, rope=rope)
    
    print("Input shape:", x.shape)
    
    # 逐步执行 forward，添加调试信息
    print("\n=== Step 1: First RMSNorm ===")
    x_norm1 = block.rms_norm_1(x)
    print("After RMSNorm1:", x_norm1.shape, "mean:", x_norm1.mean().item(), "std:", x_norm1.std().item())
    
    print("\n=== Step 2: Multi-Head Self-Attention ===")
    attn_out = block.multihead_self_attn(x_norm1, mask=None)
    print("After Attention:", attn_out.shape, "mean:", attn_out.mean().item(), "std:", attn_out.std().item())
    
    print("\n=== Step 3: First Residual ===")
    x_res1 = x + attn_out
    print("After residual 1:", x_res1.shape, "mean:", x_res1.mean().item(), "std:", x_res1.std().item())
    
    print("\n=== Step 4: Second RMSNorm ===")
    x_norm2 = block.rms_norm_2(x_res1)
    print("After RMSNorm2:", x_norm2.shape, "mean:", x_norm2.mean().item(), "std:", x_norm2.std().item())
    
    print("\n=== Step 5: SwiGLU FFN ===")
    ffn_out = block.point_wise_ff(x_norm2)
    print("After FFN:", ffn_out.shape, "mean:", ffn_out.mean().item(), "std:", ffn_out.std().item())
    
    print("\n=== Step 6: Second Residual ===")
    output = x_res1 + ffn_out
    print("Final output:", output.shape, "mean:", output.mean().item(), "std:", output.std().item())
    
    print("\n=== Complete forward pass ===")
    output_full = block.forward(x, mask=None)
    print("Output from forward():", output_full.shape, "mean:", output_full.mean().item(), "std:", output_full.std().item())
    
    print("\nAre they equal?", torch.allclose(output, output_full))

if __name__ == "__main__":
    debug_transformer()