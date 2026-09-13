import torch

def rope_params(head_dim, theta_base=10_000, context_length=512):
    assert head_dim % 2 == 0, 'Embedding dimension must be even'

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[:(head_dim//2)].float() / head_dim))

    positions = torch.arange(context_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    b, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, 'Head dimension must be even'

    x1 = x[... , : head_dim // 2]
    x2 = x[... , head_dim // 2:]

    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([-x2, x1], dim=-1)
    x_rotated = (x*cos) + (rotated*sin)
    return x_rotated.to(dtype=x.dtype)


