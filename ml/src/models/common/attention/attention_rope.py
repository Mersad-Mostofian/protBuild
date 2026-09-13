import torch
from torch import nn
from ml.src.models.common.embedding.rope import compute_rope, rope_params

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
        num_heads, qkv_bias, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by number of heads."

        self.d_out = d_out
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.Wquery = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.Wkey = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.Wvalue = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))

        cos, sin = rope_params(head_dim=self.head_dim, context_length=self.context_length)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x): # x: position embedding + token embedding
        b, num_tokens, d_in = x.shape
        
        queries = self.Wquery(x) 
        keys = self.Wkey(x)
        values = self.Wvalue(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # b, num_heads, num_tokens, head_dim
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        atten_scores = queries @ keys.transpose(2, 3)

        atten_scores.masked_fill_(self.mask[:num_tokens, :num_tokens]
                                  , -torch.inf)

        atten_weight = torch.softmax(atten_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = (atten_weight @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec