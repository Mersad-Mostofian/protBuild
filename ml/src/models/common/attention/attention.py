from torch import nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
        dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by number of heads."

        self.d_out = d_out
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.Wquery = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wkey = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wvalue = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))
    
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

        atten_scores = queries @ keys.transpose(2, 3)

        atten_scores.masked_fill_(self.mask[:num_tokens, :num_tokens]
                                  , -torch.inf)

        atten_weight = torch.softmax(atten_scores / keys.shape[-1]**0.5, dim=-1)
        atten_weight = self.dropout(atten_weight)

        context_vec = (atten_weight @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec