from torch import nn
from ml.src.models.common.attention.attention_rope import MultiHeadAttention
from ml.src.models.common.feed_forward.swiglu import FeedForward
from ml.src.models.common.normalization.rms import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, dtype, num_heads, hidden_dim, dkv_bias):
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attention = MultiHeadAttention(d_in=emb_dim, d_out=emb_dim,
         context_length=context_length,
          num_heads=num_heads , dtype=dtype, qkv_bias=dkv_bias)
        self.norm2 = RMSNorm(emb_dim)
        self.ff = FeedForward(emb_dim, hidden_dim, dtype)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x