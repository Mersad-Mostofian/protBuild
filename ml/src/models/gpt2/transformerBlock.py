from torch import nn
from ml.src.models.common.transformer.transformer import MultiHeadAttention
from ml.src.models.common.feed_forward.ff import FeedForward
from ml.src.models.common.normalization.normalization import Normalization

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, d_out, context_length, dropout, num_heads, dkv_bias):
        super().__init__()
        self.norm1 = Normalization(emb_dim)
        self.attention = MultiHeadAttention(d_in=emb_dim, d_out=d_out,
         context_length=context_length, dropout=dropout,
          num_heads=num_heads , qkv_bias=dkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = Normalization(emb_dim)
        self.ff = FeedForward(emb_dim)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x