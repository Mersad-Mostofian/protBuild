from torch import nn
import torch
from .transformerBlock import TransformerBlock
from ml.src.models.common.normalization.rms import RMSNorm

class LLAMA2(nn.Module):
    def __init__(self, emb_dim, vocab_size,
                  context_length, num_heads, hidden_dim,
                    n_layers, dtype, qkv_bias):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, dtype=dtype)
    
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(emb_dim=emb_dim,context_length=context_length, dtype=dtype,
              num_heads=num_heads, hidden_dim=hidden_dim, dkv_bias=qkv_bias) 
            for _ in range(n_layers)]
        )
        self.norm = RMSNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype)

    def forward(self, in_idx):
        tok_embedds = self.tok_emb(in_idx)
        x = tok_embedds
        x = self.transformer_blocks(x)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits