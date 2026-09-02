from torch import nn
import torch
from .transformerBlock import TransformerBlock
from ml.src.models.common.normalization.normalization import Normalization

class GPT2(nn.Module):
    def __init__(self, emb_dim, d_out, vocab_size,
                  context_length, num_heads,
                    n_layers, dropout, dkv_bias):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(emb_dim=emb_dim, d_out=d_out,
             context_length=context_length, dropout=dropout,
              num_heads=num_heads, dkv_bias=dkv_bias) 
            for _ in range(n_layers)]
        )
        self.norm = Normalization(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        positions = torch.arange(seq_len, device=in_idx.device)
        tok_embedds = self.tok_emb(in_idx)
        pos_embedds = self.pos_emb(positions)
        x = tok_embedds + pos_embedds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits