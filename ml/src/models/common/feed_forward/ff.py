from torch import nn
from ml.src.models.common.activation_functions.gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            GELU(),
            nn.Linear(4*emb_dim, emb_dim)
        )
    
    def forward(self, x):
        return self.layers(x)