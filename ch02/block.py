import torch
import torch.nn as nn
from layer_norm import LayerNorm
from multihead_attention import MultiHeadAttention
from ffn import FFN
    
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
        super().__init__()

        head_dim = embed_dim // n_head
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x