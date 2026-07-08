import torch
import torch.nn as nn
from gelu import GELU

class FFN(nn.Module):
    def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)

        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)