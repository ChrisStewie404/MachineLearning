import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y  # residual

class ResidualMLPClassifier(nn.Module):
    def __init__(self, input_dim=512, n_classes=100, num_blocks=12, hidden_dim=4096, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(input_dim, input_dim)  # optional projection
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(input_dim, hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.head_norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.head_norm(x)
        return self.head(x)
