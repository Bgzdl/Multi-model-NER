import torch.nn as nn
import torch


class BottleNeck(nn.Module):
    def __init__(self, d_model, bottle_width=4):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_model * bottle_width)
        self.layer_2 = nn.Linear(d_model * bottle_width, d_model)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.activate(x)
        x = self.layer_2(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_of_heads, dropout_rate):
        super().__init__()
        self.MLP = BottleNeck(embed_dim, 4)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_of_heads)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = embed_dim
        self.heads = num_of_heads

    def forward(self, I, T):
        self.attention(T, I, I)
        x = self.dropout(self.attention(T, I, I)[0])
        x = self.layer_norm_1(T + x)
        identity = x.clone()
        x = self.dropout(self.MLP(x))
        x = self.layer_norm_2(x + identity)
        return x


class FusionBlock(nn.Module):
    def __init__(self, layer_num: int, embed_dim: int, num_of_heads: int, dropout_rate: float):
        super().__init__()
        self.blocks = []
        self.length = layer_num
        for i in range(layer_num):
            self.blocks.append(FusionLayer(embed_dim, num_of_heads, dropout_rate))

    def to(self, device):
        for block in self.blocks:
            block.to(device)

    def cuda(self, device='cuda'):
        for block in self.blocks:
            block.to(device)

    def forward(self, I, T):
        x = T.clone()
        for i in range(self.length):
            x = self.blocks[i](I, x)
        return x
