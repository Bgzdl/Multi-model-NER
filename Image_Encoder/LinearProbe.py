import torch.nn as nn
import torch


class LinearProbe(nn.Module):
    def __init__(self, output_dim: int, activate_fun: bool = False):
        super().__init__()
        self.projection = nn.Linear(output_dim, 512)
        if activate_fun:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.projection(x)
        if self.relu:
            x = self.relu(x)
        return x
