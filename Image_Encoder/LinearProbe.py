import torch.nn as nn
import torch


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activate_fun: bool = False):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        if activate_fun:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.projection(x)
        if self.relu:
            x = self.relu(x)
        return x
