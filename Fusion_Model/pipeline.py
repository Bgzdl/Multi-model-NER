import torch
import torch.nn as nn
import torch.nn.functional as F


class Pipeline(nn.Module):
    def __init__(self, dimension, categories):
        super(Pipeline, self).__init__()

        self.dimension = dimension
        self.categories = categories

        self.fc1 = nn.Linear(in_features=self.dimension, out_features=self.categories)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        return x
