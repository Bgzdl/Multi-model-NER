import torch
import torch.nn as nn
import numpy as np


class Pipline(nn.Module):
    def __init__(self, dimension, max_length, categories):
        super(Pipline, self).__init__()
        self.dimension = dimension
        self.max_length = max_length
        self.categories = categories
        self.fc1 = nn.Linear(in_features=self.dimension, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=self.categories)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=-1)
        return x


if __name__ == "__main__":
    max_length = 10
    dimension = 10
    categories = 4
    m = Pipline(dimension=dimension, max_length=max_length, categories=categories)
    x = torch.rand(max_length, dimension)
    y = m(x)
    print(x.shape)
    print(y.shape)