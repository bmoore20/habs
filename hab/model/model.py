import torch.nn as nn
import torch.nn.functional as F


class HABsModelCNN(nn.Module):
    # Referenced: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # TODO - specify typing for parameters and returns of all methods
    # TODO - doc strings
    # TODO - meaning of parameters
    # TODO - add in 3rd conv and pool to match Keras model
    # TODO - torch.nn vs. torch.nn.functional
    # TODO - figure out if HABsModel can handle torch.FloatTensor as input (or need PIL image?)

    def __init__(self):
        super(HABsModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



