import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# TODO - meaning of parameters for CNN methods
# TODO - add in 3rd conv and pool to match Keras model
# TODO - torch.nn vs. torch.nn.functional


class HABsModelCNN(nn.Module):
    """
    Convolutional Neural Network for detecting Harmful Algal Blooms.
    """

    # Referenced: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self):
        """
        Construct instance of a HABsModelCNN object.
        """
        super(HABsModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Map input tensor to output tensor.

        :param x: Input tensor.
        :return: Tensor received after network transformation.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
