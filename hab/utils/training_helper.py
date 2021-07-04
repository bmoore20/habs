import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Tuple


def training_lap(
    model: Module, data_loader: DataLoader, optimizer: Optimizer, criterion: Module
) -> float:
    """
    Execute a training lap on the model.

    :param model: Model to be trained.
    :param data_loader: Data loader that contains training data.
    :param optimizer: Optimization algorithm used to train the model.
    :param criterion: Loss function used to train the model.
    :return: Running loss from training lap.
    """
    running_loss = 0
    model.train()
    for batch in data_loader:
        images, targets = batch

        optimizer.zero_grad()

        outputs = model(images)  # nn.module __call__()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss


def validation_lap(model: Module, data_loader: DataLoader, criterion: Module) -> float:
    """
    Execute a validation lap on the model.

    :param model: Model to be trained.
    :param data_loader: Data loader that contains training data.
    :param criterion: Loss function used to train the model.
    :return: Running loss from validation lap.
    """
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch

            outputs = model(images)  # nn.module __call__()
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss


def evaluate(
    model: Module, data_loader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Evaluate the trained model.

    :param model: Model to be evaluated.
    :param data_loader: Data loader that contains the test/validation data.
    :return predictions: Predicted probabilities for possible classes per image.
    :return classifications: Classification value for each image.
    :return total: Total number of images tested.
    :return correct: Number of test images that were classified correctly.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, targets = batch

            outputs = model(images)  # nn.module __call__()
            predictions = outputs.data
            _, classifications = torch.max(predictions, 1)
            total += targets.size(0)
            correct += (classifications == targets).sum().item()

            return predictions, classifications, total, correct
