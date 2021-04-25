import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer

from hab.model.model import HABsModelCNN


def model_selector(model_type: str) -> Module:
    """
    Retrieve specified model object.

    :param model_type: Name of desired model.
    :return: Instance of HABs Model.
    """
    if model_type == "CNN":
        return HABsModelCNN()
    else:
        raise ValueError(f"Model type must be CNN. Value received: {model_type}")


def criterion_selector(loss_type: str) -> Module:
    """
    Retrieve specified loss object.

    :param loss_type: Name of desired loss function.
    :return: Instance of Loss object.
    """
    if loss_type == "Cross Entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss type must be Cross Entropy. Value received: {loss_type}")


def optimizer_selector(optim_type: str) -> Optimizer:
    """
    Retrieve specified optimizer object.

    :param optim_type: Name of desired optimizer.
    :return: Instance of Optimizer object.
    """
    if optim_type == "Adam":
        return optim.Adam(lr=1e-3)
    else:
        raise ValueError(f"Optimizer type must be Adam. Value received: {optim_type}")
