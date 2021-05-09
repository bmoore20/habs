import torch
from torch.nn import Module
from typing import Optional

from hab.model.model import HABsModelCNN


def model_selector(model_type: str) -> torch.nn.Module:
    """
    Retrieve specified model object.

    :param model_type: Name of desired model.
    :return: Instance of HABs Model.
    """
    if model_type == "CNN":
        return HABsModelCNN()
    else:
        raise ValueError(f"Model type must be CNN. Value received: {model_type}")


def criterion_selector(loss_type: str) -> torch.nn.Module:
    """
    Retrieve specified loss object.

    :param loss_type: Name of desired loss function.
    :return: Instance of Loss object.
    """
    if loss_type == "Cross Entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(
            f"Loss type must be Cross Entropy. Value received: {loss_type}"
        )


def optimizer_selector(
    optim_type: str, model: Module, learn_rate: Optional[float] = None
) -> torch.optim.Optimizer:
    """
    Retrieve specified optimizer object.

    :param optim_type: Name of desired optimizer.
    :param model: Model that contains parameters to be optimized.
    :param learn_rate: Learning rate. If None, torch default value is used.
    :return: Instance of Optimizer object.
    """
    if optim_type == "Adam":
        if learn_rate is not None:
            return torch.optim.Adam(model.parameters(), lr=learn_rate)
        else:
            return torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise ValueError(f"Optimizer type must be Adam. Value received: {optim_type}")
