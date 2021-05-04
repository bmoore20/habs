import logging
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hab.utils import habs_logging

# ------------ logging ------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s HABs:%(levelname)s - %(name)s - %(message)s"
)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
logger.addHandler(habs_logging.ch)
logger.addHandler(habs_logging.fh)
# ---------------------------------


def training_laps(
        model: Module,
        data_loader: DataLoader,
        epochs: int,
        optimizer: Optimizer,
        criterion: Module
):
    """
    Execute training on model by completing specified number of epochs/laps through database.

    :param model:Model to be trained.
    :param data_loader: Data loader that contains training data.
    :param epochs: Number of laps to complete through dataset.
    :param optimizer: Optimization algorithm used to train the model.
    :param criterion: Loss function used to train the model.
    """
    for epoch in range(epochs):
        for i, data in enumerate(data_loader, 0):
            images, targets = data

            optimizer.zero_grad()

            outputs = model(images)  # nn.module __call__()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                logger.info("[%d, %5d] loss: %.3f" %
                            (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def evaluate(
        model: Module,
        data_loader: DataLoader
):
    """
    Evaluate the trained model.

    :param model: Model to be evaluated.
    :param data_loader: Data loader that contains the test/validation data.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, targets = data

            outputs = model(images)  # nn.module __call__()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    logger.info("Accuracy of the network on the 10000 test images: %d %%" % (
            100 * correct / total))
    
