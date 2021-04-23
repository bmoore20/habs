import logging
import typer
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from hab.dataset import HABsDataset
from hab.model.model import HABsModelCNN
from hab.transformations import Rescale, Crop
from hab.utils import habs_logging
from hab.utils .utils import model_filter, criterion_filter, optimizer_filter

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

# TODO - running_loss warning -> initiate before training loop (running_loss = 0)?
# TODO - sum() warning -> Unresolved attribute reference 'sum' for class 'bool'
# TODO - check order that individual transforms are executed in transforms.Compose (right to left, 1st then 2nd)


def train(
        train_data_dir: str,
        test_data_dir: str,
        habs_model: Module,
        epochs: int,
        optimizer: Optimizer,
        criterion: Module,
        magnitude_increase: int = 1
):
    """
    Complete training and evaluation for HABsModelCNN.

    :param train_data_dir: Directory path for training dataset.
    :param test_data_dir: Directory path for testing dataset.
    :param habs_model: Model to be trained and evaluated.
    :param epochs: Number of epochs that training loop will complete.
    :param optimizer: Optimization algorithm used to train the model.
    :param criterion: Loss function used to train the model.
    :param magnitude_increase: Amount to multiple original number of samples by.
    """
    # Referenced: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://realpython.com/python-logging/

    logger.info("Loading data.")

    # Replaces [image = np.array(image.resize((32, 32))) / 255.0] from orig program
    # ToTensor converts a PIL Image (H x W x C) in the range [0, 255] to a
    # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    # TODO - experiment with different combinations of transformations
    data_transform = transforms.Compose([
        Crop(),
        Rescale((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Calculated on ImageNet dataset
    ])

    train_dataset = HABsDataset(train_data_dir, data_transform, "train", magnitude_increase)
    test_dataset = HABsDataset(test_data_dir, data_transform, "test", magnitude_increase)

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    # criterion = nn.CrossEntropyLoss()

    logger.info("Initial Seed: %d" % (torch.initial_seed()))

    # instantiate HABs CNN
    # habs_model = model

    logger.info("Training model.")

    # train
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            images, targets = data

            optimizer.zero_grad()

            outputs = habs_model(images)  # nn.module __call__()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                logger.info("[%d, %5d] loss: %.3f" %
                            (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    logger.info("Testing model.")

    # test
    correct = 0
    total = 0
    habs_model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, targets = data

            outputs = habs_model(images)  # nn.module __call__()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    logger.info("Accuracy of the network on the 10000 test images: %d %%" % (
            100 * correct / total))


def main(
        train_dataset: str,
        test_dataset: str,
        model_type: str,
        epochs: int,
        optimizer_type: str,
        loss_type: str,
        magnitude_increase: int
):
    """
    Carry out full HABs program functionality.

    Pass in directory paths for training and testing datasets, model type,
    number of epochs, optimizer type, loss type and dataset magnitude increase value.
    """
    model = model_filter(model_type)
    optimizer = optimizer_filter(optimizer_type)
    criterion = criterion_filter(loss_type)
    train(train_dataset, test_dataset, model, epochs, optimizer, criterion, magnitude_increase)


if __name__ == "__main__":
    typer.run(main)
