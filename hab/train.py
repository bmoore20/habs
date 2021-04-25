import logging
import typer
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from hab.dataset import HABsDataset
from hab.transformations import Rescale, Crop
from hab.utils import habs_logging, selectors
from hab.utils.training_helper import training_laps, evaluate

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


def train(
        train_data_dir: str,
        test_data_dir: str,
        model: Module,
        epochs: int,
        optimizer: Optimizer,
        criterion: Module,
        magnitude_increase: int = 1
):
    """
    Complete training and evaluation for HABsModelCNN.

    :param train_data_dir: Directory path for training dataset.
    :param test_data_dir: Directory path for testing dataset.
    :param model: Model to be trained and evaluated.
    :param epochs: Number of epochs that training loop will complete.
    :param optimizer: Optimization algorithm used to train the model.
    :param criterion: Loss function used to train the model.
    :param magnitude_increase: Amount to multiple original number of samples by.
    """
    # Referenced: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://realpython.com/python-logging/

    logger.info("Loading data.")

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

    logger.info("Initial Seed: %d" % (torch.initial_seed()))

    logger.info("Training model.")
    training_laps(model, train_loader, epochs, optimizer, criterion)

    logger.info("Testing model.")
    evaluate(model, test_loader)


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
    logger.info(f"Model: {model_type} Epochs: {epochs} Optimizer: {optimizer_type} Loss: {loss_type}")
    model = selectors.model_selector(model_type)
    optimizer = selectors.optimizer_selector(optimizer_type)
    criterion = selectors.criterion_selector(loss_type)
    train(train_dataset, test_dataset, model, epochs, optimizer, criterion, magnitude_increase)


if __name__ == "__main__":
    typer.run(main)
