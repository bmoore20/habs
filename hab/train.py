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
        valid_data_dir: str,
        test_data_dir: str,
        save_model_dir: str,
        model: Module,
        epochs: int,
        optimizer: Optimizer,
        criterion: Module,
        size_of_batch: int = 1,
        magnitude_increase: int = 1
):
    """
    Complete training and evaluation for HABsModelCNN.

    :param train_data_dir: Directory path for training dataset.
    :param valid_data_dir: Directory path for validation dataset.
    :param test_data_dir: Directory path for testing dataset.
    :param save_model_dir: Directory path where trained model will be saved. 
    :param model: Model to be trained and evaluated.
    :param epochs: Number of epochs that training loop will complete.
    :param optimizer: Optimization algorithm used to train the model.
    :param criterion: Loss function used to train the model.
    :param size_of_batch: Size of batches used in epochs. Default is 1. 
    :param magnitude_increase: Amount to multiply original number of samples by. Defalut is 1.
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
    valid_dataset = HABsDataset(valid_data_dir, data_transform, "validation", magnitude_increase)
    test_dataset = HABsDataset(test_data_dir, data_transform, "test", magnitude_increase)

    train_loader = DataLoader(train_dataset, batch_size = size_of_batch)
    valid_loader = DataLoader(valid_dataset, batch_size = size_of_batch)
    test_loader = DataLoader(test_dataset, batch_size = size_of_batch)

    logger.info("Initial Seed: %d" % (torch.initial_seed()))

    logger.info("Training model.")
    for epoch in range(epoch):
        train_loss = training_lap(model, train_loader, optimizer, criterion)
        valid_loss = validation_lap(model, valid_loader, criterion)
        
        logger.info("Epoch #{} Training Loss: {:.7f} Validation Loss: {:.7f}".format(epoch, train_loss, valid_loss))

    torch.save(model.state_dict(), save_model_dir)
    logger.info("Saved trained model.")
    
    logger.info("Testing model.")
    evaluate(model, test_loader)


def main(
        train_dataset: str,
        valid_dataset: str,
        test_dataset: str,
        save_model_dir: str,
        model_type: str,
        epochs: int,
        loss_type: str,
        optimizer_type: str,
        learn_rate: float,
        batch_size: int = typer.Argument(1),
        magnitude_increase: int = typer.Argument(1)
):
    """
    Carry out full HABs program functionality.

    Pass in directory paths for training, validation and testing datasets, directory path where trained model will be saved,
    model type, numberof epochs, loss type, optimizer type, learning rate, batch size and dataset magnitude increase value.
    """
    logger.info(
        f"Model: {model_type} Epochs: {epochs} Loss: {loss_type} Optimizer: {optimizer_type} Learn Rate: {learn_rate} Mag Inc: {magnitude_increase}"
    )
    model = selectors.model_selector(model_type)
    criterion = selectors.criterion_selector(loss_type)
    optimizer = selectors.optimizer_selector(optimizer_type, learn_rate)
    train(train_dataset, valid_dataset, test_dataset, save_model_dir, model, epochs, optimizer, criterion, batch_size, magnitude_increase)


if __name__ == "__main__":
    typer.run(main)
