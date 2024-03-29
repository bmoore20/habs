import logging
import typer
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, List

from hab.dataset import HABsDataset
from hab.transformations import CropTimestamp
from hab.utils import habs_logging, selectors
from hab.utils.training_helper import training_lap, validation_lap, evaluate

# ------------ logging ------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s HABs:%(levelname)s - %(name)s - %(message)s"
)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
logger.addHandler(habs_logging.ch)
logger.addHandler(habs_logging.fh)
# ---------------------------------

# -------- tensorboard ------------
writer = habs_logging.sw
# ---------------------------------


def train(
    train_data_dir: str,
    val_data_dir: str,
    test_data_dir: str,
    save_model_dir: str,
    model: nn.Module,
    epochs: int,
    optimizer: Optimizer,
    criterion: nn.Module,
    size_of_batch: int = 1,
    magnitude_increase: int = 1,
):
    """
    Complete training and evaluation for HABsModelCNN.

    :param train_data_dir: Directory path for training dataset.
    :param val_data_dir: Directory path for validation dataset.
    :param test_data_dir: Directory path for testing dataset.
    :param save_model_dir: Directory path where trained model will be saved.
    :param model: Model to be trained and evaluated.
    :param epochs: Number of epochs that training loop will complete.
    :param optimizer: Optimization algorithm used to train the model.
    :param criterion: Loss function used to train the model.
    :param size_of_batch: Size of batches used in epochs. Default is 1.
    :param magnitude_increase: Amount to multiply original number of samples by. Default is 1.
    """

    # Referenced: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://realpython.com/python-logging/

    logger.info("Loading data.")

    # TODO - experiment with different combinations of transformations
    train_val_transform = transforms.Compose(
        [
            CropTimestamp(),
            transforms.RandomCrop((32, 32)),
            transforms.RandomRotation(90),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # Calculated on ImageNet dataset
        ]
    )

    test_transform = transforms.Compose(
        [
            CropTimestamp(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # Calculated on ImageNet dataset
        ]
    )

    logger.info(f"Train/Val Transforms Applied: {train_val_transform}")
    logger.info(f"Test Transforms Applied: {test_transform}")
    logger.info(f"Model Architecture: {model}")

    train_dataset = HABsDataset(
        train_data_dir, train_val_transform, "train", magnitude_increase
    )
    val_dataset = HABsDataset(
        val_data_dir, train_val_transform, "validation", magnitude_increase
    )
    test_dataset = HABsDataset(
        test_data_dir, test_transform, "test", magnitude_increase
    )

    train_loader = DataLoader(train_dataset, batch_size=size_of_batch)
    val_loader = DataLoader(val_dataset, batch_size=size_of_batch)
    test_loader = DataLoader(test_dataset, batch_size=size_of_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device.type}")

    logger.info("Initial Seed: %d" % (torch.initial_seed()))

    logger.info("Training model.")
    for epoch in range(epochs):
        train_loss = training_lap(model, train_loader, optimizer, criterion)
        val_loss = validation_lap(model, val_loader, criterion)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        logger.info(
            "Epoch #{} Training Loss: {:.7f} Validation Loss: {:.7f}".format(
                epoch, train_loss, val_loss
            )
        )

    torch.save(model.state_dict(), save_model_dir)
    logger.info("Saved trained model.")

    logger.info("Testing model.")
    predictions, classifications, total, correct, targets = evaluate(model, test_loader)
    logger.info(f"Predicted Values: {predictions}")
    logger.info(f"  Target Values: {targets}")
    logger.info(f"Classifications: {classifications}")
    logger.info(
        "Accuracy of the network on the test images: %d %%" % (100 * correct / total)
    )


def main(
    train_dataset: str,
    val_dataset: str,
    test_dataset: str,
    save_model_dir: str,
    model_type: str,
    epochs: int,
    loss_type: str,
    optimizer_type: str,
    learn_rate: float,
    layers: Optional[List[int]] = None,
    batch_size: int = typer.Argument(1),
    magnitude_increase: int = typer.Argument(1),
):
    """
    Carry out full HABs program functionality.

    Pass in directory paths for training, validation and testing datasets.
    Provide directory path where trained model will be saved, model type,
    number of epochs, loss type, optimizer type, learning rate, ResNet model layers,
    batch size and dataset magnitude increase value.

    :param train_dataset: Directory path for training dataset.
    :param val_dataset: Directory path for validation dataset.
    :param test_dataset: Directory path for testing dataset.
    :param save_model_dir: Directory path where trained model will be saved.
    :param model_type: Model to be trained and evaluated.
    :param epochs: Number of epochs that training loop will complete.
    :param loss_type: Loss function used to train the model ("Cross Entropy").
    :param optimizer_type: Optimization algorithm used to train the model ("Adam").
    :param learn_rate: Learning rate to use during training.
    :param layers: Layers for ResNet model.
    :param batch_size: Size of batches used in epochs.
    :param magnitude_increase: Amount to multiply original number of samples by.
    """
    logger.info(
        f"Model: {model_type} Epochs: {epochs} Loss: {loss_type} "
        f"Optimizer: {optimizer_type} Learn Rate: {learn_rate} Layers: {layers} "
        f"Batch Size: {batch_size} Mag Inc: {magnitude_increase}"
    )
    model = selectors.model_selector(model_type, layers)
    criterion = selectors.criterion_selector(loss_type)
    optimizer = selectors.optimizer_selector(optimizer_type, model, learn_rate)
    train(
        train_dataset,
        val_dataset,
        test_dataset,
        save_model_dir,
        model,
        epochs,
        optimizer,
        criterion,
        batch_size,
        magnitude_increase,
    )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    typer.run(main)
