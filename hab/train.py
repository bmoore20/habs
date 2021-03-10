from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

from hab.dataset.dataset import HABsDataset
from hab.model.model import HABsModelCNN

# TODO - add transforms as parameters to HABsDataset initialization
# TODO - logger
# TODO- check to see if pytorch weight_decay parameter is same as keras decay parameter
# optimizer = optim.Adam(lr=1e-3, weight_decay=1e-3 / 50)


def train(data_dir):
    # Referenced: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    dataset = HABsDataset(data_dir)
    test_size = int(len(dataset) * 0.75)
    train_size = int(len(dataset) * 0.25)
    train_data, test_data = random_split(dataset, [test_size, train_size])

    train_loader = DataLoader(train_data)
    test_loader = DataLoader(test_data)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=1e-3)

    # train
    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            images, targets = data

            optimizer.zero_grad()

            # QUESTION: Why are we passing in argument to model if it wasn't specified in __init__?
            outputs = HABsModelCNN(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, targets = data

            # QUESTION: Why are we passing in argument to model if it wasn't specified in __init__?
            outputs = HABsModelCNN(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="path to root directory that contains image dataset")
    args = parser.parse_args()

    train(args.dataset)


if __name__ == "__main__":
    main()
