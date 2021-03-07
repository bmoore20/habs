from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from hab.dataset import HABsDataset


def train(data_dir):
    dataset = HABsDataset(data_dir)
    test_size = int(len(dataset) * 0.75)
    train_size = int(len(dataset) * 0.25)
    train_data, test_data = random_split(dataset, [test_size, train_size])

    train_loader = DataLoader(train_data)
    test_loader = DataLoader(test_data)

    # TODO - go through train_loader and test_loader (enumerate)
    # TODO - create and load in CNN model


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="path to root directory that contains image dataset")
    args = parser.parse_args()

    train(args.dataset)


if __name__ == "__main__":
    main()
