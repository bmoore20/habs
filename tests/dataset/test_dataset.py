import pytest

from hab.dataset import HABsDataset
from hab.utils.testing_helper import make_transform


def test_incorrect_mode_given():
    with pytest.raises(ValueError):
        transform = make_transform()
        image_dataset = "../images/"

        HABsDataset(image_dataset, transform, "error")


def test_dataset_train_mode():
    transform = make_transform()
    image_dataset = "../images/"

    hab_dataset = HABsDataset(image_dataset, transform)
    data = hab_dataset[0]

    # image is labeled bga and returned tuple contains image + target
    assert (data[1] == 0) and (len(data) == 2)
