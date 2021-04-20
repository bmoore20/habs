from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Union, Tuple
from pathlib import Path
import torch


class HABsDataset(Dataset):
    """
    Dataset of images from the Finger Lakes.
    Each image can be classified as either bga (blue-green algae), clear, or turbid.

    Currently the dataset supports JPG images that are 1280 x 736 px.
    """

    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

    def __init__(self, data_dir: str, transform: transforms.Compose, mode: str = "train", magnitude_increase: int = 1):
        """
        Construct instance of a HABsDataset object.

        :param data_dir: Directory path that contains the dataset's images.
        :param transform: Transforms to apply to dataset images. None not supported.
        :param mode: Mode of the dataset. Value needs to be "train", "test", or "classify".
        :param magnitude_increase: Amount to multiple original number of samples by.
        """
        self.data_dir = data_dir
        self.image_paths = self._get_image_paths()
        self.transform = transform
        self._set_mode(mode)
        self.magnitude_increase = magnitude_increase

    def __len__(self) -> int:
        """
        Length of HABsDataset.

        :return: Number of images in dataset.
        """
        return self.magnitude_increase * len(self.image_paths)

    def _set_mode(self, mode: str):
        """
        Set behavior for the dataset.

        :param mode: Dataset mode. Value needs to be "train", "test", or "classify".
        :raises: ValueError
        """
        if mode in {"train", "test", "classify"}:
            self.mode = mode
        else:
            raise ValueError(f"Dataset mode must be either train, test, or classify. Value received: {mode}")

    def _get_image_paths(self) -> List[Path]:
        """
        Retrieve image paths for each image in the dataset.

        :return: List containing all of the image paths.
        """
        all_paths = Path(self.data_dir).glob("**/*")

        if self.mode in {"train", "test"}:
            # Only select image files that are in specified class directories
            image_paths = [fp for fp in all_paths if
                           fp.suffix == ".jpg" and fp.parent.name in {"bga", "clear", "turbid"}]
        else:
            # Images for classify do not have to be sorted into specific class directories
            image_paths = [fp for fp in all_paths if fp.suffix == ".jpg"]

        return image_paths

    def _get_image(self, idx: int) -> torch.Tensor:
        """
        Retrieve image at the specified index from list containing all of the image paths.

        :param idx: Index of where image is in list.
        :return: Image at specified location.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            raise TypeError("None type is not supported. Torchvision transform object must be given.")

        return image

    def _make_target(self, idx: int) -> int:
        """
        Create numeric target values for image.

        :param idx: Index of where image is in list.
        :return: Encoded label for specified image. Value needs to be 0 (bga), 1 (clear), or 2 (turbid).
        :raises: ValueError
        """
        image_path = self.image_paths[idx]
        class_type = image_path.parent.name

        if class_type == "bga":
            target = 0
        elif class_type == "clear":
            target = 1
        elif class_type == "turbid":
            target = 2
        else:
            raise ValueError(f"Cannot encode target. Class must be bga, clear, or turbid. Value received: {class_type}")

        return target

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        """
        Retrieve a specific image from the dataset.

        :param idx: Index of where image is in list.
        :return: Image and target value if dataset mode is "train" or "test". Only Image if dataset mode is "classify".
        """
        if self.mode in {"train", "test"}:
            image = self._get_image(idx)
            target = self._make_target(idx)
            return image, target
        else:
            # Images to be classified do not have known target values
            image = self._get_image(idx)
            return image
