import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Union, Tuple, Optional
from pathlib import Path

from hab.utils.general_utils import is_image_path


class HABsDataset(Dataset):
    """
    Dataset of images from the Finger Lakes.
    Each image can be classified as either bga (blue-green algae) or non-algae (clear or turbid).

    Currently the dataset supports JPG images that are 1280 x 736 px.
    """

    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose,
        mode: str = "train",
        oversample_strength: int = 1,
        magnitude_increase: int = 1,
    ):
        """
        Construct instance of a HABsDataset object.

        :param data_dir: Directory path that contains the dataset's images.
        :param transform: Transforms to apply to dataset images. None not supported.
        :param mode: Mode of the dataset - "train", "evaluate", or "classify"
        :param oversample_strength: The magnitude to increase the bga images by.
        :param magnitude_increase: Amount to multiple original number of samples by.
        """
        self.data_dir = data_dir
        self._set_mode(mode)
        self.image_paths = self._get_image_paths()
        self.transform = transform
        self.oversample_strength = oversample_strength
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

        :param mode: Mode of the dataset - "train", "evaluate", or "classify"
        :raises: ValueError
        """
        if mode in {"train", "evaluate", "classify"}:
            self.mode = mode
        else:
            raise ValueError(
                f"Dataset mode must be either train, evaluate, or classify. "
                f"Value received: {mode}"
            )

    def _get_image_paths(self) -> List[Path]:
        """
        Retrieve image paths for each image in the dataset.

        :return: List containing all of the image paths.
        """
        all_paths = Path(self.data_dir).glob("**/*")
        bga_paths = []
        non_algae_paths = []

        if self.mode in {"train", "evaluate"}:
            # Only select image files that are in specified class directories
            for path in all_paths:
                if is_image_path(path) and path.parent.name == "bga":
                    bga_paths.append(path)
                elif is_image_path(path) and path.parent.name == "non_algae":
                    non_algae_paths.append(path)
                else:
                    raise ValueError(
                        "Cannot process file path. Path must include bga or non_algae folder."
                    )

            # Apply oversampling for bga images and combine image paths from both classes
            image_paths = bga_paths * self.oversample_strength + non_algae_paths

        else:
            # Images for classify do not have to be sorted into specific class directories
            image_paths = [path for path in all_paths if is_image_path(path)]

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
            raise TypeError(
                "None type is not supported. Torchvision transform object must be given."
            )

        return image

    def _make_target(self, idx: int) -> int:
        """
        Create numeric target values for image.

        :param idx: Index of where image is in list.
        :return: Encoded label for specified image. Value needs to be 0 (bga) or 1 (non-algae).
        :raises: ValueError
        """
        image_path = self.image_paths[idx]
        class_type = image_path.parent.name

        if class_type == "bga":
            target = 0
        elif class_type == "non_algae":
            target = 1
        else:
            raise ValueError(
                f"Cannot encode target. Class must be bga or non_algae. "
                f"Value received: {class_type}"
            )

        return target

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        """
        Retrieve a specific image from the dataset.

        :param idx: Index of where image is in list.
        :return: Image and target value if dataset mode is "train" or "evaluate".
                 Only Image if dataset mode is "classify".
        """
        if self.mode in {"train", "evaluate"}:
            idx = idx % len(self.image_paths)  # account for magnitude increase
            image = self._get_image(idx)
            target = self._make_target(idx)
            return image, target
        else:
            # Images to be classified do not have known target values
            idx = idx % len(self.image_paths)  # account for magnitude increase
            image = self._get_image(idx)
            return image
