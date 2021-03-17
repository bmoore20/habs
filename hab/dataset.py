from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional
from pathlib import Path


class HABsDataset(Dataset):
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
    # TODO - specify typing for parameters and returns of all methods
    # TODO - doc strings
    # TODO - "train" and "test" modes have same implementation -> re-evaluate design decision to have separate calls

    def __init__(self, data_dir: str, mode: str = "train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        if mode in {"train", "test", "classify"}:
            self.mode = mode
        else:
            raise ValueError("Dataset mode must be either train, test, or classify. Value received: {}".format(mode))

    def __len__(self):
        return len(self.image_paths)

    def _get_image_paths(self):
        all_paths = Path(self.data_dir).glob("**/*")

        if self.mode in {"train", "test"}:
            # Only select image files that are in specified class directories
            image_paths = [fp for fp in all_paths if fp.suffix == ".jpg" and fp.parent.name in {"bga", "clear", "turbid"}]
        else:
            # Images for classify do not have to be sorted into specific class directories
            image_paths = [fp for fp in all_paths if fp.suffix == ".jpg"]

        return image_paths

    def _get_image(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

    def _make_target(self, idx):
        image_path = self.image_paths[idx]
        _class = image_path.parent.name

        if _class == "bga":
            target = 0
        elif _class == "clear":
            target = 1
        elif _class == "turbid":
            target = 2
        else:
            raise ValueError("Cannot encode target value. Class name must be bga, clear, or turbid.")

        return target

    def __getitem__(self, idx):
        if self.mode in {"train", "test"}:
            image = self._get_image(idx)
            target = self._make_target(idx)
            return image, target
        else:
            # Images to be classified do not have known target values
            image = self._get_image(idx)
            return image
