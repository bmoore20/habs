from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple
import numpy as np
import os


class HABsDataset(Dataset):
    # TODO - not memory efficient because images are all stored in memory first and not read as required
    # TODO - one-hot-encode targets
    # TODO - torchvision's transforms
    # TODO - move HABsDataset to utils
    # TODO - handle train/test inside dataset or outside dataset
    # Referenced https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.data_samples = []
        self._init_dataset()

    def __len__(self):
        return len(self.data_samples)

    def _init_dataset(self):
        for image_class in os.listdir(self.data_root):
            image_path = os.path.join(self.data_root, image_class)
            image = Image.open(image_path)
            image = self._transform(image)
            target = self._encode_target(image_class)

            self.data_samples.append((image, target))

    @staticmethod
    def _encode_target(target: str) -> int:
        if target == "bga":
            return 0
        elif target == "clear":
            return 1
        elif target == "turbid":
            return 2
        else:
            raise ValueError("Cannot encode. Target must be bga, clear, or turbid.")

    @staticmethod
    def _transform(self, image: Image) -> Image:
        image = np.array(image.resize((32, 32))) / 255.0
        return image

    def __getitem__(self, idx) -> Tuple[Image, int]:
        return self.data_samples[idx]
