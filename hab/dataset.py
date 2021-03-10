from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional
from pathlib import Path


class HABsDataset(Dataset):
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Referenced: https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
    # TODO - torchvision's transforms
    # TODO - specify typing for parameters and returns of all methods

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def _get_image_paths(self):
        all_paths = Path(self.data_dir).glob("**/*")

        # Only select image files. Ignore files that are not images.
        image_paths = [fp for fp in all_paths if (fp.name != ".DS_Store") and
                       (fp.parent.name == "bga" or fp.parent.name == "clear" or fp.parent.name == "turbid")]
        
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

    # TODO - pull out transform into Transform class
    # def _transform(self, image: Image) -> Image:
    #     image = np.array(image.resize((32, 32))) / 255.0
    #     return image

    def __getitem__(self, idx):
        image = self._get_image(idx)
        target = self._make_target(idx)

        return image, target
