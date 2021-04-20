from PIL import Image
from typing import Tuple


class Rescale(object):
    """
    Rescale image to a specified size.
    """
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, output_size: Tuple[int, int]):
        """
        Construct instance of a Rescale transformation.

        :param output_size: New image size dimensions.
        """
        self.output_size = output_size

    def __call__(self, sample: Image) -> Image:
        """
        Perform resizing of image.

        :param sample: Image to be rescaled.
        :return: Rescaled image.
        """
        image = sample.resize(self.output_size)

        return image


class Crop(object):
    """
    Crop image to remove date and time stamp from bottom of image.
    """

    def __init__(self):
        ...

    def __call__(self, sample: Image) -> Image:
        """
        Perform cropping of image.

        :param sample: Image to be cropped.
        :return: Cropped image.
        """
        width, height = sample.size

        left = 0
        upper = 0
        right = width
        lower = height - (height * 0.05)

        image = sample.crop((left, upper, right, lower))

        return image







