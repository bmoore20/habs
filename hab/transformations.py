from PIL import Image


class CropTimestamp(object):
    """
    Crop image to remove date and timestamp from bottom of image.
    """

    def __init__(self):
        ...

    def __repr__(self):
        return self.__class__.__name__

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
