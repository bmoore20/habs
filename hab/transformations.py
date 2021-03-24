from PIL import Image

# TODO - specify typing for parameters and returns of all methods
# TODO - doc strings


class Rescale(object):
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample: Image):
        image = sample.resize(self.output_size)

        return image


class Crop(object):

    def __init__(self):
        ...

    def __call__(self, sample: Image):
        width, height = sample.size

        left = 0
        upper = 0
        right = width
        lower = height - (height * 0.05)

        image = sample.crop((left, upper, right, lower))

        return image







