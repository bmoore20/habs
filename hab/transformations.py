from PIL import Image

# TODO - specify typing for parameters and returns of all methods
# TODO - doc strings
# TODO - crop images to remove date & time pixels from bottom of images
# TODO - do we need to convert the PIL Images to torch images?
# TODO - do we want to train on a torch image or a PIL image? -> convert back to PIL?


class Rescale(object):
    # Referenced: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample: Image):
        image = sample.resize(self.output_size)

        return image






