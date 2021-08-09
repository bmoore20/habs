from PIL import Image
from torchvision import transforms

from hab.transformations import CropTimestamp


def test_transforms():
    im = Image.open("../../testing/data/images/bgaclear00111.jpg")

    before_w, before_h = im.size
    transform = transforms.Compose([CropTimestamp()])
    im_t = transform(im)
    after_w, after_h = im_t.size

    assert (before_w == after_w) and (before_h > after_h)
