from torchvision import transforms

from hab.transformations import CropTimestamp


def make_transform():
    transform = transforms.Compose(
        [
            CropTimestamp(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # Calculated on ImageNet dataset
        ]
    )

    return transform
