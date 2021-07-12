from pathlib import Path


def is_image_path(path: Path) -> bool:
    """
    Determine if the given file path leads to an image.

    :param path: A single file path.
    :return: True if the file path leads to an image. False otherwise.
    """
    if path.suffix in {".jpg", ".jpeg", ".png", ".gif", ".svg"}:
        return True
    else:
        return False
