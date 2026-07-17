from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ResizeMeta:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int]
    target_size: int


def list_images(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Path does not exist: {source}")
    return sorted(
        item
        for item in source.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_pad_to_square(image: Image.Image, size: int, fill: int = 0):
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size.")

    if width >= height:
        new_width = size
        new_height = max(1, int(height * size / width))
    else:
        new_height = size
        new_width = max(1, int(width * size / height))

    image = image.resize((new_width, new_height), resample=Image.BILINEAR)
    output = Image.new("RGB", (size, size), color=(fill, fill, fill))
    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    output.paste(image, (pad_left, pad_top))
    meta = ResizeMeta(
        original_size=(width, height),
        resized_size=(new_width, new_height),
        padding=(pad_left, pad_top),
        target_size=size,
    )
    return output, meta


class ImagePreprocessor:
    def __init__(self, image_size: int, preprocess: str = "none") -> None:
        self.image_size = int(image_size)
        self.preprocess = preprocess.lower()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _apply_preprocess(self, image: Image.Image) -> Image.Image:
        if self.preprocess == "none":
            return image
        if self.preprocess == "gaussian_blur":
            return image.filter(ImageFilter.GaussianBlur(radius=0.8))
        if self.preprocess == "median":
            return image.filter(ImageFilter.MedianFilter(size=3))
        if self.preprocess == "sharpen":
            return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        raise ValueError(f"Unsupported preprocess mode: {self.preprocess}")

    def load_image(self, path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def transform(self, image: Image.Image):
        processed = self._apply_preprocess(image)
        squared, meta = resize_pad_to_square(processed, self.image_size, fill=0)
        tensor = transforms.ToTensor()(squared)
        tensor = self.normalize(tensor).unsqueeze(0).to(torch.float32)
        return tensor, squared, meta

    def __call__(self, path: str | Path):
        original = self.load_image(path)
        tensor, processed, meta = self.transform(original)
        return tensor, original, processed, meta
