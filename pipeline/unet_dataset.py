import os

import cv2
import numpy
import torch
import torch.utils.data

from pipeline import segmentation_settings as settings
from pipeline.array_types import ImageArray
from pipeline.image_io import (
    convert_bgr_to_rgb,
    read_color_image,
    read_grayscale_image,
    resize_image,
)

ImageMaskPair = tuple[torch.Tensor, torch.Tensor]


def prepare_unet_input(image_bgr: ImageArray) -> torch.Tensor:
    resized = resize_image(
        convert_bgr_to_rgb(image_bgr),
        settings.UNET_INPUT_WIDTH,
        settings.UNET_INPUT_HEIGHT,
        cv2.INTER_LINEAR,
    )
    scaled = resized.astype(numpy.float32) / settings.PIXEL_MAXIMUM_VALUE
    mean = numpy.array(settings.IMAGENET_MEAN, dtype=numpy.float32)
    standard_deviation = numpy.array(
        settings.IMAGENET_STANDARD_DEVIATION,
        dtype=numpy.float32,
    )
    normalized = (scaled - mean) / standard_deviation
    return torch.as_tensor(normalized.transpose(2, 0, 1))


class PedestrianSegmentationDataset(torch.utils.data.Dataset[ImageMaskPair]):
    def __init__(
        self,
        images_directory: str,
        masks_directory: str,
        image_file_names: list[str],
    ) -> None:
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.image_file_names = image_file_names

    def __len__(self) -> int:
        return len(self.image_file_names)

    def _load_mask_tensor(self, mask_path: str) -> torch.Tensor:
        mask = resize_image(
            read_grayscale_image(mask_path),
            settings.UNET_INPUT_WIDTH,
            settings.UNET_INPUT_HEIGHT,
            cv2.INTER_NEAREST,
        )
        binary_mask = (mask > 0).astype(numpy.float32)
        return torch.as_tensor(binary_mask).unsqueeze(0)

    def __getitem__(self, index: int) -> ImageMaskPair:
        file_name = self.image_file_names[index]
        image_path = os.path.join(self.images_directory, file_name)
        mask_path = os.path.join(self.masks_directory, file_name)
        image_tensor = prepare_unet_input(read_color_image(image_path))
        mask_tensor = self._load_mask_tensor(mask_path)
        return image_tensor, mask_tensor


def list_file_names(directory: str) -> list[str]:
    return sorted(os.listdir(directory))
