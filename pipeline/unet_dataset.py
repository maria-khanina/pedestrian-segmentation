import os

import cv2
import numpy
import torch
import torch.utils.data

from pipeline import config


class PedestrianSegmentationDataset(torch.utils.data.Dataset):
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

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (config.UNET_INPUT_WIDTH, config.UNET_INPUT_HEIGHT),
        )
        image = image.astype(numpy.float32) / 255.0
        mean = numpy.array(config.IMAGENET_MEAN, dtype=numpy.float32)
        standard_deviation = numpy.array(
            config.IMAGENET_STANDARD_DEVIATION,
            dtype=numpy.float32,
        )
        image = (image - mean) / standard_deviation
        image = numpy.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def _load_mask_tensor(self, mask_path: str) -> torch.Tensor:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask,
            (config.UNET_INPUT_WIDTH, config.UNET_INPUT_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
        binary_mask = (mask > 0).astype(numpy.float32)
        return torch.from_numpy(binary_mask).unsqueeze(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_name = self.image_file_names[index]
        image_path = os.path.join(self.images_directory, file_name)
        mask_path = os.path.join(self.masks_directory, file_name)
        image_tensor = self._load_image_tensor(image_path)
        mask_tensor = self._load_mask_tensor(mask_path)
        return image_tensor, mask_tensor


def list_file_names(directory: str) -> list[str]:
    return sorted(os.listdir(directory))
