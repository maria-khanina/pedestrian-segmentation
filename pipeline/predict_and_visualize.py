import logging
import os

import cv2
import numpy
import torch

from pipeline import segmentation_settings as settings
from pipeline.array_types import ImageArray
from pipeline.device import select_device
from pipeline.image_io import read_color_image, resize_image, write_image
from pipeline.logging_setup import configure_logging
from pipeline.unet_dataset import prepare_unet_input
from pipeline.unet_model import build_unet_model, load_model_state

_logger = logging.getLogger(__name__)


def _load_model(device: str) -> torch.nn.Module:
    model = build_unet_model()
    model.load_state_dict(load_model_state(settings.UNET_CHECKPOINT_PATH, device))
    return model.to(device).eval()


def _predict_binary_mask(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str,
) -> ImageArray:
    with torch.no_grad():
        logits: torch.Tensor = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.sigmoid(logits)
    predicted_mask = probabilities[0, 0] > settings.UNET_PREDICTION_THRESHOLD
    return numpy.asarray(predicted_mask.cpu(), dtype=numpy.uint8)


def _overlay_mask_on_image(
    image_bgr: ImageArray,
    predicted_mask: ImageArray,
) -> ImageArray:
    image_height, image_width = image_bgr.shape[:2]
    resized_mask = resize_image(
        predicted_mask,
        image_width,
        image_height,
        cv2.INTER_NEAREST,
    )
    colored_overlay = image_bgr.copy()
    colored_overlay[resized_mask > 0] = settings.PREDICTION_OVERLAY_COLOR_BGR
    blended = cv2.addWeighted(
        colored_overlay,
        settings.PREDICTION_OVERLAY_ALPHA,
        image_bgr,
        1.0 - settings.PREDICTION_OVERLAY_ALPHA,
        0,
    )
    return numpy.asarray(blended, dtype=numpy.uint8)


def _visualize_single_image(
    model: torch.nn.Module,
    file_name: str,
    device: str,
) -> None:
    image_path = os.path.join(
        settings.AUGMENTED_TEST_IMAGES_DIRECTORY,
        file_name,
    )
    image_bgr = read_color_image(image_path)
    predicted_mask = _predict_binary_mask(
        model,
        prepare_unet_input(image_bgr),
        device,
    )
    overlaid_image = _overlay_mask_on_image(image_bgr, predicted_mask)
    write_image(
        os.path.join(settings.PREDICTIONS_DIRECTORY, file_name),
        overlaid_image,
    )


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    os.makedirs(settings.PREDICTIONS_DIRECTORY, exist_ok=True)
    device = select_device()
    _logger.info(f"Using device: {device}")
    model = _load_model(device)
    for file_name in sorted(os.listdir(settings.AUGMENTED_TEST_IMAGES_DIRECTORY)):
        _logger.info(f"Predicting {file_name}")
        _visualize_single_image(model, file_name, device)
    _logger.info(
        f"Saved prediction overlays to {settings.PREDICTIONS_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
