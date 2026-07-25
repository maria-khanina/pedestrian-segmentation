import logging
import os

import cv2
import numpy
import torch

from pipeline import config
from pipeline.device import select_device
from pipeline.logging_setup import configure_logging
from pipeline.unet_model import build_unet_model

_logger = logging.getLogger(__name__)


def _load_model(device: str) -> torch.nn.Module:
    model = build_unet_model()
    state_dict = torch.load(
        config.UNET_CHECKPOINT_PATH,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _preprocess_image(image_bgr: numpy.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (config.UNET_INPUT_WIDTH, config.UNET_INPUT_HEIGHT),
    )
    normalized = resized.astype(numpy.float32) / 255.0
    mean = numpy.array(config.IMAGENET_MEAN, dtype=numpy.float32)
    standard_deviation = numpy.array(
        config.IMAGENET_STANDARD_DEVIATION,
        dtype=numpy.float32,
    )
    normalized = (normalized - mean) / standard_deviation
    transposed = numpy.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(transposed).unsqueeze(0)


def _predict_binary_mask(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str,
) -> numpy.ndarray:
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.sigmoid(logits)
    predicted_mask = probabilities[0, 0] > config.UNET_PREDICTION_THRESHOLD
    return predicted_mask.cpu().numpy().astype(numpy.uint8)


def _overlay_mask_on_image(
    image_bgr: numpy.ndarray,
    predicted_mask: numpy.ndarray,
) -> numpy.ndarray:
    resized_mask = cv2.resize(
        predicted_mask,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    colored_overlay = image_bgr.copy()
    colored_overlay[resized_mask > 0] = config.PREDICTION_OVERLAY_COLOR_BGR
    return cv2.addWeighted(
        colored_overlay,
        config.PREDICTION_OVERLAY_ALPHA,
        image_bgr,
        1.0 - config.PREDICTION_OVERLAY_ALPHA,
        0,
    )


def _visualize_single_image(
    model: torch.nn.Module,
    file_name: str,
    device: str,
) -> None:
    image_path = os.path.join(
        config.AUGMENTED_TEST_IMAGES_DIRECTORY,
        file_name,
    )
    image_bgr = cv2.imread(image_path)
    image_tensor = _preprocess_image(image_bgr)
    predicted_mask = _predict_binary_mask(model, image_tensor, device)
    overlaid_image = _overlay_mask_on_image(image_bgr, predicted_mask)
    output_path = os.path.join(config.PREDICTIONS_DIRECTORY, file_name)
    cv2.imwrite(output_path, overlaid_image)


def main() -> None:
    configure_logging(logging.INFO, config.PIPELINE_LOG_PATH)
    os.makedirs(config.PREDICTIONS_DIRECTORY, exist_ok=True)
    device = select_device()
    _logger.info(f"Using device: {device}")
    model = _load_model(device)
    file_names = sorted(
        os.listdir(config.AUGMENTED_TEST_IMAGES_DIRECTORY)
    )
    for file_name in file_names:
        _logger.info(f"Predicting {file_name}")
        _visualize_single_image(model, file_name, device)
    _logger.info(
        f"Saved prediction overlays to {config.PREDICTIONS_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
