import logging
import os

import PIL.Image

from pipeline import config
from pipeline.coco_export import CocoDatasetBuilder
from pipeline.detection import GroundingDinoDetector
from pipeline.device import select_device
from pipeline.logging_setup import configure_logging
from pipeline.polygon_mask_conversion import convert_mask_to_polygons
from pipeline.segmentation import SamSegmentor

_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")

_logger = logging.getLogger(__name__)


def _list_image_file_names(directory: str) -> list[str]:
    all_entries = sorted(os.listdir(directory))
    image_file_names: list[str] = []
    for entry in all_entries:
        if entry.lower().endswith(_IMAGE_FILE_EXTENSIONS):
            image_file_names.append(entry)
    return image_file_names


def _label_single_image(
    image_path: str,
    detector: GroundingDinoDetector,
    segmentor: SamSegmentor,
    dataset_builder: CocoDatasetBuilder,
    file_name: str,
) -> None:
    image = PIL.Image.open(image_path).convert("RGB")
    bounding_boxes = detector.detect_boxes(
        image,
        config.DETECTION_TEXT_PROMPT,
        config.DETECTION_BOX_THRESHOLD,
        config.DETECTION_TEXT_THRESHOLD,
    )
    image_id = dataset_builder.add_image(
        file_name,
        image.width,
        image.height,
    )
    if len(bounding_boxes) == 0:
        return
    binary_masks = segmentor.segment_from_boxes(image, bounding_boxes)
    for binary_mask in binary_masks:
        polygons = convert_mask_to_polygons(
            binary_mask,
            config.MINIMUM_CONTOUR_AREA_IN_PIXELS,
            config.POLYGON_APPROXIMATION_EPSILON_FACTOR,
        )
        if len(polygons) == 0:
            continue
        dataset_builder.add_annotation(image_id, polygons)


def main() -> None:
    configure_logging(logging.INFO, config.PIPELINE_LOG_PATH)
    device = select_device()
    _logger.info(f"Using device: {device}")
    detector = GroundingDinoDetector(
        config.GROUNDING_DINO_MODEL_IDENTIFIER,
        device,
    )
    segmentor = SamSegmentor(
        config.SAM_MODEL_IDENTIFIER,
        device,
    )
    dataset_builder = CocoDatasetBuilder(
        config.PEDESTRIAN_CATEGORY_ID,
        config.PEDESTRIAN_CATEGORY_NAME,
        config.PEDESTRIAN_SUPERCATEGORY_NAME,
    )
    image_file_names = _list_image_file_names(config.RAW_IMAGES_DIRECTORY)
    for file_name in image_file_names:
        image_path = os.path.join(config.RAW_IMAGES_DIRECTORY, file_name)
        _logger.info(f"Labeling {file_name}")
        _label_single_image(
            image_path,
            detector,
            segmentor,
            dataset_builder,
            file_name,
        )
    dataset_builder.save_to_file(config.AUTO_LABELS_COCO_PATH)
    _logger.info(
        f"Saved COCO annotations to {config.AUTO_LABELS_COCO_PATH}"
    )


if __name__ == "__main__":
    main()
