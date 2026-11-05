import logging
import os

import PIL.Image

from pipeline import segmentation_settings as settings
from pipeline.coco_export import CocoDatasetBuilder
from pipeline.detection import OpenVocabularyDetector
from pipeline.device import select_device
from pipeline.logging_setup import configure_logging
from pipeline.polygon_mask_conversion import convert_mask_to_polygons
from pipeline.segmentation import BoxPromptedSegmentor

_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")

_logger = logging.getLogger(__name__)


def _list_image_file_names(directory: str) -> list[str]:
    return [
        entry
        for entry in sorted(os.listdir(directory))
        if entry.lower().endswith(_IMAGE_FILE_EXTENSIONS)
    ]


def _label_single_image(
    image_path: str,
    detector: OpenVocabularyDetector,
    segmentor: BoxPromptedSegmentor,
    dataset_builder: CocoDatasetBuilder,
    file_name: str,
) -> None:
    image = PIL.Image.open(image_path).convert("RGB")
    bounding_boxes = detector.detect_boxes(
        image,
        settings.DETECTION_TEXT_PROMPT,
        settings.DETECTION_BOX_THRESHOLD,
        settings.DETECTION_TEXT_THRESHOLD,
    )
    image_id = dataset_builder.add_image(
        file_name,
        image.width,
        image.height,
    )
    for binary_mask in segmentor.segment_from_boxes(image, bounding_boxes):
        polygons = convert_mask_to_polygons(
            binary_mask,
            settings.MINIMUM_CONTOUR_AREA_IN_PIXELS,
            settings.POLYGON_APPROXIMATION_EPSILON_FACTOR,
        )
        if polygons:
            dataset_builder.add_annotation(image_id, polygons)


def _load_models() -> tuple[OpenVocabularyDetector, BoxPromptedSegmentor]:
    device = select_device()
    _logger.info(f"Using device: {device}")
    detector = OpenVocabularyDetector(
        settings.DETECTOR_MODEL_IDENTIFIER,
        device,
    )
    segmentor = BoxPromptedSegmentor(
        settings.SEGMENTOR_MODEL_IDENTIFIER,
        device,
    )
    return detector, segmentor


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    detector, segmentor = _load_models()
    dataset_builder = CocoDatasetBuilder(
        settings.PEDESTRIAN_CATEGORY_ID,
        settings.PEDESTRIAN_CATEGORY_NAME,
        settings.PEDESTRIAN_SUPERCATEGORY_NAME,
    )
    image_file_names = _list_image_file_names(settings.RAW_IMAGES_DIRECTORY)
    for file_name in image_file_names:
        image_path = os.path.join(settings.RAW_IMAGES_DIRECTORY, file_name)
        _logger.info(f"Labeling {file_name}")
        _label_single_image(
            image_path,
            detector,
            segmentor,
            dataset_builder,
            file_name,
        )
    dataset_builder.save_to_file(settings.AUTO_LABELS_COCO_PATH)
    _logger.info(
        f"Saved COCO annotations to {settings.AUTO_LABELS_COCO_PATH}"
    )


if __name__ == "__main__":
    main()
