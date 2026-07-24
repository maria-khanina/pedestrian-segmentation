import os

import PIL.Image
import torch

from pipeline import config
from pipeline.coco_export import CocoDatasetBuilder
from pipeline.detection import GroundingDinoDetector
from pipeline.polygon_mask_conversion import convert_mask_to_polygons
from pipeline.segmentation import SamSegmentor

IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_image_file_names(directory: str) -> list[str]:
    all_entries = sorted(os.listdir(directory))
    image_file_names: list[str] = []
    for entry in all_entries:
        if entry.lower().endswith(IMAGE_FILE_EXTENSIONS):
            image_file_names.append(entry)
    return image_file_names


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def label_single_image(
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
    device = select_device()
    print(f"Using device: {device}")
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
    image_file_names = list_image_file_names(config.RAW_IMAGES_DIRECTORY)
    for file_name in image_file_names:
        image_path = os.path.join(config.RAW_IMAGES_DIRECTORY, file_name)
        print(f"Labeling {file_name}")
        label_single_image(
            image_path,
            detector,
            segmentor,
            dataset_builder,
            file_name,
        )
    dataset_builder.save_to_file(config.AUTO_LABELS_COCO_PATH)
    print(f"Saved COCO annotations to {config.AUTO_LABELS_COCO_PATH}")


if __name__ == "__main__":
    main()
