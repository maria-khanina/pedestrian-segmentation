import json
import os

import albumentations
import cv2
import numpy

from pipeline import config
from pipeline.coco_types import (
    AnnotationsByImageId,
    CocoDataset,
    CocoImageEntry,
)
from pipeline.polygon_mask_conversion import (
    rasterize_annotations_to_mask,
)

MASK_PIXEL_VALUE_FOR_PEDESTRIAN = 255


def load_reviewed_dataset(
    coco_path: str,
) -> tuple[list[CocoImageEntry], AnnotationsByImageId]:
    with open(coco_path, "r") as coco_file:
        coco_dataset: CocoDataset = json.load(coco_file)
    annotations_by_image_id: AnnotationsByImageId = {}
    for annotation in coco_dataset["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(annotation)
    return coco_dataset["images"], annotations_by_image_id


def build_augmentation_pipeline() -> albumentations.Compose:
    horizontal_flip_probability = (
        config.AUGMENTATION_HORIZONTAL_FLIP_PROBABILITY
    )
    affine_probability = config.AUGMENTATION_AFFINE_PROBABILITY
    affine_scale_range = config.AUGMENTATION_AFFINE_SCALE_RANGE
    affine_translate_percent_range = (
        config.AUGMENTATION_AFFINE_TRANSLATE_PERCENT_RANGE
    )
    affine_rotate_degrees_range = (
        config.AUGMENTATION_AFFINE_ROTATE_DEGREES_RANGE
    )
    brightness_contrast_probability = (
        config.AUGMENTATION_BRIGHTNESS_CONTRAST_PROBABILITY
    )
    brightness_limit = config.AUGMENTATION_BRIGHTNESS_LIMIT
    contrast_limit = config.AUGMENTATION_CONTRAST_LIMIT
    gaussian_noise_probability = (
        config.AUGMENTATION_GAUSSIAN_NOISE_PROBABILITY
    )
    gaussian_noise_standard_deviation_range = (
        config
        .AUGMENTATION_GAUSSIAN_NOISE_STANDARD_DEVIATION_RANGE
    )
    return albumentations.Compose(
        [
            albumentations.Resize(
                height=config.AUGMENTED_IMAGE_HEIGHT,
                width=config.AUGMENTED_IMAGE_WIDTH,
            ),
            albumentations.HorizontalFlip(
                p=horizontal_flip_probability
            ),
            albumentations.Affine(
                scale=affine_scale_range,
                translate_percent=affine_translate_percent_range,
                rotate=affine_rotate_degrees_range,
                p=affine_probability,
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_probability,
            ),
            albumentations.GaussNoise(
                std_range=(
                    gaussian_noise_standard_deviation_range
                ),
                p=gaussian_noise_probability,
            ),
        ]
    )


def load_image_as_rgb(image_path: str) -> numpy.ndarray:
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_augmented_pair(
    image: numpy.ndarray,
    mask: numpy.ndarray,
    output_stem: str,
    copy_index: int,
) -> None:
    image_output_path = os.path.join(
        config.AUGMENTED_IMAGES_DIRECTORY,
        f"{output_stem}_{copy_index}.png",
    )
    mask_output_path = os.path.join(
        config.AUGMENTED_MASKS_DIRECTORY,
        f"{output_stem}_{copy_index}.png",
    )
    image_as_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_output_path, image_as_bgr)
    cv2.imwrite(
        mask_output_path,
        mask * MASK_PIXEL_VALUE_FOR_PEDESTRIAN,
    )


def augment_single_image(
    image_entry: CocoImageEntry,
    annotations_by_image_id: AnnotationsByImageId,
    augmentation_pipeline: albumentations.Compose,
) -> None:
    image_path = os.path.join(
        config.REVIEWED_IMAGES_DIRECTORY,
        image_entry["file_name"],
    )
    image = load_image_as_rgb(image_path)
    annotations = annotations_by_image_id.get(image_entry["id"], [])
    mask = rasterize_annotations_to_mask(
        annotations,
        image_entry["height"],
        image_entry["width"],
    )
    output_stem = os.path.splitext(image_entry["file_name"])[0]
    for copy_index in range(config.AUGMENTED_COPIES_PER_IMAGE):
        augmented = augmentation_pipeline(image=image, mask=mask)
        save_augmented_pair(
            augmented["image"],
            augmented["mask"],
            output_stem,
            copy_index,
        )


def main() -> None:
    os.makedirs(config.AUGMENTED_IMAGES_DIRECTORY, exist_ok=True)
    os.makedirs(config.AUGMENTED_MASKS_DIRECTORY, exist_ok=True)
    images, annotations_by_image_id = load_reviewed_dataset(
        config.REVIEWED_COCO_PATH
    )
    augmentation_pipeline = build_augmentation_pipeline()
    for image_entry in images:
        print(f"Augmenting {image_entry['file_name']}")
        augment_single_image(
            image_entry,
            annotations_by_image_id,
            augmentation_pipeline,
        )
    print(f"Saved augmented dataset to {config.AUGMENTED_DIRECTORY}")


if __name__ == "__main__":
    main()
