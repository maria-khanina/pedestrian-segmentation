import logging
import os
import random

import albumentations

from pipeline import segmentation_settings as settings
from pipeline.array_types import ImageArray
from pipeline.coco_io import group_annotations_by_image_id, load_coco_dataset
from pipeline.coco_types import AnnotationsByImageId, CocoImageEntry
from pipeline.image_io import (
    convert_bgr_to_rgb,
    convert_rgb_to_bgr,
    read_color_image,
    write_image,
)
from pipeline.logging_setup import configure_logging
from pipeline.polygon_mask_conversion import rasterize_annotations_to_mask

_logger = logging.getLogger(__name__)

_ImageSplit = tuple[
    list[CocoImageEntry],
    list[CocoImageEntry],
    list[CocoImageEntry],
]


def _split_images_into_train_validation_test(
    images: list[CocoImageEntry],
) -> _ImageSplit:
    shuffled_images: list[CocoImageEntry] = list(images)
    random.Random(settings.RANDOM_SEED).shuffle(shuffled_images)
    total_count = len(shuffled_images)
    test_count = round(total_count * settings.UNET_TEST_FRACTION)
    validation_count = round(total_count * settings.UNET_VALIDATION_FRACTION)
    test_images = shuffled_images[:test_count]
    validation_images = shuffled_images[
        test_count:test_count + validation_count
    ]
    training_images = shuffled_images[test_count + validation_count:]
    return training_images, validation_images, test_images


def _build_aspect_preserving_resize_steps() -> list[albumentations.BasicTransform]:
    return [
        albumentations.LongestMaxSize(
            max_size=settings.AUGMENTED_IMAGE_HEIGHT,
        ),
        albumentations.PadIfNeeded(
            min_height=settings.AUGMENTED_IMAGE_HEIGHT,
            min_width=settings.AUGMENTED_IMAGE_WIDTH,
            position="center",
        ),
    ]


def _build_randomized_steps() -> list[albumentations.BasicTransform]:
    return [
        albumentations.HorizontalFlip(
            p=settings.AUGMENTATION_HORIZONTAL_FLIP_PROBABILITY,
        ),
        albumentations.Affine(
            scale=settings.AUGMENTATION_AFFINE_SCALE_RANGE,
            translate_percent=settings.AUGMENTATION_AFFINE_TRANSLATE_PERCENT_RANGE,
            rotate=settings.AUGMENTATION_AFFINE_ROTATE_DEGREES_RANGE,
            p=settings.AUGMENTATION_AFFINE_PROBABILITY,
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=settings.AUGMENTATION_BRIGHTNESS_LIMIT,
            contrast_limit=settings.AUGMENTATION_CONTRAST_LIMIT,
            p=settings.AUGMENTATION_BRIGHTNESS_CONTRAST_PROBABILITY,
        ),
        albumentations.GaussNoise(
            std_range=settings.AUGMENTATION_GAUSSIAN_NOISE_STANDARD_DEVIATION_RANGE,
            p=settings.AUGMENTATION_GAUSSIAN_NOISE_PROBABILITY,
        ),
    ]


def _build_augmentation_pipeline() -> albumentations.Compose:
    return albumentations.Compose(
        _build_aspect_preserving_resize_steps() + _build_randomized_steps()
    )


def _build_eval_pipeline() -> albumentations.Compose:
    return albumentations.Compose(_build_aspect_preserving_resize_steps())


def _save_augmented_pair(
    image: ImageArray,
    mask: ImageArray,
    output_file_name: str,
    images_directory: str,
    masks_directory: str,
) -> None:
    write_image(
        os.path.join(images_directory, output_file_name),
        convert_rgb_to_bgr(image),
    )
    write_image(
        os.path.join(masks_directory, output_file_name),
        mask * settings.MASK_PIXEL_VALUE_FOR_PEDESTRIAN,
    )


def _augment_single_image(
    image_entry: CocoImageEntry,
    annotations_by_image_id: AnnotationsByImageId,
    augmentation_pipeline: albumentations.Compose,
    images_directory: str,
    masks_directory: str,
    number_of_copies: int,
) -> None:
    image_path = os.path.join(
        settings.REVIEWED_IMAGES_DIRECTORY,
        image_entry["file_name"],
    )
    image = convert_bgr_to_rgb(read_color_image(image_path))
    mask = rasterize_annotations_to_mask(
        annotations_by_image_id.get(image_entry["id"], []),
        image_entry["height"],
        image_entry["width"],
    )
    output_stem = os.path.splitext(image_entry["file_name"])[0]
    for copy_index in range(number_of_copies):
        augmented = augmentation_pipeline(image=image, mask=mask)
        _save_augmented_pair(
            augmented["image"],
            augmented["mask"],
            f"{output_stem}_{copy_index}.png",
            images_directory,
            masks_directory,
        )


def _augment_split(
    split_name: str,
    image_entries: list[CocoImageEntry],
    annotations_by_image_id: AnnotationsByImageId,
    augmentation_pipeline: albumentations.Compose,
    output_directories: tuple[str, str],
    number_of_copies: int,
) -> None:
    images_directory, masks_directory = output_directories
    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(masks_directory, exist_ok=True)
    for image_entry in image_entries:
        _logger.info(f"Augmenting [{split_name}] {image_entry['file_name']}")
        _augment_single_image(
            image_entry,
            annotations_by_image_id,
            augmentation_pipeline,
            images_directory,
            masks_directory,
            number_of_copies,
        )


def _augment_all_splits(
    image_split: _ImageSplit,
    annotations_by_image_id: AnnotationsByImageId,
) -> None:
    training_images, validation_images, test_images = image_split
    _augment_split(
        "train",
        training_images,
        annotations_by_image_id,
        _build_augmentation_pipeline(),
        (
            settings.AUGMENTED_TRAIN_IMAGES_DIRECTORY,
            settings.AUGMENTED_TRAIN_MASKS_DIRECTORY,
        ),
        settings.AUGMENTED_COPIES_PER_IMAGE,
    )
    _augment_split(
        "val",
        validation_images,
        annotations_by_image_id,
        _build_eval_pipeline(),
        (
            settings.AUGMENTED_VALIDATION_IMAGES_DIRECTORY,
            settings.AUGMENTED_VALIDATION_MASKS_DIRECTORY,
        ),
        1,
    )
    _augment_split(
        "test",
        test_images,
        annotations_by_image_id,
        _build_eval_pipeline(),
        (
            settings.AUGMENTED_TEST_IMAGES_DIRECTORY,
            settings.AUGMENTED_TEST_MASKS_DIRECTORY,
        ),
        1,
    )


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    coco_dataset = load_coco_dataset(settings.REVIEWED_COCO_PATH)
    images = coco_dataset["images"]
    image_split = _split_images_into_train_validation_test(images)
    training_images, validation_images, test_images = image_split
    _logger.info(
        f"Split {len(images)} reviewed images into "
        f"{len(training_images)} train / "
        f"{len(validation_images)} val / "
        f"{len(test_images)} test, by source image "
        f"(so augmented copies of one image never cross splits)"
    )
    _augment_all_splits(image_split, group_annotations_by_image_id(coco_dataset))
    _logger.info(f"Saved augmented dataset to {settings.AUGMENTED_DIRECTORY}")


if __name__ == "__main__":
    main()
