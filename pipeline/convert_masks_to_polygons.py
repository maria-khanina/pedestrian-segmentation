import logging
import sys

import numpy
import pycocotools.mask

from pipeline import config
from pipeline.coco_export import (
    compute_bounding_box_from_polygons,
    compute_total_polygons_area,
)
from pipeline.coco_io import load_coco_dataset, save_coco_dataset
from pipeline.coco_types import (
    CocoAnnotationEntry,
    CocoDataset,
    CocoRleSegmentation,
)
from pipeline.logging_setup import configure_logging
from pipeline.polygon_mask_conversion import convert_mask_to_polygons

_logger = logging.getLogger(__name__)

_USAGE_MESSAGE = (
    "Usage: python -m pipeline.convert_masks_to_polygons "
    "<input_coco_json_path> <output_coco_json_path>"
)

_ImageDimensionsById = dict[int, tuple[int, int]]


def _is_rle_segmentation(segmentation: object) -> bool:
    return isinstance(segmentation, dict)


def _convert_uncompressed_rle_to_compressed(
    rle_segmentation: CocoRleSegmentation,
    image_height: int,
    image_width: int,
) -> CocoRleSegmentation:
    return pycocotools.mask.frPyObjects(
        rle_segmentation,
        image_height,
        image_width,
    )


def _normalize_compressed_rle_counts(
    rle_segmentation: CocoRleSegmentation,
) -> CocoRleSegmentation:
    normalized_rle: CocoRleSegmentation = dict(rle_segmentation)
    counts = normalized_rle["counts"]
    if isinstance(counts, str):
        normalized_rle["counts"] = counts.encode("utf-8")
    return normalized_rle


def _decode_rle_to_binary_mask(
    rle_segmentation: CocoRleSegmentation,
    image_height: int,
    image_width: int,
) -> numpy.ndarray:
    counts = rle_segmentation["counts"]
    if isinstance(counts, list):
        compressed_rle = _convert_uncompressed_rle_to_compressed(
            rle_segmentation,
            image_height,
            image_width,
        )
    else:
        compressed_rle = _normalize_compressed_rle_counts(
            rle_segmentation
        )
    decoded_mask = pycocotools.mask.decode(compressed_rle)
    return decoded_mask.astype(numpy.uint8)


def _convert_annotation_to_polygons(
    annotation: CocoAnnotationEntry,
    image_height: int,
    image_width: int,
) -> CocoAnnotationEntry | None:
    rle_segmentation: CocoRleSegmentation = annotation["segmentation"]
    binary_mask = _decode_rle_to_binary_mask(
        rle_segmentation,
        image_height,
        image_width,
    )
    polygons = convert_mask_to_polygons(
        binary_mask,
        config.MINIMUM_CONTOUR_AREA_IN_PIXELS,
        config.POLYGON_APPROXIMATION_EPSILON_FACTOR,
    )
    if len(polygons) == 0:
        return None
    converted_annotation: CocoAnnotationEntry = dict(annotation)
    converted_annotation["segmentation"] = polygons
    converted_annotation["area"] = compute_total_polygons_area(polygons)
    converted_annotation["bbox"] = compute_bounding_box_from_polygons(
        polygons
    )
    return converted_annotation


def _build_image_dimensions_by_id(
    coco_dataset: CocoDataset,
) -> _ImageDimensionsById:
    dimensions_by_image_id: _ImageDimensionsById = {}
    for image in coco_dataset["images"]:
        dimensions_by_image_id[image["id"]] = (
            image["height"],
            image["width"],
        )
    return dimensions_by_image_id


def _convert_dataset_masks_to_polygons(
    coco_dataset: CocoDataset,
) -> tuple[CocoDataset, int, int]:
    dimensions_by_image_id = _build_image_dimensions_by_id(coco_dataset)
    converted_count = 0
    dropped_count = 0
    converted_annotations: list[CocoAnnotationEntry] = []
    for annotation in coco_dataset["annotations"]:
        if _is_rle_segmentation(annotation["segmentation"]):
            image_height, image_width = dimensions_by_image_id[
                annotation["image_id"]
            ]
            converted_annotation = _convert_annotation_to_polygons(
                annotation,
                image_height,
                image_width,
            )
            if converted_annotation is None:
                dropped_count += 1
                continue
            converted_annotations.append(converted_annotation)
            converted_count += 1
        else:
            converted_annotations.append(annotation)
    converted_dataset: CocoDataset = dict(coco_dataset)
    converted_dataset["annotations"] = converted_annotations
    return converted_dataset, converted_count, dropped_count


def main() -> None:
    configure_logging(logging.INFO, config.PIPELINE_LOG_PATH)
    if len(sys.argv) != config.CLI_SCRIPT_ARGUMENT_COUNT:
        _logger.error(_USAGE_MESSAGE)
        sys.exit(1)
    input_coco_path = sys.argv[1]
    output_coco_path = sys.argv[2]
    coco_dataset = load_coco_dataset(input_coco_path)
    converted_dataset, converted_count, dropped_count = (
        _convert_dataset_masks_to_polygons(coco_dataset)
    )
    save_coco_dataset(converted_dataset, output_coco_path)
    _logger.info(
        f"Converted {converted_count} mask annotations to polygons, "
        f"saved to {output_coco_path}"
    )
    if dropped_count > 0:
        _logger.warning(
            f"Dropped {dropped_count} mask annotations whose "
            "decoded region was too small to form a valid polygon "
            "(below MINIMUM_CONTOUR_AREA_IN_PIXELS) - check those "
            "images manually"
        )


if __name__ == "__main__":
    main()
