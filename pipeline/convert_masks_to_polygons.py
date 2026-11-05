import logging
import sys

import pycocotools.mask

from pipeline import segmentation_settings as settings
from pipeline.array_types import ImageArray
from pipeline.coco_export import (
    compute_bounding_box_from_polygons,
    compute_total_polygons_area,
)
from pipeline.coco_io import load_cvat_export, save_coco_dataset
from pipeline.coco_types import (
    CocoAnnotationEntry,
    CocoDataset,
    CocoRleSegmentation,
    CvatAnnotationEntry,
    CvatDataset,
)
from pipeline.logging_setup import configure_logging
from pipeline.polygon_mask_conversion import convert_mask_to_polygons

_logger = logging.getLogger(__name__)

_USAGE_MESSAGE = (
    "Usage: python -m pipeline.convert_masks_to_polygons "
    "<input_coco_json_path> <output_coco_json_path>"
)

_ImageDimensionsById = dict[int, tuple[int, int]]


def _decode_rle_to_binary_mask(
    rle_segmentation: CocoRleSegmentation,
    image_height: int,
    image_width: int,
) -> ImageArray:
    counts = rle_segmentation["counts"]
    if isinstance(counts, list):
        compressed_rle = pycocotools.mask.frPyObjects(
            rle_segmentation,
            image_height,
            image_width,
        )
    else:
        compressed_rle: dict[str, object] = {
            "size": rle_segmentation["size"],
            "counts": counts.encode("utf-8"),
        }
    return pycocotools.mask.decode(compressed_rle)


def _build_polygon_annotation(
    annotation: CvatAnnotationEntry,
    polygons: list[list[float]],
) -> CocoAnnotationEntry:
    return CocoAnnotationEntry(
        id=annotation["id"],
        image_id=annotation["image_id"],
        category_id=annotation["category_id"],
        segmentation=polygons,
        area=compute_total_polygons_area(polygons),
        bbox=compute_bounding_box_from_polygons(polygons),
        iscrowd=annotation["iscrowd"],
    )


def _convert_rle_annotation(
    annotation: CvatAnnotationEntry,
    rle_segmentation: CocoRleSegmentation,
    image_dimensions: tuple[int, int],
) -> CocoAnnotationEntry | None:
    image_height, image_width = image_dimensions
    binary_mask = _decode_rle_to_binary_mask(
        rle_segmentation,
        image_height,
        image_width,
    )
    polygons = convert_mask_to_polygons(
        binary_mask,
        settings.MINIMUM_CONTOUR_AREA_IN_PIXELS,
        settings.POLYGON_APPROXIMATION_EPSILON_FACTOR,
    )
    if len(polygons) == 0:
        return None
    return _build_polygon_annotation(annotation, polygons)


def _build_image_dimensions_by_id(
    cvat_dataset: CvatDataset,
) -> _ImageDimensionsById:
    dimensions_by_image_id: _ImageDimensionsById = {}
    for image in cvat_dataset["images"]:
        dimensions_by_image_id[image["id"]] = (image["height"], image["width"])
    return dimensions_by_image_id


def _convert_annotation(
    annotation: CvatAnnotationEntry,
    dimensions_by_image_id: _ImageDimensionsById,
) -> CocoAnnotationEntry | None:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return CocoAnnotationEntry(
            id=annotation["id"],
            image_id=annotation["image_id"],
            category_id=annotation["category_id"],
            segmentation=segmentation,
            area=annotation["area"],
            bbox=annotation["bbox"],
            iscrowd=annotation["iscrowd"],
        )
    return _convert_rle_annotation(
        annotation,
        segmentation,
        dimensions_by_image_id[annotation["image_id"]],
    )


def _convert_dataset_masks_to_polygons(
    cvat_dataset: CvatDataset,
) -> tuple[CocoDataset, int]:
    dimensions_by_image_id = _build_image_dimensions_by_id(cvat_dataset)
    converted_annotations: list[CocoAnnotationEntry] = []
    for annotation in cvat_dataset["annotations"]:
        converted = _convert_annotation(annotation, dimensions_by_image_id)
        if converted is not None:
            converted_annotations.append(converted)
    converted_dataset = CocoDataset(
        images=cvat_dataset["images"],
        annotations=converted_annotations,
        categories=cvat_dataset["categories"],
    )
    dropped_count = len(cvat_dataset["annotations"]) - len(converted_annotations)
    return converted_dataset, dropped_count


def _count_mask_annotations(cvat_dataset: CvatDataset) -> int:
    return sum(
        1
        for annotation in cvat_dataset["annotations"]
        if not isinstance(annotation["segmentation"], list)
    )


def _report_conversion(
    converted_count: int,
    dropped_count: int,
    output_coco_path: str,
) -> None:
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


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    if len(sys.argv) != settings.CLI_SCRIPT_ARGUMENT_COUNT:
        _logger.error(_USAGE_MESSAGE)
        sys.exit(1)
    input_coco_path, output_coco_path = sys.argv[1:]
    cvat_dataset = load_cvat_export(input_coco_path)
    converted_dataset, dropped_count = _convert_dataset_masks_to_polygons(
        cvat_dataset,
    )
    save_coco_dataset(converted_dataset, output_coco_path)
    mask_count = _count_mask_annotations(cvat_dataset)
    _report_conversion(mask_count - dropped_count, dropped_count, output_coco_path)


if __name__ == "__main__":
    main()
