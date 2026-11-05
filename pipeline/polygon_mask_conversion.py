import cv2
import numpy

from pipeline import segmentation_settings as settings
from pipeline.array_types import ContourArray, FloatArray, ImageArray
from pipeline.coco_types import CocoAnnotationEntry


def _flatten_contour_to_polygon(approximated_contour: ContourArray) -> list[float]:
    flattened_polygon: list[float] = []
    for point in approximated_contour:
        flattened_polygon.append(float(point[0][0]))
        flattened_polygon.append(float(point[0][1]))
    return flattened_polygon


def _simplify_contour(
    contour: ContourArray,
    epsilon_factor: float,
) -> ContourArray:
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approximated = cv2.approxPolyDP(contour, epsilon, True)
    return numpy.asarray(approximated, dtype=numpy.int32)


def _find_external_contours(binary_mask: ImageArray) -> list[ContourArray]:
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [numpy.asarray(contour, dtype=numpy.int32) for contour in contours]


def convert_mask_to_polygons(
    binary_mask: ImageArray,
    minimum_contour_area: float,
    epsilon_factor: float,
) -> list[list[float]]:
    polygons: list[list[float]] = []
    for contour in _find_external_contours(binary_mask):
        if cv2.contourArea(contour) < minimum_contour_area:
            continue
        approximated_contour = _simplify_contour(contour, epsilon_factor)
        if len(approximated_contour) < settings.MINIMUM_POLYGON_POINT_COUNT:
            continue
        polygons.append(_flatten_contour_to_polygon(approximated_contour))
    return polygons


def convert_polygon_to_point_array(polygon: list[float]) -> FloatArray:
    coordinates = numpy.asarray(polygon, dtype=numpy.float32)
    return coordinates.reshape(-1, 2)


def rasterize_annotations_to_mask(
    annotations: list[CocoAnnotationEntry],
    image_height: int,
    image_width: int,
) -> ImageArray:
    mask = numpy.zeros((image_height, image_width), dtype=numpy.uint8)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            point_array = convert_polygon_to_point_array(polygon)
            rounded_points = numpy.round(point_array).astype(numpy.int32)
            cv2.fillPoly(mask, [rounded_points], 1)
    return mask
