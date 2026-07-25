import cv2
import numpy

from pipeline.coco_types import CocoAnnotationEntry


def _flatten_contour_to_polygon(
    approximated_contour: numpy.ndarray,
) -> list[float]:
    flattened_polygon: list[float] = []
    for point in approximated_contour:
        flattened_polygon.append(float(point[0][0]))
        flattened_polygon.append(float(point[0][1]))
    return flattened_polygon


def convert_mask_to_polygons(
    binary_mask: numpy.ndarray,
    minimum_contour_area: float,
    epsilon_factor: float,
) -> list[list[float]]:
    contours, _ = cv2.findContours(
        binary_mask.astype(numpy.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons: list[list[float]] = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < minimum_contour_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(approximated_contour) < 3:
            continue
        polygons.append(_flatten_contour_to_polygon(approximated_contour))
    return polygons


def convert_polygon_to_point_array(
    polygon: list[float],
    point_dtype: numpy.dtype,
    round_coordinates: bool,
) -> numpy.ndarray:
    number_of_points = len(polygon) // 2
    point_array = numpy.zeros((number_of_points, 2), dtype=point_dtype)
    for point_index in range(number_of_points):
        x_value = polygon[point_index * 2]
        y_value = polygon[point_index * 2 + 1]
        if round_coordinates:
            x_value = round(x_value)
            y_value = round(y_value)
        point_array[point_index, 0] = x_value
        point_array[point_index, 1] = y_value
    return point_array


def rasterize_annotations_to_mask(
    annotations: list[CocoAnnotationEntry],
    image_height: int,
    image_width: int,
) -> numpy.ndarray:
    mask = numpy.zeros((image_height, image_width), dtype=numpy.uint8)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            point_array = convert_polygon_to_point_array(
                polygon,
                numpy.int32,
                True,
            )
            cv2.fillPoly(mask, [point_array], 1)
    return mask
