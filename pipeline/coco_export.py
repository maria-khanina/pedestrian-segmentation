import json
import os

from pipeline.coco_types import (
    CocoAnnotationEntry,
    CocoCategoryEntry,
    CocoDataset,
    CocoImageEntry,
)


def _compute_polygon_area(polygon: list[float]) -> float:
    signed_area_sum = 0.0
    number_of_points = len(polygon) // 2
    for point_index in range(number_of_points):
        current_x = polygon[point_index * 2]
        current_y = polygon[point_index * 2 + 1]
        next_index = (point_index + 1) % number_of_points
        next_x = polygon[next_index * 2]
        next_y = polygon[next_index * 2 + 1]
        signed_area_sum += (current_x * next_y - next_x * current_y)
    return abs(signed_area_sum) / 2.0


def compute_total_polygons_area(polygons: list[list[float]]) -> float:
    total_area = 0.0
    for polygon in polygons:
        total_area += _compute_polygon_area(polygon)
    return total_area


def compute_bounding_box_from_polygons(
    polygons: list[list[float]],
) -> list[float]:
    all_x_coordinates: list[float] = []
    all_y_coordinates: list[float] = []
    for polygon in polygons:
        all_x_coordinates.extend(polygon[0::2])
        all_y_coordinates.extend(polygon[1::2])
    minimum_x = min(all_x_coordinates)
    minimum_y = min(all_y_coordinates)
    maximum_x = max(all_x_coordinates)
    maximum_y = max(all_y_coordinates)
    return [
        minimum_x,
        minimum_y,
        maximum_x - minimum_x,
        maximum_y - minimum_y,
    ]


class CocoDatasetBuilder:
    def __init__(
        self,
        category_id: int,
        category_name: str,
        supercategory_name: str,
    ) -> None:
        self.images: list[CocoImageEntry] = []
        self.annotations: list[CocoAnnotationEntry] = []
        self.category = CocoCategoryEntry(
            id=category_id,
            name=category_name,
            supercategory=supercategory_name,
        )
        self.next_image_id = 1
        self.next_annotation_id = 1

    def add_image(self, file_name: str, width: int, height: int) -> int:
        image_id = self.next_image_id
        self.images.append(
            CocoImageEntry(
                id=image_id,
                file_name=file_name,
                width=width,
                height=height,
            )
        )
        self.next_image_id += 1
        return image_id

    def add_annotation(
        self,
        image_id: int,
        polygons: list[list[float]],
    ) -> None:
        annotation_id = self.next_annotation_id
        self.annotations.append(
            CocoAnnotationEntry(
                id=annotation_id,
                image_id=image_id,
                category_id=self.category["id"],
                segmentation=polygons,
                area=compute_total_polygons_area(polygons),
                bbox=compute_bounding_box_from_polygons(polygons),
                iscrowd=0,
            )
        )
        self.next_annotation_id += 1

    def build(self) -> CocoDataset:
        return CocoDataset(
            images=self.images,
            annotations=self.annotations,
            categories=[self.category],
        )

    def save_to_file(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as output_file:
            json.dump(self.build(), output_file, indent=2)
