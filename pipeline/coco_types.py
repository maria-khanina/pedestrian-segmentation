from typing import TypedDict


class BoundingBox(TypedDict):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class CocoImageEntry(TypedDict):
    id: int
    file_name: str
    width: int
    height: int


class CocoAnnotationEntry(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]]
    area: float
    bbox: list[float]
    iscrowd: int


class CocoRleSegmentation(TypedDict):
    counts: str | list[int]
    size: list[int]


class CvatAnnotationEntry(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float]] | CocoRleSegmentation
    area: float
    bbox: list[float]
    iscrowd: int


class CocoCategoryEntry(TypedDict):
    id: int
    name: str
    supercategory: str


class CocoDataset(TypedDict):
    images: list[CocoImageEntry]
    annotations: list[CocoAnnotationEntry]
    categories: list[CocoCategoryEntry]


class CvatDataset(TypedDict):
    images: list[CocoImageEntry]
    annotations: list[CvatAnnotationEntry]
    categories: list[CocoCategoryEntry]


AnnotationsByImageId = dict[int, list[CocoAnnotationEntry]]
