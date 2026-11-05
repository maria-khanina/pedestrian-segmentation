import json

from pipeline.coco_types import AnnotationsByImageId, CocoDataset, CvatDataset


def load_coco_dataset(coco_path: str) -> CocoDataset:
    with open(coco_path, "r") as coco_file:
        coco_dataset: CocoDataset = json.load(coco_file)
    return coco_dataset


def load_cvat_export(export_path: str) -> CvatDataset:
    with open(export_path, "r") as export_file:
        cvat_dataset: CvatDataset = json.load(export_file)
    return cvat_dataset


def save_coco_dataset(
    coco_dataset: CocoDataset,
    output_path: str,
) -> None:
    with open(output_path, "w") as output_file:
        json.dump(coco_dataset, output_file, indent=2)


def group_annotations_by_image_id(
    coco_dataset: CocoDataset,
) -> AnnotationsByImageId:
    annotations_by_image_id: AnnotationsByImageId = {}
    for annotation in coco_dataset["annotations"]:
        image_annotations = annotations_by_image_id.setdefault(
            annotation["image_id"],
            [],
        )
        image_annotations.append(annotation)
    return annotations_by_image_id
