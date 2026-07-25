import json

from pipeline.coco_types import AnnotationsByImageId, CocoDataset


def load_coco_dataset(coco_path: str) -> CocoDataset:
    with open(coco_path, "r") as coco_file:
        coco_dataset: CocoDataset = json.load(coco_file)
    return coco_dataset


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
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(annotation)
    return annotations_by_image_id
