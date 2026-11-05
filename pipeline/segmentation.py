import numpy
import PIL.Image
import torch
import transformers

from pipeline.array_types import ImageArray
from pipeline.coco_types import BoundingBox

_CandidateMasks = tuple[torch.Tensor, torch.Tensor]


class BoxPromptedSegmentor:

    def __init__(self, model_identifier: str, device: str) -> None:
        self.device = device
        self.processor = transformers.SamProcessor.from_pretrained(
            model_identifier
        )
        model = transformers.SamModel.from_pretrained(model_identifier)
        self.model = model.to(device).eval()

    def _bounding_boxes_to_minmax_lists(
        self,
        bounding_boxes: list[BoundingBox],
    ) -> list[list[float]]:
        return [
            [
                bounding_box["x_min"],
                bounding_box["y_min"],
                bounding_box["x_max"],
                bounding_box["y_max"],
            ]
            for bounding_box in bounding_boxes
        ]

    def _run_model(
        self,
        image: PIL.Image.Image,
        minmax_boxes: list[list[float]],
    ) -> _CandidateMasks:
        model_inputs = self.processor(
            image,
            input_boxes=[minmax_boxes],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            model_outputs = self.model(
                **model_inputs,
                multimask_output=True,
            )
        processed_masks = self.processor.post_process_masks(
            model_outputs.pred_masks.cpu(),
            model_inputs["original_sizes"].cpu(),
            model_inputs["reshaped_input_sizes"].cpu(),
        )
        return processed_masks[0], model_outputs.iou_scores.cpu()[0]

    def segment_from_boxes(
        self,
        image: PIL.Image.Image,
        bounding_boxes: list[BoundingBox],
    ) -> list[ImageArray]:
        if len(bounding_boxes) == 0:
            return []
        minmax_boxes = self._bounding_boxes_to_minmax_lists(bounding_boxes)
        candidate_masks, iou_scores = self._run_model(image, minmax_boxes)
        best_mask_indices = iou_scores.argmax(dim=1)
        return [
            numpy.asarray(box_masks[best_index], dtype=numpy.uint8)
            for box_masks, best_index in zip(candidate_masks, best_mask_indices)
        ]
