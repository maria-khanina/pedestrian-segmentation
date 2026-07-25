import numpy
import PIL.Image
import torch
import transformers

from pipeline.coco_types import BoundingBox


class SamSegmentor:

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
        minmax_boxes: list[list[float]] = []
        for bounding_box in bounding_boxes:
            minmax_boxes.append(
                [
                    bounding_box["x_min"],
                    bounding_box["y_min"],
                    bounding_box["x_max"],
                    bounding_box["y_max"],
                ]
            )
        return minmax_boxes

    def _run_model(
        self,
        image: PIL.Image.Image,
        minmax_boxes: list[list[float]],
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        post_process = self.processor.image_processor.post_process_masks
        processed_masks = post_process(
            model_outputs.pred_masks.cpu(),
            model_inputs["original_sizes"].cpu(),
            model_inputs["reshaped_input_sizes"].cpu(),
        )
        candidate_masks = processed_masks[0].numpy()
        iou_scores = model_outputs.iou_scores.cpu().numpy()[0]
        return candidate_masks, iou_scores

    def segment_from_boxes(
        self,
        image: PIL.Image.Image,
        bounding_boxes: list[BoundingBox],
    ) -> list[numpy.ndarray]:
        if len(bounding_boxes) == 0:
            return []
        minmax_boxes = self._bounding_boxes_to_minmax_lists(bounding_boxes)
        candidate_masks, iou_scores = self._run_model(image, minmax_boxes)
        binary_masks: list[numpy.ndarray] = []
        for box_index in range(candidate_masks.shape[0]):
            best_mask_index = int(numpy.argmax(iou_scores[box_index]))
            best_mask = candidate_masks[box_index, best_mask_index]
            binary_masks.append(best_mask.astype(numpy.uint8))
        return binary_masks
