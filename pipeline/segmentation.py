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
    ) -> transformers.utils.ModelOutput:
        model_inputs = self.processor(
            image,
            input_boxes=[minmax_boxes],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            model_outputs = self.model(
                **model_inputs,
                multimask_output=False,
            )
        post_process = self.processor.image_processor.post_process_masks
        processed_masks = post_process(
            model_outputs.pred_masks.cpu(),
            model_inputs["original_sizes"].cpu(),
            model_inputs["reshaped_input_sizes"].cpu(),
        )
        return processed_masks[0]

    def segment_from_boxes(
        self,
        image: PIL.Image.Image,
        bounding_boxes: list[BoundingBox],
    ) -> list[numpy.ndarray]:
        if len(bounding_boxes) == 0:
            return []
        minmax_boxes = self._bounding_boxes_to_minmax_lists(bounding_boxes)
        single_image_masks = self._run_model(image, minmax_boxes)
        single_image_masks = single_image_masks.squeeze(1).numpy()
        binary_masks: list[numpy.ndarray] = []
        for mask_index in range(single_image_masks.shape[0]):
            binary_masks.append(
                single_image_masks[mask_index].astype(numpy.uint8)
            )
        return binary_masks
