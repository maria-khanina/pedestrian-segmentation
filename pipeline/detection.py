import PIL.Image
import torch
import transformers
import transformers.utils

from pipeline.coco_types import BoundingBox

_DetectionModelOutput = tuple[transformers.utils.ModelOutput, torch.Tensor]


class OpenVocabularyDetector:
    def __init__(self, model_identifier: str, device: str) -> None:
        self.device = device
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_identifier
        )
        model_class = transformers.AutoModelForZeroShotObjectDetection
        model = model_class.from_pretrained(model_identifier)
        self.model = model.to(device).eval()

    def _run_model(
        self,
        image: PIL.Image.Image,
        text_prompt: str,
    ) -> _DetectionModelOutput:
        model_inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)
        return model_outputs, model_inputs.input_ids

    def _build_bounding_box(
        self,
        raw_box: torch.Tensor,
        confidence: float,
    ) -> BoundingBox:
        x_min, y_min, x_max, y_max = (float(value) for value in raw_box)
        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            confidence=confidence,
        )

    def detect_boxes(
        self,
        image: PIL.Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[BoundingBox]:
        model_outputs, input_ids = self._run_model(image, text_prompt)
        image_height_width: list[tuple[int, int]] = [(image.height, image.width)]
        processed_results = self.processor.post_process_grounded_object_detection(
            model_outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=image_height_width,
        )
        single_image_result = processed_results[0]
        return [
            self._build_bounding_box(raw_box, float(score))
            for raw_box, score in zip(
                single_image_result["boxes"],
                single_image_result["scores"],
            )
        ]
