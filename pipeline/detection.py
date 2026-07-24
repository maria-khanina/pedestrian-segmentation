import PIL.Image
import torch
import transformers

from pipeline.coco_types import BoundingBox

DetectionModelOutput = tuple[transformers.utils.ModelOutput, torch.Tensor]


class GroundingDinoDetector:
    def __init__(self, model_identifier: str, device: str) -> None:
        self.device = device
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_identifier
        )
        model_class = (
            transformers.AutoModelForZeroShotObjectDetection
        )
        model = model_class.from_pretrained(model_identifier)
        self.model = model.to(device).eval()

    def _run_model(
        self,
        image: PIL.Image.Image,
        text_prompt: str,
    ) -> DetectionModelOutput:
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
        return BoundingBox(
            x_min=float(raw_box[0]),
            y_min=float(raw_box[1]),
            x_max=float(raw_box[2]),
            y_max=float(raw_box[3]),
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
        image_height_width = [(image.height, image.width)]
        post_process = (
            self.processor.post_process_grounded_object_detection
        )
        processed_results = post_process(
            model_outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=image_height_width,
        )
        single_image_result = processed_results[0]
        bounding_boxes: list[BoundingBox] = []
        detected_boxes = single_image_result["boxes"]
        detected_scores = single_image_result["scores"]
        for detection_index in range(detected_boxes.shape[0]):
            bounding_boxes.append(
                self._build_bounding_box(
                    detected_boxes[detection_index],
                    float(detected_scores[detection_index]),
                )
            )
        return bounding_boxes
