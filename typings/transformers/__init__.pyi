from collections.abc import Iterator, Mapping
from typing import Self

import PIL.Image
import torch

from transformers.utils import ModelOutput

class BatchFeature(Mapping[str, torch.Tensor]):
    input_ids: torch.Tensor
    def to(self, device: str) -> BatchFeature: ...
    def __getitem__(self, key: str) -> torch.Tensor: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

class SamImageSegmentationOutput(ModelOutput):
    pred_masks: torch.Tensor
    iou_scores: torch.Tensor

class AutoProcessor:
    @classmethod
    def from_pretrained(cls, model_identifier: str) -> Self: ...
    def __call__(
        self,
        *,
        images: PIL.Image.Image,
        text: str,
        return_tensors: str,
    ) -> BatchFeature: ...
    def post_process_grounded_object_detection(
        self,
        outputs: ModelOutput,
        input_ids: torch.Tensor,
        *,
        threshold: float,
        text_threshold: float,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]: ...

class AutoModelForZeroShotObjectDetection:
    @classmethod
    def from_pretrained(cls, model_identifier: str) -> Self: ...
    def to(self, device: str) -> Self: ...
    def eval(self) -> Self: ...
    def __call__(self, **inputs: torch.Tensor) -> ModelOutput: ...

class SamProcessor:
    @classmethod
    def from_pretrained(cls, model_identifier: str) -> Self: ...
    def __call__(
        self,
        images: PIL.Image.Image,
        *,
        input_boxes: list[list[list[float]]],
        return_tensors: str,
    ) -> BatchFeature: ...
    def post_process_masks(
        self,
        masks: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
    ) -> list[torch.Tensor]: ...

class SamModel:
    @classmethod
    def from_pretrained(cls, model_identifier: str) -> Self: ...
    def to(self, device: str) -> Self: ...
    def eval(self) -> Self: ...
    def __call__(
        self,
        *,
        multimask_output: bool,
        **inputs: torch.Tensor,
    ) -> SamImageSegmentationOutput: ...
