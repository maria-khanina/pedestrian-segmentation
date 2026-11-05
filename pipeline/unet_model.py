from typing import Protocol, cast

import segmentation_models_pytorch
import torch
import torch.serialization

from pipeline import segmentation_settings as settings

ModelState = dict[str, torch.Tensor]


class _StateSerialization(Protocol):
    def load(
        self,
        checkpoint_path: str,
        /,
        *,
        map_location: str,
        weights_only: bool,
    ) -> ModelState: ...

    def save(self, model_state: ModelState, checkpoint_path: str, /) -> None: ...


_SERIALIZATION = cast(_StateSerialization, torch.serialization)


def build_unet_model() -> torch.nn.Module:
    return segmentation_models_pytorch.Unet(
        encoder_name=settings.UNET_ENCODER_NAME,
        encoder_weights=settings.UNET_ENCODER_WEIGHTS,
        in_channels=settings.UNET_INPUT_CHANNELS,
        classes=settings.UNET_OUTPUT_CLASSES,
    )


def load_model_state(checkpoint_path: str, device: str) -> ModelState:
    return _SERIALIZATION.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )


def save_model_state(model_state: ModelState, checkpoint_path: str) -> None:
    _SERIALIZATION.save(model_state, checkpoint_path)
