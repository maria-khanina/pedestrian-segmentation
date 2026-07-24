import segmentation_models_pytorch
import torch.nn

from pipeline import config


def build_unet_model() -> torch.nn.Module:
    return segmentation_models_pytorch.Unet(
        encoder_name=config.UNET_ENCODER_NAME,
        encoder_weights=config.UNET_ENCODER_WEIGHTS,
        in_channels=config.UNET_INPUT_CHANNELS,
        classes=config.UNET_OUTPUT_CLASSES,
    )
