import torch

class Unet(torch.nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_weights: str,
        in_channels: int,
        classes: int,
    ) -> None: ...
