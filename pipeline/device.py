import torch


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
