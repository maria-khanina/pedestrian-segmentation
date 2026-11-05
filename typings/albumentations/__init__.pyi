import numpy
import numpy.typing

class BasicTransform: ...

class LongestMaxSize(BasicTransform):
    def __init__(self, *, max_size: int) -> None: ...

class PadIfNeeded(BasicTransform):
    def __init__(self, *, min_height: int, min_width: int, position: str) -> None: ...

class HorizontalFlip(BasicTransform):
    def __init__(self, *, p: float) -> None: ...

class Affine(BasicTransform):
    def __init__(
        self,
        *,
        scale: tuple[float, float],
        translate_percent: tuple[float, float],
        rotate: tuple[float, float],
        p: float,
    ) -> None: ...

class RandomBrightnessContrast(BasicTransform):
    def __init__(
        self,
        *,
        brightness_limit: float,
        contrast_limit: float,
        p: float,
    ) -> None: ...

class GaussNoise(BasicTransform):
    def __init__(self, *, std_range: tuple[float, float], p: float) -> None: ...

class Compose:
    def __init__(self, transforms: list[BasicTransform]) -> None: ...
    def __call__(
        self,
        *,
        image: numpy.typing.NDArray[numpy.uint8],
        mask: numpy.typing.NDArray[numpy.uint8],
    ) -> dict[str, numpy.typing.NDArray[numpy.uint8]]: ...
