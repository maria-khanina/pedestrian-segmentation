from collections.abc import Mapping

import numpy
import numpy.typing

def frPyObjects(
    pyobj: Mapping[str, object],
    h: int,
    w: int,
) -> dict[str, object]: ...
def decode(rleObjs: Mapping[str, object]) -> numpy.typing.NDArray[numpy.uint8]: ...
