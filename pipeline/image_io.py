import cv2
import numpy

from pipeline.array_types import ImageArray


def read_color_image(image_path: str) -> ImageArray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    return numpy.asarray(image, dtype=numpy.uint8)


def read_grayscale_image(image_path: str) -> ImageArray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    return numpy.asarray(image, dtype=numpy.uint8)


def convert_bgr_to_rgb(image: ImageArray) -> ImageArray:
    return numpy.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=numpy.uint8)


def convert_rgb_to_bgr(image: ImageArray) -> ImageArray:
    return numpy.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), dtype=numpy.uint8)


def resize_image(
    image: ImageArray,
    width: int,
    height: int,
    interpolation: int,
) -> ImageArray:
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    return numpy.asarray(resized, dtype=numpy.uint8)


def write_image(image_path: str, image: ImageArray) -> None:
    cv2.imwrite(image_path, image)
