import logging
import os
import shutil
import zipfile

import requests

from pipeline import segmentation_settings as settings
from pipeline.logging_setup import configure_logging

_logger = logging.getLogger(__name__)


def _download_dataset_archive(destination_path: str) -> None:
    response = requests.get(
        settings.DATASET_DOWNLOAD_URL,
        stream=True,
        timeout=settings.DOWNLOAD_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    with open(destination_path, "wb") as archive_file:
        for chunk in response.iter_content(
            chunk_size=settings.DOWNLOAD_CHUNK_SIZE_IN_BYTES
        ):
            archive_file.write(chunk)


def _extract_dataset_archive(
    archive_path: str,
    extraction_directory: str,
) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive_file:
        archive_file.extractall(extraction_directory)


def _move_extracted_images_to_raw_directory(
    extraction_directory: str,
    raw_images_directory: str,
) -> None:
    source_images_directory = os.path.join(
        extraction_directory,
        settings.DATASET_ARCHIVE_ROOT_DIRECTORY_NAME,
        settings.DATASET_IMAGES_SUBDIRECTORY_NAME,
    )
    image_file_names = sorted(os.listdir(source_images_directory))
    for image_file_name in image_file_names:
        source_path = os.path.join(source_images_directory, image_file_name)
        destination_path = os.path.join(raw_images_directory, image_file_name)
        shutil.copyfile(source_path, destination_path)


def _clean_up_intermediate_files(
    archive_path: str,
    extraction_directory: str,
) -> None:
    archive_root_directory = os.path.join(
        extraction_directory,
        settings.DATASET_ARCHIVE_ROOT_DIRECTORY_NAME,
    )
    os.remove(archive_path)
    shutil.rmtree(archive_root_directory)


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    os.makedirs(settings.RAW_DIRECTORY, exist_ok=True)
    os.makedirs(settings.RAW_IMAGES_DIRECTORY, exist_ok=True)
    _logger.info("Downloading dataset archive...")
    _download_dataset_archive(settings.DATASET_ARCHIVE_PATH)
    _logger.info("Extracting archive...")
    _extract_dataset_archive(
        settings.DATASET_ARCHIVE_PATH,
        settings.RAW_DIRECTORY,
    )
    _move_extracted_images_to_raw_directory(
        settings.RAW_DIRECTORY,
        settings.RAW_IMAGES_DIRECTORY,
    )
    _clean_up_intermediate_files(
        settings.DATASET_ARCHIVE_PATH,
        settings.RAW_DIRECTORY,
    )
    _logger.info(f"Saved images to {settings.RAW_IMAGES_DIRECTORY}")


if __name__ == "__main__":
    main()
