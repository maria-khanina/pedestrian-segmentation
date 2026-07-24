import os
import shutil
import zipfile

import requests

from pipeline import config


def download_dataset_archive(destination_path: str) -> None:
    response = requests.get(
        config.DATASET_DOWNLOAD_URL,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    with open(destination_path, "wb") as archive_file:
        for chunk in response.iter_content(
            chunk_size=config.DOWNLOAD_CHUNK_SIZE_IN_BYTES
        ):
            archive_file.write(chunk)


def extract_dataset_archive(
    archive_path: str,
    extraction_directory: str,
) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive_file:
        archive_file.extractall(extraction_directory)


def move_extracted_images_to_raw_directory(
    extraction_directory: str,
    raw_images_directory: str,
) -> None:
    source_images_directory = os.path.join(
        extraction_directory,
        config.DATASET_ARCHIVE_ROOT_DIRECTORY_NAME,
        config.DATASET_IMAGES_SUBDIRECTORY_NAME,
    )
    image_file_names = sorted(os.listdir(source_images_directory))
    for image_file_name in image_file_names:
        source_path = os.path.join(source_images_directory, image_file_name)
        destination_path = os.path.join(raw_images_directory, image_file_name)
        shutil.copyfile(source_path, destination_path)


def clean_up_intermediate_files(
    archive_path: str,
    extraction_directory: str,
) -> None:
    archive_root_directory = os.path.join(
        extraction_directory,
        config.DATASET_ARCHIVE_ROOT_DIRECTORY_NAME,
    )
    os.remove(archive_path)
    shutil.rmtree(archive_root_directory)


def main() -> None:
    os.makedirs(config.RAW_DIRECTORY, exist_ok=True)
    os.makedirs(config.RAW_IMAGES_DIRECTORY, exist_ok=True)
    print("Downloading dataset archive...")
    download_dataset_archive(config.DATASET_ARCHIVE_PATH)
    print("Extracting archive...")
    extract_dataset_archive(
        config.DATASET_ARCHIVE_PATH,
        config.RAW_DIRECTORY,
    )
    move_extracted_images_to_raw_directory(
        config.RAW_DIRECTORY,
        config.RAW_IMAGES_DIRECTORY,
    )
    clean_up_intermediate_files(
        config.DATASET_ARCHIVE_PATH,
        config.RAW_DIRECTORY,
    )
    print(f"Saved images to {config.RAW_IMAGES_DIRECTORY}")


if __name__ == "__main__":
    main()
