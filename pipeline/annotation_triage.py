import csv
import logging
import sys
from typing import TypedDict

import cv2

from pipeline import segmentation_settings as settings
from pipeline.coco_io import group_annotations_by_image_id, load_coco_dataset
from pipeline.coco_types import CocoAnnotationEntry, CocoDataset
from pipeline.logging_setup import configure_logging
from pipeline.polygon_mask_conversion import convert_polygon_to_point_array

_logger = logging.getLogger(__name__)

_USAGE_MESSAGE = (
    "Usage: python -m pipeline.annotation_triage "
    "<input_coco_json_path> <output_csv_path>"
)

_SIGNAL_NOT_APPLICABLE_FLOAT = -1.0
_SIGNAL_NOT_APPLICABLE_INT = -1

_SIGNAL_NAME_ANNOTATION_COUNT = "annotation_count"
_SIGNAL_NAME_MINIMUM_AREA = "minimum_area"
_SIGNAL_NAME_VERTEX_COUNT = "maximum_vertex_count"
_SIGNAL_NAME_ASPECT_RATIO = "maximum_aspect_ratio"
_SIGNAL_NAME_SOLIDITY = "minimum_solidity"

_CSV_HEADER_ROW = [
    "file_name",
    "cvat_frame_number",
    "priority_rank",
    "worst_signal",
    "annotation_count",
    "minimum_area",
    "maximum_vertex_count",
    "maximum_aspect_ratio",
    "minimum_solidity",
]


class _ImageSignalValues(TypedDict):
    file_name: str
    annotation_count: int
    minimum_area: float
    maximum_vertex_count: int
    maximum_aspect_ratio: float
    minimum_solidity: float


_SuspicionScores = list[tuple[float, str]]
_SignalRanks = list[tuple[str, dict[str, int]]]


def _compute_polygon_vertex_count(polygon: list[float]) -> int:
    return len(polygon) // 2


def _compute_polygon_solidity(polygon: list[float]) -> float:
    point_array = convert_polygon_to_point_array(polygon)
    polygon_area = cv2.contourArea(point_array)
    convex_hull_points = cv2.convexHull(point_array)
    convex_hull_area = cv2.contourArea(convex_hull_points)
    if convex_hull_area == 0.0:
        return 0.0
    return polygon_area / convex_hull_area


def _compute_maximum_vertex_count(
    annotation: CocoAnnotationEntry,
) -> int:
    vertex_counts: list[int] = []
    for polygon in annotation["segmentation"]:
        vertex_counts.append(_compute_polygon_vertex_count(polygon))
    return max(vertex_counts)


def _compute_minimum_solidity(
    annotation: CocoAnnotationEntry,
) -> float:
    solidities: list[float] = []
    for polygon in annotation["segmentation"]:
        solidities.append(_compute_polygon_solidity(polygon))
    return min(solidities)


def _compute_bounding_box_aspect_ratio(
    annotation: CocoAnnotationEntry,
) -> float:
    _, _, bounding_box_width, bounding_box_height = annotation["bbox"]
    if bounding_box_height == 0.0:
        return 0.0
    return bounding_box_width / bounding_box_height


def _compute_image_signal_values(
    file_name: str,
    annotations: list[CocoAnnotationEntry],
) -> _ImageSignalValues:
    if not annotations:
        return _ImageSignalValues(
            file_name=file_name,
            annotation_count=0,
            minimum_area=_SIGNAL_NOT_APPLICABLE_FLOAT,
            maximum_vertex_count=_SIGNAL_NOT_APPLICABLE_INT,
            maximum_aspect_ratio=_SIGNAL_NOT_APPLICABLE_FLOAT,
            minimum_solidity=_SIGNAL_NOT_APPLICABLE_FLOAT,
        )
    return _ImageSignalValues(
        file_name=file_name,
        annotation_count=len(annotations),
        minimum_area=min(annotation["area"] for annotation in annotations),
        maximum_vertex_count=max(
            _compute_maximum_vertex_count(annotation) for annotation in annotations
        ),
        maximum_aspect_ratio=max(
            _compute_bounding_box_aspect_ratio(annotation)
            for annotation in annotations
        ),
        minimum_solidity=min(
            _compute_minimum_solidity(annotation) for annotation in annotations
        ),
    )


def _assign_ranks(
    file_names_sorted_most_suspicious_first: list[str],
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    rank_index = 0
    while rank_index < len(file_names_sorted_most_suspicious_first):
        file_name = file_names_sorted_most_suspicious_first[rank_index]
        ranks[file_name] = rank_index + 1
        rank_index += 1
    return ranks


def _rank_by_suspicion_scores(
    suspicion_scores: _SuspicionScores,
) -> dict[str, int]:
    sorted_scores = sorted(suspicion_scores, reverse=True)
    file_names_sorted: list[str] = []
    for _, file_name in sorted_scores:
        file_names_sorted.append(file_name)
    return _assign_ranks(file_names_sorted)


def _compute_count_deviation_scores(
    image_signal_values: list[_ImageSignalValues],
) -> _SuspicionScores:
    counts: list[int] = []
    for values in image_signal_values:
        counts.append(values["annotation_count"])
    sorted_counts = sorted(counts)
    median_count = sorted_counts[len(sorted_counts) // 2]
    scores: _SuspicionScores = []
    for values in image_signal_values:
        deviation = abs(values["annotation_count"] - median_count)
        scores.append((float(deviation), values["file_name"]))
    return scores


def _compute_minimum_area_scores(
    image_signal_values: list[_ImageSignalValues],
) -> _SuspicionScores:
    scores: _SuspicionScores = []
    for values in image_signal_values:
        if values["minimum_area"] == _SIGNAL_NOT_APPLICABLE_FLOAT:
            continue
        scores.append((-values["minimum_area"], values["file_name"]))
    return scores


def _compute_vertex_count_extremity_scores(
    image_signal_values: list[_ImageSignalValues],
) -> _SuspicionScores:
    applicable_values: list[_ImageSignalValues] = [
        values
        for values in image_signal_values
        if values["maximum_vertex_count"] != _SIGNAL_NOT_APPLICABLE_INT
    ]
    sorted_vertex_counts: list[int] = sorted(
        values["maximum_vertex_count"] for values in applicable_values
    )
    median_vertex_count = sorted_vertex_counts[len(sorted_vertex_counts) // 2]
    return [
        (
            float(abs(values["maximum_vertex_count"] - median_vertex_count)),
            values["file_name"],
        )
        for values in applicable_values
    ]


def _compute_maximum_aspect_ratio_scores(
    image_signal_values: list[_ImageSignalValues],
) -> _SuspicionScores:
    scores: _SuspicionScores = []
    for values in image_signal_values:
        if values["maximum_aspect_ratio"] == _SIGNAL_NOT_APPLICABLE_FLOAT:
            continue
        scores.append(
            (values["maximum_aspect_ratio"], values["file_name"])
        )
    return scores


def _compute_minimum_solidity_scores(
    image_signal_values: list[_ImageSignalValues],
) -> _SuspicionScores:
    scores: _SuspicionScores = []
    for values in image_signal_values:
        if values["minimum_solidity"] == _SIGNAL_NOT_APPLICABLE_FLOAT:
            continue
        scores.append((-values["minimum_solidity"], values["file_name"]))
    return scores


def _compute_all_signal_ranks(
    image_signal_values: list[_ImageSignalValues],
) -> _SignalRanks:
    return [
        (
            _SIGNAL_NAME_ANNOTATION_COUNT,
            _rank_by_suspicion_scores(
                _compute_count_deviation_scores(image_signal_values)
            ),
        ),
        (
            _SIGNAL_NAME_MINIMUM_AREA,
            _rank_by_suspicion_scores(
                _compute_minimum_area_scores(image_signal_values)
            ),
        ),
        (
            _SIGNAL_NAME_VERTEX_COUNT,
            _rank_by_suspicion_scores(
                _compute_vertex_count_extremity_scores(
                    image_signal_values
                )
            ),
        ),
        (
            _SIGNAL_NAME_ASPECT_RATIO,
            _rank_by_suspicion_scores(
                _compute_maximum_aspect_ratio_scores(image_signal_values)
            ),
        ),
        (
            _SIGNAL_NAME_SOLIDITY,
            _rank_by_suspicion_scores(
                _compute_minimum_solidity_scores(image_signal_values)
            ),
        ),
    ]


def _find_worst_rank_and_signal(
    file_name: str,
    signal_ranks: _SignalRanks,
) -> tuple[int, str]:
    candidates: list[tuple[int, str]] = [
        (ranks_by_file_name[file_name], signal_name)
        for signal_name, ranks_by_file_name in signal_ranks
        if file_name in ranks_by_file_name
    ]
    return min(candidates, key=lambda candidate: candidate[0])


def _compute_cvat_frame_numbers(
    file_names: list[str],
) -> dict[str, int]:
    sorted_file_names = sorted(file_names)
    frame_numbers: dict[str, int] = {}
    frame_index = 0
    while frame_index < len(sorted_file_names):
        frame_numbers[sorted_file_names[frame_index]] = frame_index
        frame_index += 1
    return frame_numbers


def _format_optional_float_field(value: float) -> str:
    if value == _SIGNAL_NOT_APPLICABLE_FLOAT:
        return ""
    return str(value)


def _format_optional_int_field(value: int) -> str:
    if value == _SIGNAL_NOT_APPLICABLE_INT:
        return ""
    return str(value)


def _build_csv_row(
    values: _ImageSignalValues,
    frame_number: int,
    priority_rank: int,
    worst_signal_name: str,
) -> list[str]:
    return [
        values["file_name"],
        str(frame_number),
        str(priority_rank),
        worst_signal_name,
        str(values["annotation_count"]),
        _format_optional_float_field(values["minimum_area"]),
        _format_optional_int_field(values["maximum_vertex_count"]),
        _format_optional_float_field(values["maximum_aspect_ratio"]),
        _format_optional_float_field(values["minimum_solidity"]),
    ]


def _extract_file_names(
    image_signal_values: list[_ImageSignalValues],
) -> list[str]:
    file_names: list[str] = []
    for values in image_signal_values:
        file_names.append(values["file_name"])
    return file_names


def _build_ranked_rows(
    image_signal_values: list[_ImageSignalValues],
    signal_ranks: _SignalRanks,
    frame_numbers: dict[str, int],
) -> list[tuple[int, list[str]]]:
    rows_with_rank: list[tuple[int, list[str]]] = []
    for values in image_signal_values:
        file_name = values["file_name"]
        priority_rank, worst_signal_name = _find_worst_rank_and_signal(
            file_name,
            signal_ranks,
        )
        row = _build_csv_row(
            values,
            frame_numbers[file_name],
            priority_rank,
            worst_signal_name,
        )
        rows_with_rank.append((priority_rank, row))
    return rows_with_rank


def _write_csv_rows(
    rows_with_rank_sorted: list[tuple[int, list[str]]],
    output_csv_path: str,
) -> None:
    with open(output_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(_CSV_HEADER_ROW)
        for _, row in rows_with_rank_sorted:
            csv_writer.writerow(row)


def _write_priority_csv(
    image_signal_values: list[_ImageSignalValues],
    signal_ranks: _SignalRanks,
    output_csv_path: str,
) -> None:
    file_names = _extract_file_names(image_signal_values)
    frame_numbers = _compute_cvat_frame_numbers(file_names)
    rows_with_rank = _build_ranked_rows(
        image_signal_values,
        signal_ranks,
        frame_numbers,
    )
    rows_with_rank_sorted = sorted(rows_with_rank)
    _write_csv_rows(rows_with_rank_sorted, output_csv_path)


def _compute_all_image_signal_values(
    coco_dataset: CocoDataset,
) -> list[_ImageSignalValues]:
    annotations_by_image_id = group_annotations_by_image_id(coco_dataset)
    return [
        _compute_image_signal_values(
            image["file_name"],
            annotations_by_image_id.get(image["id"], []),
        )
        for image in coco_dataset["images"]
    ]


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    if len(sys.argv) != settings.CLI_SCRIPT_ARGUMENT_COUNT:
        _logger.error(_USAGE_MESSAGE)
        sys.exit(1)
    input_coco_path, output_csv_path = sys.argv[1:]
    coco_dataset = load_coco_dataset(input_coco_path)
    image_signal_values = _compute_all_image_signal_values(coco_dataset)
    signal_ranks = _compute_all_signal_ranks(image_signal_values)
    _write_priority_csv(image_signal_values, signal_ranks, output_csv_path)
    _logger.info(f"Wrote review-priority ranking to {output_csv_path}")


if __name__ == "__main__":
    main()
