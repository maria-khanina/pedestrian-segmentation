import logging
import re
import sys

import matplotlib.figure
import matplotlib.pyplot

from pipeline import segmentation_settings as settings
from pipeline.logging_setup import configure_logging

_logger = logging.getLogger(__name__)

_USAGE_MESSAGE = (
    "Usage: python -m pipeline.plot_training_curves "
    "<input_log_path> <output_image_path>"
)

_RUN_START_MARKER = "Epoch 1:"

_EPOCH_LINE_PATTERN = re.compile(
    r"Epoch (\d+): training loss=([\d.]+), validation loss=([\d.]+)"
)

_EpochLosses = tuple[list[int], list[float], list[float]]


def _read_log_lines(log_path: str) -> list[str]:
    with open(log_path, "r") as log_file:
        return log_file.readlines()


def _extract_latest_run_lines(log_lines: list[str]) -> list[str]:
    run_start_indices: list[int] = [
        line_index
        for line_index, line in enumerate(log_lines)
        if _RUN_START_MARKER in line
    ]
    return log_lines[run_start_indices[-1]:] if run_start_indices else log_lines


def _parse_epoch_losses(log_lines: list[str]) -> _EpochLosses:
    matches = (_EPOCH_LINE_PATTERN.search(line) for line in log_lines)
    rows: list[tuple[str, ...]] = [
        match.groups() for match in matches if match is not None
    ]
    epoch_numbers: list[int] = [int(row[0]) for row in rows]
    training_losses: list[float] = [float(row[1]) for row in rows]
    validation_losses: list[float] = [float(row[2]) for row in rows]
    return epoch_numbers, training_losses, validation_losses


def _build_loss_curve_figure(
    epoch_numbers: list[int],
    training_losses: list[float],
    validation_losses: list[float],
) -> matplotlib.figure.Figure:
    figure, axes = matplotlib.pyplot.subplots(
        figsize=settings.LOSS_PLOT_SIZE_IN_INCHES,
    )
    axes.plot(
        epoch_numbers,
        training_losses,
        label="training loss",
        marker="o",
    )
    axes.plot(
        epoch_numbers,
        validation_losses,
        label="validation loss",
        marker="o",
    )
    axes.set_xlabel("epoch")
    axes.set_ylabel("loss (BCE + soft Dice)")
    axes.set_title("Training vs. validation loss")
    axes.legend()
    axes.grid(True)
    return figure


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    if len(sys.argv) != settings.CLI_SCRIPT_ARGUMENT_COUNT:
        _logger.error(_USAGE_MESSAGE)
        sys.exit(1)
    input_log_path, output_image_path = sys.argv[1:]
    latest_run_lines = _extract_latest_run_lines(_read_log_lines(input_log_path))
    epoch_numbers, training_losses, validation_losses = (
        _parse_epoch_losses(latest_run_lines)
    )
    figure = _build_loss_curve_figure(
        epoch_numbers,
        training_losses,
        validation_losses,
    )
    figure.savefig(output_image_path)
    _logger.info(f"Saved loss curve plot to {output_image_path}")


if __name__ == "__main__":
    main()
