import logging
import re
import sys

import matplotlib.figure
import matplotlib.pyplot

from pipeline import config
from pipeline.logging_setup import configure_logging

_logger = logging.getLogger(__name__)

_USAGE_MESSAGE = (
    "Usage: python -m pipeline.plot_training_curves "
    "<input_log_path> <output_image_path>"
)

_RUN_START_MARKER = "Using device:"

_EPOCH_LINE_PATTERN = re.compile(
    r"Epoch (\d+): training loss=([\d.]+), validation loss=([\d.]+)"
)

_EpochLosses = tuple[list[int], list[float], list[float]]


def _read_log_lines(log_path: str) -> list[str]:
    with open(log_path, "r") as log_file:
        return log_file.readlines()


def _extract_latest_run_lines(log_lines: list[str]) -> list[str]:
    latest_run_start_index = 0
    line_index = 0
    while line_index < len(log_lines):
        if _RUN_START_MARKER in log_lines[line_index]:
            latest_run_start_index = line_index
        line_index += 1
    return log_lines[latest_run_start_index:]


def _parse_epoch_losses(log_lines: list[str]) -> _EpochLosses:
    epoch_numbers: list[int] = []
    training_losses: list[float] = []
    validation_losses: list[float] = []
    for line in log_lines:
        match = _EPOCH_LINE_PATTERN.search(line)
        if match is None:
            continue
        epoch_numbers.append(int(match.group(1)))
        training_losses.append(float(match.group(2)))
        validation_losses.append(float(match.group(3)))
    return epoch_numbers, training_losses, validation_losses


def _build_loss_curve_figure(
    epoch_numbers: list[int],
    training_losses: list[float],
    validation_losses: list[float],
) -> matplotlib.figure.Figure:
    figure, axes = matplotlib.pyplot.subplots(figsize=(8, 5))
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
    configure_logging(logging.INFO, config.PIPELINE_LOG_PATH)
    if len(sys.argv) != config.CLI_SCRIPT_ARGUMENT_COUNT:
        _logger.error(_USAGE_MESSAGE)
        sys.exit(1)
    input_log_path = sys.argv[1]
    output_image_path = sys.argv[2]
    log_lines = _read_log_lines(input_log_path)
    latest_run_lines = _extract_latest_run_lines(log_lines)
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
