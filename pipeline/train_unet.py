import copy
import logging
import os

import torch
import torch.utils.data

from pipeline import config
from pipeline.device import select_device
from pipeline.logging_setup import configure_logging
from pipeline.unet_dataset import (
    PedestrianSegmentationDataset,
    list_file_names,
)
from pipeline.unet_model import build_unet_model

_DICE_SCORE_SMOOTHING_TERM = 1.0
_IOU_SCORE_SMOOTHING_TERM = 1.0

_logger = logging.getLogger(__name__)

_ModelState = dict[str, torch.Tensor]

_DataLoaderTriple = tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]

_EvaluationMetrics = tuple[float, float, float]

_BestTrainingResult = tuple[_ModelState, float, int]


def _build_data_loader(
    images_directory: str,
    masks_directory: str,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    file_names = list_file_names(images_directory)
    dataset = PedestrianSegmentationDataset(
        images_directory,
        masks_directory,
        file_names,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.UNET_BATCH_SIZE,
        shuffle=shuffle,
    )


def _build_all_data_loaders() -> _DataLoaderTriple:
    training_loader = _build_data_loader(
        config.AUGMENTED_TRAIN_IMAGES_DIRECTORY,
        config.AUGMENTED_TRAIN_MASKS_DIRECTORY,
        True,
    )
    validation_loader = _build_data_loader(
        config.AUGMENTED_VALIDATION_IMAGES_DIRECTORY,
        config.AUGMENTED_VALIDATION_MASKS_DIRECTORY,
        False,
    )
    test_loader = _build_data_loader(
        config.AUGMENTED_TEST_IMAGES_DIRECTORY,
        config.AUGMENTED_TEST_MASKS_DIRECTORY,
        False,
    )
    return training_loader, validation_loader, test_loader


def _compute_dice_score(
    predicted_binary_mask: torch.Tensor,
    ground_truth_mask: torch.Tensor,
) -> torch.Tensor:
    predicted_flat = predicted_binary_mask.reshape(-1)
    ground_truth_flat = ground_truth_mask.reshape(-1)
    intersection = (predicted_flat * ground_truth_flat).sum()
    union = predicted_flat.sum() + ground_truth_flat.sum()
    numerator = 2.0 * intersection + _DICE_SCORE_SMOOTHING_TERM
    denominator = union + _DICE_SCORE_SMOOTHING_TERM
    return numerator / denominator


def _compute_iou_score(
    predicted_binary_mask: torch.Tensor,
    ground_truth_mask: torch.Tensor,
) -> torch.Tensor:
    predicted_flat = predicted_binary_mask.reshape(-1)
    ground_truth_flat = ground_truth_mask.reshape(-1)
    intersection = (predicted_flat * ground_truth_flat).sum()
    union = (
        predicted_flat.sum()
        + ground_truth_flat.sum()
        - intersection
    )
    numerator = intersection + _IOU_SCORE_SMOOTHING_TERM
    denominator = union + _IOU_SCORE_SMOOTHING_TERM
    return numerator / denominator


def _train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def _evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: str,
) -> _EvaluationMetrics:
    model.eval()
    total_loss = 0.0
    total_dice_score = 0.0
    total_iou_score = 0.0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += loss_function(logits, masks).item()
            probabilities = torch.sigmoid(logits)
            predicted_binary_mask = (
                probabilities > config.UNET_PREDICTION_THRESHOLD
            ).float()
            total_dice_score += _compute_dice_score(
                predicted_binary_mask,
                masks,
            ).item()
            total_iou_score += _compute_iou_score(
                predicted_binary_mask,
                masks,
            ).item()
    number_of_batches = len(data_loader)
    return (
        total_loss / number_of_batches,
        total_dice_score / number_of_batches,
        total_iou_score / number_of_batches,
    )


def _run_training_loop(
    model: torch.nn.Module,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> _BestTrainingResult:
    best_validation_loss = float("inf")
    best_validation_dice_score = -1.0
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch_index = -1
    epochs_without_improvement = 0

    for epoch_index in range(config.UNET_NUMBER_OF_EPOCHS):
        training_loss = _train_one_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
            device,
        )
        validation_loss, validation_dice_score, validation_iou_score = (
            _evaluate(model, validation_loader, loss_function, device)
        )
        _logger.info(
            f"Epoch {epoch_index + 1}: "
            f"training loss={training_loss:.4f}, "
            f"validation loss={validation_loss:.4f}, "
            f"validation dice={validation_dice_score:.4f}, "
            f"validation iou={validation_iou_score:.4f}"
        )

        if validation_dice_score > best_validation_dice_score:
            best_validation_dice_score = validation_dice_score
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_index = epoch_index

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                epochs_without_improvement
                >= config.EARLY_STOPPING_PATIENCE_IN_EPOCHS
            ):
                _logger.info(
                    "Stopping early: validation loss has not "
                    f"improved for {epochs_without_improvement} epochs"
                )
                break

    return best_model_state, best_validation_dice_score, best_epoch_index


def _save_best_checkpoint(best_model_state: _ModelState) -> None:
    os.makedirs(config.CHECKPOINTS_DIRECTORY, exist_ok=True)
    torch.save(best_model_state, config.UNET_CHECKPOINT_PATH)
    _logger.info(
        f"Saved best checkpoint to {config.UNET_CHECKPOINT_PATH}"
    )


def main() -> None:
    configure_logging(logging.INFO, config.PIPELINE_LOG_PATH)
    device = select_device()
    _logger.info(f"Using device: {device}")

    training_loader, validation_loader, test_loader = (
        _build_all_data_loaders()
    )

    model = build_unet_model().to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.UNET_LEARNING_RATE,
    )

    best_model_state, best_validation_dice_score, best_epoch_index = (
        _run_training_loop(
            model,
            training_loader,
            validation_loader,
            loss_function,
            optimizer,
            device,
        )
    )
    _logger.info(
        f"Best validation dice={best_validation_dice_score:.4f} "
        f"at epoch {best_epoch_index + 1}"
    )
    model.load_state_dict(best_model_state)
    _save_best_checkpoint(best_model_state)

    test_loss, test_dice_score, test_iou_score = _evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )
    _logger.info(
        f"Held-out test set: loss={test_loss:.4f}, "
        f"dice={test_dice_score:.4f}, iou={test_iou_score:.4f}"
    )


if __name__ == "__main__":
    main()
