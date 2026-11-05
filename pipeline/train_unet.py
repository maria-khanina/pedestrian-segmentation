import copy
import logging
import os
from typing import NamedTuple, Protocol, cast

import torch
import torch.utils.data

from pipeline import segmentation_settings as settings
from pipeline.device import select_device
from pipeline.logging_setup import configure_logging
from pipeline.unet_dataset import (
    ImageMaskPair,
    PedestrianSegmentationDataset,
    list_file_names,
)
from pipeline.unet_model import ModelState, build_unet_model, save_model_state

_DICE_SCORE_SMOOTHING_TERM = 1.0
_IOU_SCORE_SMOOTHING_TERM = 1.0

_logger = logging.getLogger(__name__)

_DataLoader = torch.utils.data.DataLoader[ImageMaskPair]
_EvaluationMetrics = tuple[float, float, float]


class _DifferentiableLoss(Protocol):
    def backward(self) -> None: ...


class _TrainingContext(NamedTuple):
    model: torch.nn.Module
    bce_loss_function: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: str


class _TrainingProgress:
    def __init__(self, model: torch.nn.Module) -> None:
        self.best_validation_loss = float("inf")
        self.best_validation_dice_score = -1.0
        self.best_model_state: ModelState = copy.deepcopy(model.state_dict())
        self.best_epoch_index = -1
        self.epochs_without_improvement = 0

    def remember_if_best_dice(
        self,
        model: torch.nn.Module,
        epoch_index: int,
        validation_dice_score: float,
    ) -> None:
        if validation_dice_score > self.best_validation_dice_score:
            self.best_validation_dice_score = validation_dice_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch_index = epoch_index

    def record_epoch(
        self,
        model: torch.nn.Module,
        epoch_index: int,
        validation_metrics: _EvaluationMetrics,
    ) -> bool:
        validation_loss, validation_dice_score, _ = validation_metrics
        self.remember_if_best_dice(model, epoch_index, validation_dice_score)
        return self.should_stop_after(validation_loss)

    def should_stop_after(self, validation_loss: float) -> bool:
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        patience = settings.EARLY_STOPPING_PATIENCE_IN_EPOCHS
        return self.epochs_without_improvement >= patience


def _build_data_loader(
    images_directory: str,
    masks_directory: str,
    shuffle: bool,
) -> _DataLoader:
    dataset = PedestrianSegmentationDataset(
        images_directory,
        masks_directory,
        list_file_names(images_directory),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.UNET_BATCH_SIZE,
        shuffle=shuffle,
    )


def _build_all_data_loaders() -> tuple[_DataLoader, _DataLoader, _DataLoader]:
    training_loader = _build_data_loader(
        settings.AUGMENTED_TRAIN_IMAGES_DIRECTORY,
        settings.AUGMENTED_TRAIN_MASKS_DIRECTORY,
        True,
    )
    validation_loader = _build_data_loader(
        settings.AUGMENTED_VALIDATION_IMAGES_DIRECTORY,
        settings.AUGMENTED_VALIDATION_MASKS_DIRECTORY,
        False,
    )
    test_loader = _build_data_loader(
        settings.AUGMENTED_TEST_IMAGES_DIRECTORY,
        settings.AUGMENTED_TEST_MASKS_DIRECTORY,
        False,
    )
    return training_loader, validation_loader, test_loader


def _compute_dice_score(
    predicted_mask: torch.Tensor,
    ground_truth_mask: torch.Tensor,
) -> torch.Tensor:
    intersection = (predicted_mask * ground_truth_mask).sum()
    union = predicted_mask.sum() + ground_truth_mask.sum()
    numerator = 2.0 * intersection + _DICE_SCORE_SMOOTHING_TERM
    return numerator / (union + _DICE_SCORE_SMOOTHING_TERM)


def _compute_iou_score(
    predicted_mask: torch.Tensor,
    ground_truth_mask: torch.Tensor,
) -> torch.Tensor:
    intersection = (predicted_mask * ground_truth_mask).sum()
    union = predicted_mask.sum() + ground_truth_mask.sum() - intersection
    numerator = intersection + _IOU_SCORE_SMOOTHING_TERM
    return numerator / (union + _IOU_SCORE_SMOOTHING_TERM)


def _compute_combined_loss(
    predicted_logits: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    bce_loss_function: torch.nn.Module,
) -> torch.Tensor:
    bce_loss: torch.Tensor = bce_loss_function(predicted_logits, ground_truth_mask)
    predicted_probabilities = torch.sigmoid(predicted_logits)
    dice_score = _compute_dice_score(predicted_probabilities, ground_truth_mask)
    return bce_loss + (1.0 - dice_score)


def _run_training_step(
    context: _TrainingContext,
    images: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    masks = masks.to(context.device)
    context.optimizer.zero_grad()
    predictions: torch.Tensor = context.model(images.to(context.device))
    loss = _compute_combined_loss(predictions, masks, context.bce_loss_function)
    cast(_DifferentiableLoss, loss).backward()
    context.optimizer.step()
    return loss.item()


def _train_one_epoch(context: _TrainingContext, data_loader: _DataLoader) -> float:
    context.model.train()
    total_loss = 0.0
    for images, masks in data_loader:
        total_loss += _run_training_step(context, images, masks)
    return total_loss / len(data_loader)


def _evaluate_batch(
    context: _TrainingContext,
    images: torch.Tensor,
    masks: torch.Tensor,
) -> _EvaluationMetrics:
    masks = masks.to(context.device)
    logits: torch.Tensor = context.model(images.to(context.device))
    loss = _compute_combined_loss(logits, masks, context.bce_loss_function)
    probabilities = torch.sigmoid(logits)
    predicted_mask = (probabilities > settings.UNET_PREDICTION_THRESHOLD).float()
    return (
        loss.item(),
        _compute_dice_score(predicted_mask, masks).item(),
        _compute_iou_score(predicted_mask, masks).item(),
    )


def _evaluate(
    context: _TrainingContext,
    data_loader: _DataLoader,
) -> _EvaluationMetrics:
    context.model.eval()
    batch_metrics: list[_EvaluationMetrics] = []
    with torch.no_grad():
        for images, masks in data_loader:
            batch_metrics.append(_evaluate_batch(context, images, masks))
    losses, dice_scores, iou_scores = zip(*batch_metrics)
    number_of_batches = len(data_loader)
    return (
        sum(losses) / number_of_batches,
        sum(dice_scores) / number_of_batches,
        sum(iou_scores) / number_of_batches,
    )


def _log_epoch(
    epoch_index: int,
    training_loss: float,
    validation_metrics: _EvaluationMetrics,
) -> None:
    validation_loss, validation_dice_score, validation_iou_score = validation_metrics
    _logger.info(
        f"Epoch {epoch_index + 1}: "
        f"training loss={training_loss:.4f}, "
        f"validation loss={validation_loss:.4f}, "
        f"validation dice={validation_dice_score:.4f}, "
        f"validation iou={validation_iou_score:.4f}"
    )


def _run_training_loop(
    context: _TrainingContext,
    training_loader: _DataLoader,
    validation_loader: _DataLoader,
) -> _TrainingProgress:
    progress = _TrainingProgress(context.model)
    for epoch_index in range(settings.UNET_NUMBER_OF_EPOCHS):
        training_loss = _train_one_epoch(context, training_loader)
        validation_metrics = _evaluate(context, validation_loader)
        _log_epoch(epoch_index, training_loss, validation_metrics)
        if progress.record_epoch(context.model, epoch_index, validation_metrics):
            _logger.info(
                "Stopping early: validation loss has not improved for "
                f"{progress.epochs_without_improvement} epochs"
            )
            break
    return progress


def _save_best_checkpoint(best_model_state: ModelState) -> None:
    os.makedirs(settings.CHECKPOINTS_DIRECTORY, exist_ok=True)
    save_model_state(best_model_state, settings.UNET_CHECKPOINT_PATH)
    _logger.info(f"Saved best checkpoint to {settings.UNET_CHECKPOINT_PATH}")


def _build_training_context() -> _TrainingContext:
    device = select_device()
    _logger.info(f"Using device: {device}")
    model = build_unet_model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.UNET_LEARNING_RATE,
    )
    return _TrainingContext(model, torch.nn.BCEWithLogitsLoss(), optimizer, device)


def main() -> None:
    configure_logging(logging.INFO, settings.PIPELINE_LOG_PATH)
    context = _build_training_context()
    training_loader, validation_loader, test_loader = _build_all_data_loaders()
    progress = _run_training_loop(context, training_loader, validation_loader)
    _logger.info(
        f"Best validation dice={progress.best_validation_dice_score:.4f} "
        f"at epoch {progress.best_epoch_index + 1}"
    )
    context.model.load_state_dict(progress.best_model_state)
    _save_best_checkpoint(progress.best_model_state)
    test_loss, test_dice_score, test_iou_score = _evaluate(context, test_loader)
    _logger.info(
        f"Held-out test set: loss={test_loss:.4f}, "
        f"dice={test_dice_score:.4f}, iou={test_iou_score:.4f}"
    )


if __name__ == "__main__":
    main()
