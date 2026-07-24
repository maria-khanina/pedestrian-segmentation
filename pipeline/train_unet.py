import os

import torch
import torch.utils.data

from pipeline import config
from pipeline.unet_dataset import (
    PedestrianSegmentationDataset,
    list_augmented_file_names,
)
from pipeline.unet_model import build_unet_model

DICE_SCORE_SMOOTHING_TERM = 1.0

DataLoaderPair = tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_file_names(file_names: list[str]) -> tuple[list[str], list[str]]:
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    permutation = torch.randperm(len(file_names), generator=generator)
    validation_size = int(
        len(file_names) * config.UNET_VALIDATION_FRACTION
    )
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    training_file_names: list[str] = []
    validation_file_names: list[str] = []
    for index in training_indices.tolist():
        training_file_names.append(file_names[index])
    for index in validation_indices.tolist():
        validation_file_names.append(file_names[index])
    return training_file_names, validation_file_names


def build_data_loaders(
    training_file_names: list[str],
    validation_file_names: list[str],
) -> DataLoaderPair:
    training_dataset = PedestrianSegmentationDataset(
        training_file_names
    )
    validation_dataset = PedestrianSegmentationDataset(
        validation_file_names
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=config.UNET_BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.UNET_BATCH_SIZE,
        shuffle=False,
    )
    return training_loader, validation_loader


def compute_dice_score(
    predicted_probabilities: torch.Tensor,
    ground_truth_masks: torch.Tensor,
) -> torch.Tensor:
    predicted_flat = predicted_probabilities.reshape(-1)
    ground_truth_flat = ground_truth_masks.reshape(-1)
    intersection = (predicted_flat * ground_truth_flat).sum()
    union = predicted_flat.sum() + ground_truth_flat.sum()
    numerator = 2.0 * intersection + DICE_SCORE_SMOOTHING_TERM
    denominator = union + DICE_SCORE_SMOOTHING_TERM
    return numerator / denominator


def train_one_epoch(
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


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    model.eval()
    total_dice_score = 0.0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = torch.sigmoid(model(images))
            total_dice_score += compute_dice_score(
                predictions,
                masks,
            ).item()
    return total_dice_score / len(data_loader)


def main() -> None:
    device = select_device()
    print(f"Using device: {device}")
    file_names = list_augmented_file_names()
    training_file_names, validation_file_names = split_file_names(
        file_names
    )
    training_loader, validation_loader = build_data_loaders(
        training_file_names,
        validation_file_names,
    )
    model = build_unet_model().to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.UNET_LEARNING_RATE,
    )
    os.makedirs(config.CHECKPOINTS_DIRECTORY, exist_ok=True)
    for epoch_index in range(config.UNET_NUMBER_OF_EPOCHS):
        training_loss = train_one_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
            device,
        )
        validation_dice_score = evaluate(
            model,
            validation_loader,
            device,
        )
        print(
            f"Epoch {epoch_index + 1}: "
            f"training loss={training_loss:.4f}, "
            f"validation dice={validation_dice_score:.4f}"
        )
    torch.save(model.state_dict(), config.UNET_CHECKPOINT_PATH)
    print(f"Saved checkpoint to {config.UNET_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
