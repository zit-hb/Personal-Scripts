#!/usr/bin/env python3

# -------------------------------------------------------
# Script: train_resnet50.py
#
# Description:
#   Trains a ResNet50-based binary image classifier.
#   Uses ImageFolder datasets, computes metrics, and saves
#   the best model checkpoint based on F1-score.
#
# Usage:
#   ./train_resnet50.py [options] --train_dir TRAIN_DIR --val_dir VAL_DIR
#
# Arguments:
#   -T, --train_dir TRAIN_DIR           Path to training data directory.
#   -V, --val_dir VAL_DIR               Path to validation data directory.
#
# Options:
#   -b, --batch_size BATCH_SIZE         Training batch size. (default: 32)
#   -e, --epochs EPOCHS                 Number of epochs. (default: 20)
#   -l, --learning_rate LR              Base learning rate. (default: 1e-4)
#       --backbone_lr_scale SCALE       LR multiplier for backbone when unfrozen. (default: 0.1)
#       --freeze_backbone_epochs N      Train only the final layer for first N epochs (with pretrained). (default: 3)
#   -o, --output OUTPUT_PATH            Path to save best model. (default: best_model.pth)
#   -w, --weights                       Use pre-trained ImageNet weights.
#   -v, --verbose                       Enable verbose logging (INFO level).
#   -d, --debug                         Enable debug logging (DEBUG level).
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
#   - torch (install via: pip install torch==2.9.0 torchvision==0.24.0)
#   - scikit-learn (install via: pip install scikit-learn==1.7.2)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a ResNet50-based binary image classifier."
    )
    parser.add_argument(
        "-T",
        "--train_dir",
        type=str,
        required=True,
        help="Path to training data directory.",
    )
    parser.add_argument(
        "-V",
        "--val_dir",
        type=str,
        required=True,
        help="Path to validation data directory.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,  # 32 is a standard GPU-friendly batch size; good balance of stability and memory.
        help="Training batch size. (default: 32)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,  # 20 epochs is enough to converge with transfer learning on a small dataset; adjust if needed.
        help="Number of epochs. (default: 20)",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=1e-4,  # 1e-4 is a conservative LR for fine-tuning with Adam; avoids destroying pretrained weights.
        help="Base learning rate. (default: 1e-4)",
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,  # Train backbone 10x slower than head to make fine-tuning gentle and stable.
        help=(
            "Learning rate multiplier for backbone parameters when unfrozen. "
            "(default: 0.1)"
        ),
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=3,  # First 3 epochs only train head so it adapts before touching pretrained features.
        help=(
            "Number of initial epochs to train only the final layer when using "
            "pre-trained weights. (default: 3)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="best_model.pth",
        help="Path to save best model. (default: best_model.pth)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        action="store_true",
        help="Use pre-trained ImageNet weights.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level).",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> torch.device:
    """
    Returns the available device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def validate_data_dirs(train_dir: str, val_dir: str) -> None:
    """
    Validates that training and validation directories exist.
    """
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        logging.error("Train or validation directory does not exist.")
        sys.exit(1)


def get_train_transform() -> transforms.Compose:
    """
    Returns the training data transforms with strong augmentation,
    including occlusion-style noise via RandomErasing.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224
            ),  # 224x224 is ResNet50's native input size.
            transforms.RandomHorizontalFlip(),  # Simulates left/right variation in scenes.
            transforms.ColorJitter(),  # Random brightness/contrast/saturation/hue to handle lighting & color shifts.
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],  # ImageNet mean; matches pretrained weights' expectations.
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],  # ImageNet std; keeps feature scales consistent.
            ),
            transforms.RandomErasing(
                p=0.2,  # Apply occlusion in ~20% of samples to teach robustness to partial noses.
                scale=(
                    0.01,
                    0.1,
                ),  # Erase 1–10% of the image area; enough to cover parts, not entire object.
                ratio=(
                    0.3,
                    3.3,
                ),  # Aspect ratio range from thin to wide blocks; standard recommended range.
                value=0,  # Erase to 0 (black in normalized space); simple, effective occlusion.
                inplace=True,
            ),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """
    Returns the validation data transforms.
    Uses a full-image resize to preserve content.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Deterministic resize; avoids cropping away the tag.
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Same normalization as train.
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_datasets(
    train_dir: str,
    val_dir: str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> Tuple[ImageFolder, ImageFolder]:
    """
    Creates training and validation datasets.
    """
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    logging.info(f"Train classes: {train_dataset.classes}")
    logging.info(f"Val classes:   {val_dataset.classes}")
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: ImageFolder,
    val_dataset: ImageFolder,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for training and validation datasets.
    Adjusts num_workers and pin_memory based on device.
    """
    if device.type == "cuda":
        num_workers = (
            4  # Parallel workers for faster disk loading when GPU is available.
        )
        pin_memory = True  # Pin memory to speed up transfers to GPU.
    else:
        num_workers = 0  # Avoid multiprocessing issues / overhead on simple CPU setups.
        pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffling is critical so batches are well-mixed.
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation to keep it deterministic.
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def compute_class_weights(
    train_dataset: ImageFolder,
    device: torch.device,
) -> torch.FloatTensor:
    """
    Computes class weights for balancing the loss function.
    Inverse-frequency style: rarer class gets higher weight.
    """
    class_counts = Counter(train_dataset.targets)
    total_count = sum(class_counts.values())
    # total_count / count_i: simple heuristic to penalize mistakes on underrepresented classes more.
    class_weights = [total_count / class_counts[i] for i in range(len(class_counts))]
    logging.info(f"Class counts: {class_counts}")
    logging.info(f"Class weights: {class_weights}")
    return torch.FloatTensor(class_weights).to(device)


def create_criterion(
    class_weights: torch.FloatTensor,
) -> nn.Module:
    """
    Creates the loss criterion.
    """
    return nn.CrossEntropyLoss(weight=class_weights)


def create_model(use_pretrained_weights: bool, device: torch.device) -> nn.Module:
    """
    Creates and returns the ResNet50 model for binary classification.
    """
    if use_pretrained_weights:
        weights = (
            ResNet50_Weights.IMAGENET1K_V1
        )  # Standard ImageNet pretrained checkpoint.
    else:
        weights = None

    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features, 2
    )  # Two logits: [no_nose, nose] (or vice versa, via class_to_idx).
    model = model.to(device)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all layers except the final fully-connected layer.
    """
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreezes all layers for fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


def create_optimizer(
    model: nn.Module,
    base_lr: float,
    backbone_lr_scale: float,
) -> optim.Optimizer:
    """
    Creates an Adam optimizer with parameter groups so that
    the backbone can use a scaled learning rate.
    Only parameters with requires_grad=True are included.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": base_lr
                * backbone_lr_scale,  # Backbone learns slower to preserve pretrained features.
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": base_lr,  # Head learns faster to quickly adapt to the new task.
            }
        )

    if not param_groups:
        logging.error("No trainable parameters found for optimizer.")
        sys.exit(1)

    optimizer = optim.Adam(param_groups)
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler.ReduceLROnPlateau:
    """
    Creates a scheduler that reduces LR on validation F1 plateau.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # We want to maximize F1.
        factor=0.5,  # Halve LR when progress stalls; common conservative choice.
        patience=3,  # Wait 3 epochs without F1 improvement before reducing LR.
    )
    return scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch and returns the average training loss.
    """
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluates the model on the validation set and returns metrics.
    """
    model.eval()
    all_preds: Any = []
    all_labels: Any = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return accuracy, precision, recall, f1


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int,
    output_path: str,
    device: torch.device,
    base_lr: float,
    backbone_lr_scale: float,
    use_pretrained: bool,
    freeze_backbone_epochs: int,
) -> None:
    """
    Runs the training and validation loop, with:
    - optional warmup (frozen backbone),
    - backbone unfreezing after warmup,
    - LR scheduling based on F1,
    - best model selection by F1.
    """
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        if (
            use_pretrained
            and freeze_backbone_epochs > 0
            and epoch == freeze_backbone_epochs + 1
        ):
            logging.info("Unfreezing backbone for fine-tuning.")
            unfreeze_backbone(model)
            optimizer = create_optimizer(model, base_lr, backbone_lr_scale)
            scheduler = create_scheduler(optimizer)

        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logging.info(f"Epoch {epoch}/{epochs} - Train loss: {epoch_loss:.4f}")

        accuracy, precision, recall, f1 = evaluate(model, val_loader, device)
        logging.info(
            f"Val Acc: {accuracy:.4f}  Prec: {precision:.4f}  "
            f"Rec: {recall:.4f}  F1: {f1:.4f}"
        )

        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output_path)
            logging.info(f"New best model saved with F1: {best_f1:.4f}")

    logging.info("Training completed.")


def main() -> None:
    """
    Main function to orchestrate the training process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    validate_data_dirs(args.train_dir, args.val_dir)

    device = get_device()
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset, val_dataset = create_datasets(
        args.train_dir,
        args.val_dir,
        train_transform,
        val_transform,
    )

    class_weights = compute_class_weights(train_dataset, device)
    criterion = create_criterion(class_weights)

    model = create_model(args.weights, device)

    if args.weights and args.freeze_backbone_epochs > 0:
        logging.info(
            f"Freezing backbone for first {args.freeze_backbone_epochs} epochs."
        )
        freeze_backbone(model)

    optimizer = create_optimizer(
        model,
        base_lr=args.learning_rate,
        backbone_lr_scale=args.backbone_lr_scale,
    )
    scheduler = create_scheduler(optimizer)

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        device,
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        output_path=args.output,
        device=device,
        base_lr=args.learning_rate,
        backbone_lr_scale=args.backbone_lr_scale,
        use_pretrained=args.weights,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )


if __name__ == "__main__":
    main()
