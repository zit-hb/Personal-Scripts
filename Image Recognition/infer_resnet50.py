#!/usr/bin/env python3

# -------------------------------------------------------
# Script: infer_resnet50.py
#
# Description:
# Loads a trained ResNet50-based image classifier
# and runs inference on one or more input images.
# Supports an arbitrary number of classes as defined
# by the trained model.
#
# Usage:
#   ./infer_resnet50.py [options]
#
# Options:
#   -m, --model MODEL_PATH   Path to trained model file (default: best_model.pth).
#   -i, --input INPUT_PATH   Path to image file or directory with images.
#   -c, --class_names NAMES  Names for classes in order of their indices.
#                            If omitted, auto-generates class_0..class_N-1.
#   -b, --batch_size N       Batch size for batched inference (default: 32).
#   -v, --verbose            Enable verbose logging (INFO level).
#   -vv, --debug             Enable debug logging (DEBUG level).
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
#   - torch (install via: pip install torch==2.9.0 torchvision==0.24.0)
#   - pillow (install via: pip install pillow==12.0.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a ResNet50-based image classifier."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="best_model.pth",
        help="Path to trained model file (default: best_model.pth).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to image file or directory with images.",
    )
    parser.add_argument(
        "-c",
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="Names for classes in order of their indices. "
        "If omitted, auto-generates class_0..class_N-1.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "-vv",
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


def get_transform() -> transforms.Compose:
    """
    Returns the transform for inference (must match validation transform).
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def validate_model_path(model_path: str) -> None:
    """
    Validates that the model file exists.
    """
    if not os.path.isfile(model_path):
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)


def collect_image_paths(input_path: str) -> List[str]:
    """
    Collects image file paths from a single file or directory.
    """
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        image_paths: List[str] = []
        for root, _, files in os.walk(input_path):
            for name in files:
                lower = name.lower()
                if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
                    image_paths.append(os.path.join(root, name))
        if not image_paths:
            logging.error("No image files found in the specified directory.")
            sys.exit(1)
        return sorted(image_paths)

    logging.error(f"Input path does not exist: {input_path}")
    sys.exit(1)


class ImageDataset(Dataset):
    """
    Dataset for loading and transforming images for inference.
    """

    def __init__(self, image_paths: List[str], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logging.error(f"Failed to load image {path}: {exc}")
            raise
        tensor = self.transform(image)
        return tensor, path


def create_model(model_path: str, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Creates the ResNet50 model architecture, infers number of classes
    from the checkpoint, loads weights, and returns the model and class count.
    """
    # Load only tensor weights to avoid executing arbitrary code via pickle.
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    if "fc.weight" not in state_dict:
        logging.error("Checkpoint does not contain fc.weight.")
        sys.exit(1)

    num_classes = state_dict["fc.weight"].shape[0]
    # Use new API: weights=None instead of deprecated pretrained=False
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, num_classes


def create_dataloader(
    image_paths: List[str],
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    """
    Creates a DataLoader for inference.
    Adjusts num_workers and pin_memory based on device to avoid
    shared memory issues on CPU-only environments.
    """
    if device.type == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def resolve_class_names(
    arg_class_names: List[str] | None,
    num_classes: int,
) -> List[str]:
    """
    Resolves class names based on provided arguments and number of classes.
    """
    if arg_class_names is None:
        return [f"class_{i}" for i in range(num_classes)]

    if len(arg_class_names) != num_classes:
        logging.error(
            "Number of provided class names (%d) does not match number of "
            "classes in model (%d).",
            len(arg_class_names),
            num_classes,
        )
        sys.exit(1)

    return arg_class_names


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> None:
    """
    Runs inference and prints predictions for each image.
    """
    num_classes = len(class_names)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            confs, preds = torch.max(probs, dim=1)

            for path, pred, conf, prob_vec in zip(
                paths,
                preds.cpu().tolist(),
                confs.cpu().tolist(),
                probs.cpu().tolist(),
            ):
                class_idx = int(pred)
                if class_idx < 0 or class_idx >= num_classes:
                    logging.error(
                        "Predicted class index %d out of range for %d classes.",
                        class_idx,
                        num_classes,
                    )
                    continue

                class_name = class_names[class_idx]
                confidence = float(conf)

                parts = [
                    f"{path}",
                    f"pred={class_name} (idx={class_idx})",
                ]
                for i, p in enumerate(prob_vec):
                    parts.append(f"p[{class_names[i]}]={float(p):.4f}")
                parts.append(f"conf={confidence:.4f}")

                print("\t".join(parts))


def main() -> None:
    """
    Main function to orchestrate the inference process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    validate_model_path(args.model)

    device = get_device()
    transform = get_transform()

    image_paths = collect_image_paths(args.input)
    dataloader = create_dataloader(image_paths, transform, args.batch_size, device)

    model, num_classes = create_model(args.model, device)
    class_names = resolve_class_names(args.class_names, num_classes)

    predict(model, dataloader, class_names, device)


if __name__ == "__main__":
    main()
