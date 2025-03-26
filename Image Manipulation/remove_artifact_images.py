#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_artifact_images.py
#
# Description:
# This script scans through all images in a specified directory
# (optionally recursively) and detects if they have high compression artifacts.
# Images can be removed based on:
#   - A specified threshold (scores above threshold are removed), OR
#   - A specified percentage (remove top X% worst images by artifact score).
#
# Usage:
#   ./remove_artifact_images.py [directory] [options]
#
# Arguments:
#   - [directory]: The path to the directory containing images.
#
# Options:
#   -t THRESHOLD, --threshold THRESHOLD
#                             Threshold for compression artifact detection.
#                             Images with scores above this threshold are removed.
#   -p PERCENTAGE, --percentage PERCENTAGE
#                             Remove this percentage of the highest scoring (worst) images.
#   -D, --per-directory       If using percentage mode, remove worst images per directory (default is global).
#   -n, --dry-run             Simulate the removal of images with high artifacts without deleting them.
#   -r, --recursive           Recursively traverse subdirectories.
#   -v, --verbose             Enable verbose logging (INFO level).
#   -vv, --debug              Enable debug logging (DEBUG level).
#
# Template: ubuntu22.04
#
# Requirements:
#   - OpenCV (install via: apt-get install -y python3-opencv)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Remove images with high compression artifacts based on threshold or percentage."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory containing images.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Threshold for compression artifact detection. Images above this score are removed.",
    )
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        help="Remove this percentage of the highest scoring (worst) images.",
    )
    parser.add_argument(
        "-D",
        "--per-directory",
        action="store_true",
        help="If using percentage mode, remove worst images per directory (default is global).",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Simulate the removal of images with high artifacts without deleting them.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively traverse through subdirectories.",
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

    args = parser.parse_args()

    # Validate arguments
    if args.percentage is not None and args.threshold is not None:
        parser.error(
            "You cannot specify both --threshold and --percentage at the same time."
        )

    # Set default threshold if neither threshold nor percentage is provided
    if args.percentage is None and args.threshold is None:
        args.threshold = 1000.0

    return args


def setup_logging(verbose: bool, debug: bool) -> None:
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


def get_image_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Retrieves a list of image file paths from the specified directory.
    """
    supported_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    image_files: List[str] = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        try:
            for file in os.listdir(directory):
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(directory, file))
        except FileNotFoundError:
            logging.error(f"Directory '{directory}' not found.")
        except PermissionError:
            logging.error(f"Permission denied when accessing '{directory}'.")

    return image_files


def calculate_artifact_score(image: np.ndarray) -> float:
    """
    Calculates a compression artifact score based on blockiness detection.
    A common compression artifact in JPEG images is blockiness caused by the 8x8 DCT blocks.

    Approach:
    - Convert the image to grayscale.
    - JPEG compression typically operates on 8x8 pixel blocks. Compression artifacts often manifest
      as discontinuities along block boundaries.
    - We measure blockiness by summing the absolute differences in intensity across the vertical and horizontal
      boundaries that align with 8-pixel multiples.

    Steps:
    1. Convert the image to grayscale.
    2. For every vertical boundary at columns x = 8, 16, 24, ...:
       - Compute the absolute difference between pixels at column x-1 and x for all rows.
       - Sum these differences to get the vertical blockiness contribution.
    3. For every horizontal boundary at rows y = 8, 16, 24, ...:
       - Compute the absolute difference between pixels at row y-1 and y for all columns.
       - Sum these differences to get the horizontal blockiness contribution.
    4. The final artifact score is the sum of vertical and horizontal blockiness values.

    This metric will be higher for images with more pronounced block boundaries, which often correlate
    with high compression artifacts.

    Note:
    - This is a heuristic that focuses primarily on blockiness, a common JPEG artifact.
    - In practice, other factors may influence perceived compression quality, but this provides a
      logical, block-based approach for measuring common JPEG-like artifacts.
    """
    # Convert to grayscale
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    block_size: int = 8

    vertical_blockiness: float = 0.0
    # Check vertical boundaries
    for x in range(block_size, w, block_size):
        if x >= w:
            break
        col_diff: np.ndarray = np.abs(gray[:, x] - gray[:, x - 1])
        vertical_blockiness += float(np.sum(col_diff))

    horizontal_blockiness: float = 0.0
    # Check horizontal boundaries
    for y in range(block_size, h, block_size):
        if y >= h:
            break
        row_diff: np.ndarray = np.abs(gray[y, :] - gray[y - 1, :])
        horizontal_blockiness += float(np.sum(row_diff))

    artifact_score: float = vertical_blockiness + horizontal_blockiness
    return artifact_score


def get_artifact_score(image_path: str) -> Optional[float]:
    """
    Returns the artifact score for the given image.
    """
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
        return None

    score: float = calculate_artifact_score(image)
    return score


def remove_image(image_path: str, dry_run: bool = False) -> None:
    """
    Removes the specified image file.
    """
    if dry_run:
        logging.info(f"[Dry Run] Would remove: {image_path}")
    else:
        try:
            os.remove(image_path)
            logging.info(f"Removed: {image_path}")
        except FileNotFoundError:
            logging.error(f"File '{image_path}' not found.")
        except PermissionError:
            logging.error(f"Permission denied when removing '{image_path}'.")
        except Exception as e:
            logging.error(f"Failed to remove '{image_path}': {e}")


def remove_by_threshold(
    image_files: List[str], threshold: float, dry_run: bool
) -> None:
    """
    Removes images that have artifact scores above the given threshold.
    """
    any_removed = False
    for img_path in image_files:
        score = get_artifact_score(img_path)
        if score is None:
            continue
        has_artifacts = score > threshold
        status: str = "high_artifacts" if has_artifacts else "low_artifacts"
        logging.debug(
            f"Image '{img_path}' -> Score: {score:.2f}, Threshold: {threshold}, Status: {status}"
        )
        if has_artifacts:
            any_removed = True
            remove_image(img_path, dry_run)

    if any_removed:
        if dry_run:
            logging.info(
                "Some images would be removed due to high compression artifacts (dry run)."
            )
        else:
            logging.info("Some images were removed due to high compression artifacts.")
    else:
        logging.info("No images with high compression artifacts found.")


def remove_by_percentage(
    image_files: List[str],
    percentage: float,
    dry_run: bool,
    per_directory: bool,
    base_directory: str,
) -> None:
    """
    Removes a certain percentage of the highest scoring (worst) images.
    If per_directory is True, this is done for each directory separately.
    If per_directory is False, this is done globally.
    """
    # Compute scores for all images
    scored_images = []
    for img_path in image_files:
        score = get_artifact_score(img_path)
        if score is not None:
            scored_images.append((img_path, score))

    if not scored_images:
        logging.warning("No images could be scored.")
        return

    # For artifacts, a higher score means worse (more artifacts).
    # We want to remove the top X% by score.
    if not per_directory:
        # Global mode: sort all images by score descending
        scored_images.sort(key=lambda x: x[1], reverse=True)
        count_to_remove = int(len(scored_images) * (percentage / 100.0))
        images_to_remove = scored_images[:count_to_remove]
        for img_path, score in images_to_remove:
            logging.debug(f"Removing (global) '{img_path}' -> Score: {score:.2f}")
            remove_image(img_path, dry_run)
        if images_to_remove:
            logging.info(f"Removed {count_to_remove} images ({percentage}% globally).")
        else:
            logging.info("No images removed.")
    else:
        # Per-directory mode
        dir_groups: Dict[str, List[Tuple[str, float]]] = {}
        for img_path, score in scored_images:
            rel_path = os.path.relpath(img_path, base_directory)
            parent_dir = os.path.dirname(rel_path)
            if parent_dir not in dir_groups:
                dir_groups[parent_dir] = []
            dir_groups[parent_dir].append((img_path, score))

        total_removed = 0
        for d, imgs in dir_groups.items():
            # Sort descending by score
            imgs.sort(key=lambda x: x[1], reverse=True)
            count_to_remove = int(len(imgs) * (percentage / 100.0))
            images_to_remove = imgs[:count_to_remove]
            for img_path, score in images_to_remove:
                logging.debug(
                    f"Removing (per-directory) '{img_path}' in '{d}' -> Score: {score:.2f}"
                )
                remove_image(img_path, dry_run)
            total_removed += count_to_remove

        if total_removed > 0:
            logging.info(
                f"Removed {total_removed} images ({percentage}% per directory)."
            )
        else:
            logging.info("No images removed.")


def main() -> None:
    """
    Main function to orchestrate the removal of images with high compression artifacts.
    """
    args: argparse.Namespace = parse_arguments()
    setup_logging(args.verbose, args.debug)

    directory: str = args.directory
    threshold: Optional[float] = args.threshold
    percentage: Optional[float] = args.percentage
    dry_run: bool = args.dry_run
    recursive: bool = args.recursive
    per_directory: bool = args.per_directory

    if not os.path.isdir(directory):
        logging.error(f"Input path '{directory}' is not a directory.")
        sys.exit(1)

    image_files: List[str] = get_image_files(directory, recursive)
    if not image_files:
        logging.warning(f"No image files found in directory '{directory}'.")
        sys.exit(0)

    if percentage is not None:
        mode_description = f"percentage mode ({percentage}%)"
        mode_details = "per directory" if per_directory else "global"
    else:
        mode_description = "threshold mode"
        mode_details = f"threshold: {threshold}"

    logging.info(
        f"Processing {len(image_files)} image(s) in '{directory}' in {mode_description} ({mode_details})."
    )

    if percentage is not None:
        # Percentage-based removal (highest artifact scores)
        remove_by_percentage(image_files, percentage, dry_run, per_directory, directory)
    else:
        # Threshold-based removal
        remove_by_threshold(image_files, threshold, dry_run)


if __name__ == "__main__":
    main()
