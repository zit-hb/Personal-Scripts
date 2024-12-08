#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_blurry_images.py
#
# Description:
# This script scans through all images in a specified directory
# (optionally recursively) and either:
#   - Removes images below a given blur score threshold, OR
#   - Removes a certain percentage of the lowest scoring images globally (default),
#     or per directory if --per-directory is specified.
#
# Usage:
# ./remove_blurry_images.py [directory] [options]
#
# Arguments:
#   - [directory]: The path to the directory containing images.
#
# Options:
#   -t THRESHOLD, --threshold THRESHOLD
#                               Threshold for blurriness detection.
#                               Images below this score are removed.
#   -p PERCENTAGE, --percentage PERCENTAGE
#                               Remove this percentage of the lowest scoring images.
#                               By default, this is done globally.
#   -D, --per-directory         If using percentage mode, remove the lowest scoring images per directory.
#   -m METHOD, --method METHOD  Method to use for blurriness detection (default: laplacian).
#                               Choices: "laplacian", "sobel", "tenengrad".
#   -n, --dry-run               Simulate removal of blurry images without deleting them.
#   -r, --recursive             Recursively traverse subdirectories.
#   -v, --verbose               Enable verbose logging (INFO level).
#   -d, --debug                 Enable debug logging (DEBUG level).
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
        description='Remove blurry images from a directory based on threshold or percentage.'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='The path to the directory containing images.'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        help='Threshold for blurriness detection. Images below this score will be removed.'
    )
    parser.add_argument(
        '-p', '--percentage',
        type=float,
        help='Remove this percentage of the lowest scoring images.'
    )
    parser.add_argument(
        '-D', '--per-directory',
        action='store_true',
        help='If using percentage mode, remove the lowest scoring images per directory (default is global).'
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        default='laplacian',
        choices=['laplacian', 'sobel', 'tenengrad'],
        help='Method to use for blurriness detection.'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Simulate removal of blurry images without deleting them.'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively traverse through subdirectories.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.percentage is not None and args.threshold is not None:
        parser.error("You cannot specify both --threshold and --percentage at the same time.")

    # Set default thresholds if threshold is used and not provided
    if args.percentage is None and args.threshold is None:
        # Default thresholds by method
        if args.method == 'laplacian':
            args.threshold = 100.0
        elif args.method == 'sobel':
            args.threshold = 100.0
        elif args.method == 'tenengrad':
            args.threshold = 300.0

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

    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def get_image_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Retrieves a list of image file paths from the specified directory.
    """
    supported_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
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


def calculate_laplacian_variance(gray_image: np.ndarray) -> float:
    """
    Calculates the Laplacian variance of a grayscale image.
    """
    laplacian: np.ndarray = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance: float = float(laplacian.var())
    return variance


def calculate_sobel_variance(gray_image: np.ndarray) -> float:
    """
    Calculates the Sobel variance of a grayscale image.
    """
    sobelx: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel: np.ndarray = np.hypot(sobelx, sobely)
    variance: float = float(sobel.var())
    return variance


def calculate_tenengrad_variance(gray_image: np.ndarray) -> float:
    """
    Calculates the Tenengrad variance of a grayscale image.
    """
    gx: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy: np.ndarray = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude: np.ndarray = np.sqrt(gx ** 2 + gy ** 2)
    variance: float = float(np.mean(gradient_magnitude ** 2))
    return variance


def get_image_score(image_path: str, method: str) -> Optional[float]:
    """
    Calculates the blurriness score for the image using the specified method.
    Returns None if image cannot be loaded.
    """
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
        return None

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'laplacian':
        score: float = calculate_laplacian_variance(gray)
    elif method == 'sobel':
        score = calculate_sobel_variance(gray)
    elif method == 'tenengrad':
        score = calculate_tenengrad_variance(gray)
    else:
        logging.error(f"Unknown method '{method}'.")
        return None

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


def remove_by_threshold(image_files: List[str], threshold: float, method: str, dry_run: bool) -> None:
    """
    Removes images that are below the given threshold.
    """
    any_removed = False
    for img_path in image_files:
        score = get_image_score(img_path, method)
        if score is None:
            continue
        is_blurry = score < threshold
        status: str = 'blurry' if is_blurry else 'sharp'
        logging.debug(f"Image '{img_path}' -> Score: {score:.2f}, Threshold: {threshold}, Status: {status}")
        if is_blurry:
            any_removed = True
            remove_image(img_path, dry_run)

    if any_removed:
        if dry_run:
            logging.info("Some blurry images would be removed (dry run).")
        else:
            logging.info("Some blurry images were removed.")
    else:
        logging.info("No blurry images found.")


def remove_by_percentage(image_files: List[str], percentage: float, method: str, dry_run: bool,
                         per_directory: bool, base_directory: str) -> None:
    """
    Removes a certain percentage of the lowest scoring images.
    If per_directory is True, this is done for each directory separately.
    If per_directory is False, this is done across all images at once (global).
    """
    # Compute scores for all images
    scored_images = []
    for img_path in image_files:
        score = get_image_score(img_path, method)
        if score is not None:
            scored_images.append((img_path, score))

    if not scored_images:
        logging.warning("No images could be scored.")
        return

    if not per_directory:
        # Global mode: sort all images together
        scored_images.sort(key=lambda x: x[1])  # sort by score ascending
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
        # Group images by their parent directory relative to base_directory
        dir_groups: Dict[str, List[Tuple[str, float]]] = {}
        for img_path, score in scored_images:
            rel_path = os.path.relpath(img_path, base_directory)
            parent_dir = os.path.dirname(rel_path)
            if parent_dir not in dir_groups:
                dir_groups[parent_dir] = []
            dir_groups[parent_dir].append((img_path, score))

        total_removed = 0
        for d, imgs in dir_groups.items():
            imgs.sort(key=lambda x: x[1])  # sort by score ascending
            count_to_remove = int(len(imgs) * (percentage / 100.0))
            images_to_remove = imgs[:count_to_remove]
            for img_path, score in images_to_remove:
                logging.debug(f"Removing (per-directory) '{img_path}' in '{d}' -> Score: {score:.2f}")
                remove_image(img_path, dry_run)
            total_removed += count_to_remove

        if total_removed > 0:
            logging.info(f"Removed {total_removed} images ({percentage}% per directory).")
        else:
            logging.info("No images removed.")


def main() -> None:
    """
    Main function to orchestrate the removal of blurry images.
    """
    args: argparse.Namespace = parse_arguments()
    setup_logging(args.verbose, args.debug)

    directory: str = args.directory
    threshold: Optional[float] = args.threshold
    percentage: Optional[float] = args.percentage
    method: str = args.method
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

    logging.info(f"Processing {len(image_files)} image(s) in '{directory}' with method '{method}' in {mode_description} ({mode_details}).")

    if percentage is not None:
        # Percentage-based removal
        remove_by_percentage(image_files, percentage, method, dry_run, per_directory, directory)
    else:
        # Threshold-based removal
        remove_by_threshold(image_files, threshold, method, dry_run)


if __name__ == '__main__':
    main()
