#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_images_by_size.py
#
# Description:
# This script searches for images and removes those that do not meet
# specified dimension requirements. It supports various image formats
# and provides options for dry runs, verbosity, recursive directory
# traversal, and moving files instead of deleting them.
#
# Usage:
# ./remove_images_by_size.py [options] [directory|image]
#
# Arguments:
#   - [directory|image]: The image file or directory to process.
#
# Options:
#   -W WIDTH, --required-width WIDTH     Limit images to certain width (e.g., 1024, >1024, <1024).
#                                        Defaults to '>=600' if not provided, keeping images with 600 or more pixels.
#   -H HEIGHT, --required-height HEIGHT  Limit images to certain height (e.g., 768, >768, <768).
#                                        Defaults to '>=600' if not provided, keeping images with 600 or more pixels.
#   -m DIR, --move-dir DIR               Move files to this directory instead of deleting.
#   -n, --dry-run                        Show what would be done without making any changes.
#   -v, --verbose                        Enable verbose output.
#   -r, --recursive                      Process directories recursively.
#   -h, --help                           Display this help message.
#
# Template: ubuntu24.04
#
# Requirements:
#   - Pillow (install via: pip install Pillow==11.1.0)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import re
import sys
from PIL import Image
from typing import List, Tuple, Optional


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove images that do not meet specified dimension requirements."
    )
    parser.add_argument(
        "-W",
        "--required-width",
        type=str,
        default=">=600",
        help="Limit input images to certain width (e.g., 1024, >1024, <1024). Defaults to '>=600' if not provided.",
    )
    parser.add_argument(
        "-H",
        "--required-height",
        type=str,
        default=">=600",
        help="Limit input images to certain height (e.g., 768, >768, <768). Defaults to '>=600' if not provided.",
    )
    parser.add_argument(
        "-m",
        "--move-dir",
        type=str,
        help="Move files to this directory instead of deleting.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any changes.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively.",
    )
    parser.add_argument(
        "path",
        type=str,
        help="The image file or directory to process.",
    )
    args = parser.parse_args()
    return args


def setup_logging(verbose: bool):
    """Sets up the logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_images(path: str, recursive: bool) -> List[str]:
    """Collects all image files from the provided path."""
    supported_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp")
    image_files = []

    if os.path.isfile(path):
        if path.lower().endswith(supported_extensions):
            image_files.append(path)
        else:
            logging.warning(f"File '{path}' is not a supported image format, skipping.")
    elif os.path.isdir(path):
        if recursive:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(path):
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(path, file))
    else:
        logging.warning(f"Path '{path}' is neither a file nor a directory, skipping.")

    if not image_files:
        logging.error("No image files found in the specified path.")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to process.")
    return image_files


def parse_dimension_requirement(requirement_str: str) -> Tuple[str, int]:
    """Parses a dimension requirement string into an operator and value."""
    match = re.match(r"(<=|>=|<|>|=)?\s*(\d+)", requirement_str.strip())
    if not match:
        logging.error(f"Invalid dimension requirement '{requirement_str}'.")
        sys.exit(1)

    operator = match.group(1) if match.group(1) else "="
    value = int(match.group(2))
    return operator, value


def check_dimension(dimension: int, operator: str, value: int) -> bool:
    """Checks if a dimension satisfies the requirement."""
    if operator == ">":
        return dimension > value
    elif operator == "<":
        return dimension < value
    elif operator == "=":
        return dimension == value
    elif operator == ">=":
        return dimension >= value
    elif operator == "<=":
        return dimension <= value
    else:
        logging.error(f"Unknown operator '{operator}'.")
        sys.exit(1)


def filter_images_by_size(
    img_path: str,
    width_requirement: Optional[Tuple[str, int]],
    height_requirement: Optional[Tuple[str, int]],
) -> bool:
    """Determines whether an image meets the dimension requirements."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width_requirement:
                if not check_dimension(width, *width_requirement):
                    logging.info(
                        f"Selecting '{img_path}' due to width requirement (width: {width})."
                    )
                    return False
            if height_requirement:
                if not check_dimension(height, *height_requirement):
                    logging.info(
                        f"Selecting '{img_path}' due to height requirement (height: {height})."
                    )
                    return False
            return True
    except Exception as e:
        logging.error(f"Failed to open image '{img_path}': {e}")
        return False


def remove_or_move_image(img_path: str, move_dir: Optional[str], dry_run: bool):
    """Removes or moves an image based on the provided options."""
    if dry_run:
        action = "Would remove" if not move_dir else f"Would move to '{move_dir}'"
        logging.info(f"{action} '{img_path}'.")
        return

    if move_dir:
        try:
            os.makedirs(move_dir, exist_ok=True)
            destination = os.path.join(move_dir, os.path.basename(img_path))
            os.rename(img_path, destination)
            logging.info(f"Moved '{img_path}' to '{destination}'.")
        except Exception as e:
            logging.error(f"Failed to move '{img_path}' to '{move_dir}': {e}")
    else:
        try:
            os.remove(img_path)
            logging.info(f"Removed '{img_path}'.")
        except Exception as e:
            logging.error(f"Failed to remove '{img_path}': {e}")


def main():
    """Main function that orchestrates the image removal process."""
    args = parse_arguments()
    setup_logging(args.verbose)

    # Parse dimension requirements
    width_requirement = None
    if args.required_width:
        width_requirement = parse_dimension_requirement(args.required_width)
    height_requirement = None
    if args.required_height:
        height_requirement = parse_dimension_requirement(args.required_height)

    image_files = collect_images(args.path, args.recursive)

    for img_path in image_files:
        if not os.access(img_path, os.R_OK):
            logging.warning(f"Cannot read file '{img_path}', skipping.")
            continue

        # Filter images based on dimension requirements
        if not filter_images_by_size(img_path, width_requirement, height_requirement):
            remove_or_move_image(img_path, args.move_dir, args.dry_run)
        else:
            if args.verbose:
                logging.info(f"Keeping '{img_path}' as it meets all requirements.")


if __name__ == "__main__":
    main()
