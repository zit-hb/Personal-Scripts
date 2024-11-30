#!/usr/bin/env python3

# -------------------------------------------------------
# Script: crop_images.py
#
# Description:
# This script batch-crops images by removing a specified number of pixels or percentages
# from the top, bottom, left, or right sides of each image. It supports various image
# formats and provides options for dry runs, verbosity, output directories, and filtering
# by image dimensions.
#
# Usage:
# ./crop_images.py [options] [directory|image]
#
#  [directory|image]: The image file or directory to process.
#
# Options:
#   -T TOP, --top TOP                    Number of pixels or percentage to crop from the top.
#   -B BOTTOM, --bottom BOTTOM           Number of pixels or percentage to crop from the bottom.
#   -L LEFT, --left LEFT                 Number of pixels or percentage to crop from the left.
#   -R RIGHT, --right RIGHT              Number of pixels or percentage to crop from the right.
#   -p, --percentage                     Interpret crop values as percentages.
#   -W WIDTH, --required-width WIDTH     Limit input images to certain width (e.g., 1024, >1024, <1024).
#   -H HEIGHT, --required-height HEIGHT  Limit input images to certain height (e.g., 768, >768, <768).
#   -o DIR, --output-dir DIR             Output directory for the cropped images.
#   -k, --backup                         Keep a backup of the original images.
#   -n, --dry-run                        Show what would be done without making any changes.
#   -v, --verbose                        Enable verbose output.
#   -r, --recursive                      Process directories recursively.
#   -h, --help                           Display this help message.
#
# Template: ubuntu22.04
#
# Requirements:
# - Pillow (install via: pip install Pillow==11.0.0)
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
        description='Batch-crop images by removing specified pixels or percentages from each side.'
    )
    parser.add_argument(
        '-T', '--top',
        type=str,
        default='0',
        help='Number of pixels or percentage to crop from the top.'
    )
    parser.add_argument(
        '-B', '--bottom',
        type=str,
        default='0',
        help='Number of pixels or percentage to crop from the bottom.'
    )
    parser.add_argument(
        '-L', '--left',
        type=str,
        default='0',
        help='Number of pixels or percentage to crop from the left.'
    )
    parser.add_argument(
        '-R', '--right',
        type=str,
        default='0',
        help='Number of pixels or percentage to crop from the right.'
    )
    parser.add_argument(
        '-p', '--percentage',
        action='store_true',
        help='Interpret crop values as percentages.'
    )
    parser.add_argument(
        '-W', '--required-width',
        type=str,
        help='Limit input images to certain width (e.g., 1024, >1024, <1024).'
    )
    parser.add_argument(
        '-H', '--required-height',
        type=str,
        help='Limit input images to certain height (e.g., 768, >768, <768).'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Output directory for the cropped images.'
    )
    parser.add_argument(
        '-k', '--backup',
        action='store_true',
        help='Keep a backup of the original images.'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively.'
    )
    parser.add_argument(
        'path',
        type=str,
        help='The image file or directory to process.'
    )
    args = parser.parse_args()
    return args


def setup_logging(verbose: bool):
    """Sets up the logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def collect_images(path: str, recursive: bool) -> List[str]:
    """Collects all image files from the provided path."""
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp')
    image_files = []

    if os.path.isfile(path):
        if path.lower().endswith(supported_extensions):
            image_files.append(path)
        else:
            logging.error(f"File '{path}' is not a supported image format.")
            sys.exit(1)
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
        logging.error(f"Path '{path}' is neither a file nor a directory.")
        sys.exit(1)

    if not image_files:
        logging.error("No image files found in the specified path.")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to process.")
    return image_files


def parse_crop_value(value_str: str) -> Tuple[float, bool]:
    """Parses a crop value string to determine if it's a pixel or percentage value."""
    is_percentage = False
    if value_str.endswith('%'):
        is_percentage = True
        value_str = value_str[:-1]

    try:
        value = float(value_str)
    except ValueError:
        logging.error(f"Invalid crop value '{value_str}'. Must be a number or percentage.")
        sys.exit(1)

    return value, is_percentage


def calculate_crop_pixels(img_size: Tuple[int, int], crop_values: dict, percentage_mode: bool) -> dict:
    """Calculates the number of pixels to crop from each side."""
    width, height = img_size
    crop_pixels = {}

    for side in ['top', 'bottom', 'left', 'right']:
        value, is_percentage = parse_crop_value(crop_values[side])
        if percentage_mode or is_percentage:
            pixels = int((value / 100) * (height if side in ['top', 'bottom'] else width))
        else:
            pixels = int(value)
        crop_pixels[side] = pixels

    return crop_pixels


def crop_image(img_path: str, crop_pixels: dict, output_dir: Optional[str], dry_run: bool, backup: bool):
    """Crops a single image based on the specified pixel values."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            new_left = crop_pixels['left']
            new_upper = crop_pixels['top']
            new_right = width - crop_pixels['right']
            new_lower = height - crop_pixels['bottom']

            if new_left >= new_right or new_upper >= new_lower:
                logging.error(f"Crop values too large for image '{img_path}', skipping.")
                return

            if dry_run:
                logging.info(f"Would crop '{img_path}' to box ({new_left}, {new_upper}, {new_right}, {new_lower}).")
                return

            cropped_img = img.crop((new_left, new_upper, new_right, new_lower))

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, os.path.basename(img_path))
            else:
                output_path = img_path

            if backup and output_path == img_path:
                backup_path = img_path + '.bak'
                if not os.path.exists(backup_path):
                    os.rename(img_path, backup_path)
                else:
                    logging.warning(f"Backup file '{backup_path}' already exists, skipping backup.")
                cropped_img.save(img_path)
            else:
                cropped_img.save(output_path)

            logging.info(f"Cropped image saved as '{output_path}'.")

    except Exception as e:
        logging.error(f"Failed to process image '{img_path}': {e}")


def parse_dimension_requirement(requirement_str: str) -> Tuple[str, int]:
    """Parses a dimension requirement string into an operator and value."""
    match = re.match(r'(<=|>=|<|>|=)?\s*(\d+)', requirement_str.strip())
    if not match:
        logging.error(f"Invalid dimension requirement '{requirement_str}'.")
        sys.exit(1)

    operator = match.group(1) if match.group(1) else '='
    value = int(match.group(2))
    return operator, value


def check_dimension(dimension: int, operator: str, value: int) -> bool:
    """Checks if a dimension satisfies the requirement."""
    if operator == '>':
        return dimension > value
    elif operator == '<':
        return dimension < value
    elif operator == '=':
        return dimension == value
    elif operator == '>=':
        return dimension >= value
    elif operator == '<=':
        return dimension <= value
    else:
        logging.error(f"Unknown operator '{operator}'.")
        sys.exit(1)


def filter_images_by_size(img_path: str, width_requirement: Optional[Tuple[str, int]], height_requirement: Optional[Tuple[str, int]]) -> bool:
    """Determines whether an image meets the dimension requirements."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width_requirement:
                if not check_dimension(width, *width_requirement):
                    logging.info(
                        f"Skipping '{img_path}' due to width requirement "
                        f"(width: {width})."
                    )
                    return False
            if height_requirement:
                if not check_dimension(height, *height_requirement):
                    logging.info(
                        f"Skipping '{img_path}' due to height requirement "
                        f"(height: {height})."
                    )
                    return False
            return True
    except Exception as e:
        logging.error(f"Failed to open image '{img_path}': {e}")
        return False


def main():
    """Main function that orchestrates the image cropping process."""
    args = parse_arguments()
    setup_logging(args.verbose)

    crop_values = {
        'top': args.top,
        'bottom': args.bottom,
        'left': args.left,
        'right': args.right
    }

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

        # Filter images based on width and height requirements
        if not filter_images_by_size(img_path, width_requirement, height_requirement):
            continue

        try:
            with Image.open(img_path) as img:
                crop_pixels = calculate_crop_pixels(
                    img.size, crop_values, args.percentage
                )
        except Exception as e:
            logging.error(f"Failed to open image '{img_path}': {e}")
            continue

        crop_image(
            img_path,
            crop_pixels,
            args.output_dir,
            args.dry_run,
            args.backup
        )


if __name__ == '__main__':
    main()
