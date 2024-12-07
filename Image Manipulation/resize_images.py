#!/usr/bin/env python3

# -------------------------------------------------------
# Script: resize_images.py
#
# Description:
# This script resizes image files to ensure they do not exceed specified
# maximum dimensions or fall below specified minimum dimensions. It supports
# various image formats and provides options for dry runs, verbosity,
# recursive directory traversal, and output directories.
#
# Usage:
# ./resize_images.py [options] [directory|image]
#
# Arguments:
#   - [directory|image]: The image file or directory to process.
#
# Options:
#   -W WIDTH, --max-width WIDTH       Maximum width of images (e.g., 2048).
#   -H HEIGHT, --max-height HEIGHT    Maximum height of images (e.g., 2048).
#   -w WIDTH, --min-width WIDTH       Minimum width of images (e.g., 800).
#   -e HEIGHT, --min-height HEIGHT    Minimum height of images (e.g., 600).
#   -o DIR, --output-dir DIR          Output directory for the resized images.
#   -k, --backup                      Keep a backup of the original images.
#   -n, --dry-run                     Show what would be done without making any changes.
#   -v, --verbose                     Enable verbose output.
#   -r, --recursive                   Process directories recursively.
#   --keep-aspect-ratio               Maintain the aspect ratio (default).
#   --ignore-aspect-ratio             Do not maintain the aspect ratio.
#   -h, --help                        Display this help message.
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
import sys
from PIL import Image
from typing import List, Optional


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Resize images to fit within specified dimensions.'
    )
    parser.add_argument(
        '-W', '--max-width',
        type=int,
        help='Maximum width of images (e.g., 2048).'
    )
    parser.add_argument(
        '-H', '--max-height',
        type=int,
        help='Maximum height of images (e.g., 2048).'
    )
    parser.add_argument(
        '-w', '--min-width',
        type=int,
        help='Minimum width of images (e.g., 800).'
    )
    parser.add_argument(
        '-e', '--min-height',
        type=int,
        help='Minimum height of images (e.g., 600).'
    )
    parser.add_argument(
        '--keep-aspect-ratio',
        action='store_true',
        default=True,
        help='Maintain the aspect ratio (default).'
    )
    parser.add_argument(
        '--ignore-aspect-ratio',
        action='store_false',
        dest='keep_aspect_ratio',
        help='Do not maintain the aspect ratio.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Output directory for the resized images.'
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


def resize_image(
    img_path: str,
    max_width: Optional[int],
    max_height: Optional[int],
    min_width: Optional[int],
    min_height: Optional[int],
    keep_aspect_ratio: bool,
    output_dir: Optional[str],
    dry_run: bool,
    backup: bool
):
    """Resizes a single image based on specified dimensions."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            new_width, new_height = width, height

            # Determine if resizing is needed
            resize_needed = False

            # Check if the image exceeds max dimensions
            if max_width and width > max_width:
                resize_needed = True
            if max_height and height > max_height:
                resize_needed = True

            # Check if the image is below min dimensions
            if min_width and width < min_width:
                resize_needed = True
            if min_height and height < min_height:
                resize_needed = True

            if not resize_needed:
                if dry_run or args.verbose:
                    logging.info(f"Skipping '{img_path}' (Size: {width}x{height})")
                return

            # Calculate new dimensions
            if keep_aspect_ratio:
                aspect_ratio = width / height

                # Adjust dimensions based on max constraints
                if max_width and width > max_width:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                if max_height and new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)

                # Adjust dimensions based on min constraints
                if min_width and new_width < min_width:
                    new_width = min_width
                    new_height = int(new_width / aspect_ratio)
                if min_height and new_height < min_height:
                    new_height = min_height
                    new_width = int(new_height * aspect_ratio)
            else:
                new_width = width
                new_height = height

                if max_width and width > max_width:
                    new_width = max_width
                if max_height and height > max_height:
                    new_height = max_height
                if min_width and width < min_width:
                    new_width = min_width
                if min_height and height < min_height:
                    new_height = min_height

            new_size = (new_width, new_height)

            if dry_run:
                logging.info(f"Would resize '{img_path}' from {width}x{height} to {new_width}x{new_height}")
                return

            resized_img = img.resize(new_size, Image.ANTIALIAS)

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
                resized_img.save(img_path)
            else:
                resized_img.save(output_path)

            logging.info(f"Resized image saved as '{output_path}' (New size: {new_width}x{new_height})")

    except Exception as e:
        logging.error(f"Failed to process image '{img_path}': {e}")


def main():
    """Main function that orchestrates the image resizing process."""
    global args
    args = parse_arguments()
    setup_logging(args.verbose)

    image_files = collect_images(args.path, args.recursive)

    for img_path in image_files:
        if not os.access(img_path, os.R_OK):
            logging.warning(f"Cannot read file '{img_path}', skipping.")
            continue

        resize_image(
            img_path,
            args.max_width,
            args.max_height,
            args.min_width,
            args.min_height,
            args.keep_aspect_ratio,
            args.output_dir,
            args.dry_run,
            args.backup
        )


if __name__ == '__main__':
    main()
