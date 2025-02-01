#!/usr/bin/env python3

# -------------------------------------------------------
# Script: convert_images.py
#
# Description:
# This script converts image files from one format to another.
# It supports various image formats and provides options for
# dry runs, verbosity, recursive directory traversal, output
# directories, quality settings, and more.
#
# Usage:
# ./convert_images.py [options] [directory|image]
#
# Arguments:
#    - [directory|image]: The image file or directory to process.
#
# Options:
#   -f FORMAT, --format FORMAT           Output image format (default: jpeg).
#   -q QUALITY, --quality QUALITY        Output image quality (1-100). Defaults to 85.
#   -o DIR, --output-dir DIR             Output directory for the converted images.
#   -r, --recursive                      Process directories recursively.
#   -n, --dry-run                        Show what would be done without making any changes.
#   -v, --verbose                        Enable verbose output.
#   -h, --help                           Display this help message.
#
# Template: ubuntu22.04
#
# Requirements:
#   - Pillow (install via: pip install Pillow==11.0.0)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from PIL import Image
from typing import List


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert images from one format to another."
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="jpeg",
        help="Output image format.",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=85,
        help="Output image quality (1-100).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for the converted images.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively.",
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
        "path",
        nargs="+",
        help="The image file or directory to process.",
    )
    args = parser.parse_args()

    # Validate quality value
    if not (1 <= args.quality <= 100):
        parser.error("Quality must be between 1 and 100.")

    return args


def setup_logging(verbose: bool):
    """Sets up the logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_images(paths: List[str], recursive: bool) -> List[str]:
    """Collects all image files from the provided paths."""
    supported_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp")
    image_files = []

    for path in paths:
        if os.path.isfile(path):
            if path.lower().endswith(supported_extensions):
                image_files.append(path)
            else:
                logging.warning(
                    f"File '{path}' is not a supported image format, skipping."
                )
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
            logging.warning(
                f"Path '{path}' is neither a file nor a directory, skipping."
            )

    if not image_files:
        logging.error("No image files found in the specified path(s).")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to process.")
    return image_files


def convert_image(
    img_path: str, output_format: str, quality: int, output_dir: str, dry_run: bool
):
    """Converts a single image to the specified format."""
    try:
        with Image.open(img_path) as img:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_file = f"{base_name}.{output_format.lower()}"
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_file)
            else:
                output_path = os.path.join(os.path.dirname(img_path), output_file)

            if dry_run:
                logging.info(
                    f"Would convert '{img_path}' to '{output_path}' with format '{output_format}' and quality {quality}"
                )
                return

            if img.mode in ("RGBA", "LA") and output_format.lower() in ("jpg", "jpeg"):
                # JPEG doesn't support transparency; need to convert
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                img = background
            else:
                img = img.convert("RGB")

            img.save(output_path, format=output_format.upper(), quality=quality)
            logging.info(f"Converted '{img_path}' to '{output_path}'")
    except Exception as e:
        logging.error(f"Failed to convert image '{img_path}': {e}")


def main():
    """Main function that orchestrates the image conversion process."""
    args = parse_arguments()
    setup_logging(args.verbose)

    image_files = collect_images(args.path, args.recursive)

    for img_path in image_files:
        convert_image(
            img_path, args.format, args.quality, args.output_dir, args.dry_run
        )


if __name__ == "__main__":
    main()
