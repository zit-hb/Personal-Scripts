#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_tiles.py
#
# Description:
# This script splits a given image into smaller tiles of specified width and height.
# It ensures that the image dimensions are evenly divisible by the tile size.
# The tiles are saved in an output directory with configurable naming formats.
#
# Usage:
# ./create_tiles.py [options] <image_path>
#
#  <image_path>: The image file to process.
#
# Options:
#   -H HEIGHT, --height HEIGHT           Height of each tile in pixels (default: 1024).
#   -W WIDTH, --width WIDTH              Width of each tile in pixels (default: 1024).
#   -o DIR, --output-dir DIR             Output directory for the tiles (default: "tiles").
#   -f FORMAT, --filename-format FORMAT  Filename format for the tiles (default: "{row}_{column}.{extension}").
#   -v, --verbose                        Enable verbose output.
#   -h, --help                           Display this help message.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - Pillow (install via: pip install Pillow)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from PIL import Image
from typing import Tuple

Image.MAX_IMAGE_PIXELS = 9000000000


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Split a single image into smaller tiles of specified size.'
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=1024,
        help='Height of each tile in pixels.'
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=1024,
        help='Width of each tile in pixels.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='tiles',
        help='Output directory for the tiles.'
    )
    parser.add_argument(
        '-f', '--filename-format',
        type=str,
        default='{row}_{column}.{extension}',
        help='Filename format for the tiles.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='The image file to process.'
    )
    args = parser.parse_args()
    return args


def setup_logging(verbose: bool):
    """Sets up the logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def validate_image_path(image_path: str) -> None:
    """Validates that the image path exists and is a file."""
    if not os.path.isfile(image_path):
        logging.error(f"The path '{image_path}' does not exist or is not a file.")
        sys.exit(1)

    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp')
    if not image_path.lower().endswith(supported_extensions):
        logging.error(f"File '{image_path}' is not a supported image format.")
        sys.exit(1)

    logging.debug(f"Image path '{image_path}' is valid.")


def validate_tile_size(image_size: Tuple[int, int], tile_size: Tuple[int, int]) -> bool:
    """Checks if the image can be evenly divided into tiles of the given size."""
    img_width, img_height = image_size
    tile_width, tile_height = tile_size
    if img_width % tile_width != 0 or img_height % tile_height != 0:
        logging.error(
            f"Image size {img_width}x{img_height} is not evenly divisible by "
            f"tile size {tile_width}x{tile_height}."
        )
        return False
    return True


def generate_filename(format_str: str, row: int, column: int, extension: str) -> str:
    """Generates a filename based on the provided format."""
    return format_str.format(row=row, column=column, extension=extension)


def split_image_into_tiles(
        img_path: str,
        tile_size: Tuple[int, int],
        output_dir: str,
        filename_format: str
):
    """Splits a single image into tiles and saves them."""
    try:
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            tile_width, tile_height = tile_size

            logging.debug(f"Processing image '{img_path}' with size {img_width}x{img_height}.")

            if not validate_tile_size((img_width, img_height), tile_size):
                logging.error(f"Cannot split image '{img_path}' into tiles of size {tile_width}x{tile_height}.")
                sys.exit(1)

            os.makedirs(output_dir, exist_ok=True)

            rows = img_height // tile_height
            columns = img_width // tile_width

            _, ext = os.path.splitext(os.path.basename(img_path))
            ext = ext.lstrip('.').lower()

            for row in range(rows):
                for column in range(columns):
                    left = column * tile_width
                    upper = row * tile_height
                    right = left + tile_width
                    lower = upper + tile_height

                    tile = img.crop((left, upper, right, lower))
                    tile_filename = generate_filename(
                        filename_format,
                        row=row,
                        column=column,
                        extension=ext
                    )
                    tile_path = os.path.join(output_dir, tile_filename)
                    tile.save(tile_path)
                    logging.debug(f"Saved tile '{tile_path}'.")

            logging.info(f"Created {rows * columns} tiles for image '{img_path}' in '{output_dir}'.")

    except Exception as e:
        logging.error(f"Failed to process image '{img_path}': {e}")
        sys.exit(1)


def main():
    """Main function that orchestrates the image tiling process."""
    args = parse_arguments()
    setup_logging(args.verbose)

    validate_image_path(args.image_path)

    tile_size = (args.width, args.height)

    try:
        with Image.open(args.image_path) as img:
            image_size = img.size
            logging.debug(f"Opened image '{args.image_path}' with size {image_size}.")
    except Exception as e:
        logging.error(f"Failed to open image '{args.image_path}': {e}")
        sys.exit(1)

    split_image_into_tiles(
        img_path=args.image_path,
        tile_size=tile_size,
        output_dir=args.output_dir,
        filename_format=args.filename_format
    )

    logging.info("Tile creation process completed successfully.")


if __name__ == '__main__':
    main()
