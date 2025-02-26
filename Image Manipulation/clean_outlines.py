#!/usr/bin/env python3

# -------------------------------------------------------
# Script: clean_outlines.py
#
# Description:
# Cleans up a scanned or photographed outline image by removing shadows,
# artifacts, and imperfections to produce a clear single-color image.
#
# Usage:
#   ./clean_outlines.py [options] input_image output_image
#
# Arguments:
#   - input_image: Path to the input image file.
#   - output_image: Path to save the cleaned output image.
#
# Options:
#   -c, --color COLOR       Specify the line color in hex format (default: #000000).
#   -f, --bg-color COLOR    Specify a background color in hex format (e.g., #FFFFFF).
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu22.04
#
# Requirements:
#   - OpenCV (install via: apt-get install -y python3-opencv)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import re
from typing import Tuple, Optional

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Clean up an image by removing imperfections and producing a clear outline."
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "output_image",
        type=str,
        help="Path to save the cleaned output image.",
    )
    parser.add_argument(
        "-c",
        "--color",
        type=str,
        metavar="COLOR",
        default="#000000",
        help="Specify the line color in hex format (default: #000000).",
    )
    parser.add_argument(
        "-f",
        "--bg-color",
        type=str,
        metavar="COLOR",
        help="Specify a background color in hex format (e.g., #FFFFFF).",
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


def parse_hex_color(hex_color: str) -> Optional[Tuple[int, int, int]]:
    """
    Parses a hex color string and returns a BGR tuple.
    """
    logging.debug(f"Parsing hex color '{hex_color}'.")
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_color)
    if not match:
        logging.error(
            f"Invalid color format '{hex_color}'. Please provide a hex color like '#FFFFFF'."
        )
        return None
    hex_value = match.group(1)
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    logging.debug(f"Parsed color: B={b}, G={g}, R={r}.")
    return (b, g, r)


def load_image(image_path: str) -> "cv2.Mat":
    """
    Loads an image from the specified path.
    """
    logging.debug(f"Loading image from '{image_path}'.")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        logging.error(
            f"Failed to load image '{image_path}'. Ensure the file exists and is an image."
        )
        sys.exit(1)
    logging.debug(f"Image '{image_path}' loaded successfully.")
    return image


def convert_to_grayscale(image: "cv2.Mat") -> "cv2.Mat":
    """
    Converts the image to grayscale.
    """
    logging.debug("Converting image to grayscale.")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def apply_blur(image: "cv2.Mat", kernel_size: Tuple[int, int] = (5, 5)) -> "cv2.Mat":
    """
    Applies Gaussian blur to the image to reduce noise.
    """
    logging.debug(f"Applying Gaussian blur with kernel size {kernel_size}.")
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def threshold_image(image: "cv2.Mat") -> "cv2.Mat":
    """
    Applies Otsu's thresholding to convert the image to black and white.
    """
    logging.debug("Applying Otsu's thresholding.")
    _, thresholded_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded_image


def supports_transparency(output_path: str) -> bool:
    """
    Determines if the output format supports transparency based on the file extension.
    """
    _, ext = os.path.splitext(output_path)
    transparent_formats = {
        ".png",
        ".tiff",
        ".tif",
        ".webp",
    }
    is_supported = ext.lower() in transparent_formats
    logging.debug(
        f"Output file extension '{ext}' supports transparency: {is_supported}."
    )
    return is_supported


def create_transparent_image(
    thresholded: "cv2.Mat", line_color: Tuple[int, int, int]
) -> "cv2.Mat":
    """
    Creates a transparent image with colored lines on a transparent background.
    """
    logging.debug("Creating transparent image with colored lines.")
    # Invert thresholded image: lines are black (0), background is white (255)
    inverted = cv2.bitwise_not(thresholded)
    # Create alpha channel based on inverted image
    alpha = inverted
    # Create BGR channels with the specified line color
    b, g, r = line_color
    # Create mask where the actual lines are black == 0
    mask = thresholded == 0

    # Initialize BGR channels
    b_channel = np.zeros_like(thresholded, dtype=np.uint8)
    g_channel = np.zeros_like(thresholded, dtype=np.uint8)
    r_channel = np.zeros_like(thresholded, dtype=np.uint8)

    # Put line color where mask is True
    b_channel[mask] = b
    g_channel[mask] = g
    r_channel[mask] = r

    # Merge BGR and alpha channels
    transparent_image = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return transparent_image


def bg_color(
    thresholded: "cv2.Mat",
    background_color: Tuple[int, int, int],
    line_color: Tuple[int, int, int],
) -> "cv2.Mat":
    """
    Fills the background of the thresholded image with the specified color
    and the lines with the specified color, without adding/saturating.
    """
    logging.debug("Filling background and lines with specified colors.")

    # Create the output image filled with the background color
    combined = np.full_like(
        cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR), background_color, dtype=np.uint8
    )

    # The mask for lines (which are black == 0 in thresholded)
    mask = thresholded == 0

    # Directly overwrite background with line color where mask is True
    combined[mask] = line_color

    return combined


def save_image(image: "cv2.Mat", path: str) -> None:
    """
    Saves the image to the specified path.
    """
    logging.debug(f"Saving image to '{path}'.")
    success = cv2.imwrite(path, image)
    if not success:
        logging.error(f"Failed to save image to '{path}'.")
        sys.exit(1)
    logging.debug(f"Image saved to '{path}' successfully.")


def clean_image(
    input_path: str,
    output_path: str,
    bg_color_color: Optional[Tuple[int, int, int]],
    line_color: Tuple[int, int, int],
) -> None:
    """
    Cleans the input image and saves the processed image to the output path.
    """
    logging.info(f"Starting cleanup of image '{input_path}'.")
    image = load_image(input_path)
    grayscale = convert_to_grayscale(image)
    blurred = apply_blur(grayscale)
    thresholded = threshold_image(blurred)

    if bg_color_color:
        processed_image = bg_color(thresholded, bg_color_color, line_color)
    else:
        if supports_transparency(output_path):
            processed_image = create_transparent_image(thresholded, line_color)
        else:
            # Default to white background if bg_color not provided and format not transparent
            processed_image = bg_color(thresholded, (255, 255, 255), line_color)

    save_image(processed_image, output_path)
    logging.info(f"Image cleaned and saved to '{output_path}'.")


def main() -> None:
    """
    Main function to orchestrate the image cleanup process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    line_color = parse_hex_color(args.color)
    if line_color is None:
        sys.exit(1)

    fill_color = None
    if args.bg_color:
        fill_color = parse_hex_color(args.bg_color)
        if fill_color is None:
            sys.exit(1)

    clean_image(
        args.input_image,
        args.output_image,
        bg_color_color=fill_color,
        line_color=line_color,
    )


if __name__ == "__main__":
    main()
