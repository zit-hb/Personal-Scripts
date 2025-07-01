#!/usr/bin/env python3

# -------------------------------------------------------
# Script: clean_outlines.py
#
# Description:
# Cleans up a scanned or photographed outline image by removing shadows,
# artifacts, and imperfections to produce a clear single-color image.
# Supports multiple line-strengths via Otsu, adaptive thresholding, or Canny,
# plus optional illumination normalization and morphological cleanup.
#
# Usage:
#   ./clean_outlines.py [options] input_image output_image
#
# Arguments:
#   - input_image: Path to the input image file.
#   - output_image: Path to save the cleaned output image.
#
# Options:
#   -c, --color COLOR              Specify the line color in hex format (default: #000000).
#   -f, --bg-color COLOR           Specify a background color in hex format (e.g., #FFFFFF).
#   -m, --method METHOD            Line detection method: otsu, adaptive, canny (default: otsu).
#   -s, --canny-sigma FLOAT        Sigma for Canny thresholds (default: 0.33).
#   -b, --adaptive-block-size INT  Block size (odd, ≥3) for adaptive thresholding (default: 11).
#   -C, --adaptive-C INT           C constant for adaptive thresholding (default: 2).
#   -i, --illum-kernel-size INT    Kernel size for illumination normalization (odd ≥3; default: 0 = off).
#   -k, --morph-kernel-size INT    Kernel size for morphological closing (odd ≥3; default: 3).
#   -v, --verbose                  Enable verbose logging (INFO level).
#   -vv, --debug                   Enable debug logging (DEBUG level).
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
        "-m",
        "--method",
        type=str,
        choices=["otsu", "adaptive", "canny"],
        default="otsu",
        help="Line detection method: otsu, adaptive, canny (default: otsu).",
    )
    parser.add_argument(
        "-s",
        "--canny-sigma",
        type=float,
        default=0.33,
        help="Sigma for Canny thresholds (default: 0.33).",
    )
    parser.add_argument(
        "-b",
        "--adaptive-block-size",
        type=int,
        default=11,
        help="Block size (odd ≥3) for adaptive thresholding (default: 11).",
    )
    parser.add_argument(
        "-C",
        "--adaptive-C",
        type=int,
        default=2,
        help="C constant for adaptive thresholding (default: 2).",
    )
    parser.add_argument(
        "-i",
        "--illum-kernel-size",
        type=int,
        default=0,
        help="Kernel size for illumination normalization (odd ≥3; default: 0 = off).",
    )
    parser.add_argument(
        "-k",
        "--morph-kernel-size",
        type=int,
        default=3,
        help="Kernel size for morphological closing (odd ≥3; default: 3).",
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
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: "cv2.Mat", kernel_size: int) -> "cv2.Mat":
    """
    Performs illumination normalization by background estimation.
    """
    logging.debug(f"Normalizing illumination with kernel size {kernel_size}.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        logging.warning(
            f"Illumination kernel size {kernel_size} invalid; skipping normalization."
        )
        return gray
    # Estimate background via morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # Avoid division by zero
    background = cv2.add(background, 1)
    normed = cv2.divide(gray, background, scale=255)
    return cv2.equalizeHist(normed)


def apply_blur(image: "cv2.Mat", kernel_size: Tuple[int, int] = (5, 5)) -> "cv2.Mat":
    """
    Applies Gaussian blur to the image to reduce noise.
    """
    logging.debug(f"Applying Gaussian blur with kernel size {kernel_size}.")
    return cv2.GaussianBlur(image, kernel_size, 0)


def threshold_image(
    image: "cv2.Mat",
    method: str,
    canny_sigma: float,
    adaptive_block_size: int,
    adaptive_C: int,
) -> "cv2.Mat":
    """
    Detects lines using the specified method and returns a binary image.
    """
    logging.debug(f"Applying threshold method: {method}.")
    if method == "otsu":
        logging.debug("Applying Otsu's thresholding.")
        _, thresholded = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        bs = adaptive_block_size
        if bs < 3 or bs % 2 == 0:
            logging.warning(f"Adaptive block size {bs} invalid; using 11.")
            bs = 11
        logging.debug(
            f"Applying adaptive thresholding with block size {bs}, C={adaptive_C}."
        )
        thresholded = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            adaptive_C,
        )
    elif method == "canny":
        logging.debug(f"Applying Canny edge detection with sigma={canny_sigma}.")
        v = np.median(image)
        lower = int(max(0, (1.0 - canny_sigma) * v))
        upper = int(min(255, (1.0 + canny_sigma) * v))
        logging.debug(f"Canny thresholds: lower={lower}, upper={upper}.")
        edges = cv2.Canny(image, lower, upper)
        thresholded = cv2.bitwise_not(edges)
    else:
        logging.error(f"Unknown thresholding method '{method}'.")
        sys.exit(1)
    return thresholded


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
    inverted = cv2.bitwise_not(thresholded)
    alpha = inverted
    b, g, r = line_color
    mask = thresholded == 0

    b_channel = np.zeros_like(thresholded, dtype=np.uint8)
    g_channel = np.zeros_like(thresholded, dtype=np.uint8)
    r_channel = np.zeros_like(thresholded, dtype=np.uint8)

    b_channel[mask] = b
    g_channel[mask] = g
    r_channel[mask] = r

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
    combined = np.full_like(
        cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR), background_color, dtype=np.uint8
    )
    mask = thresholded == 0
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
    method: str,
    canny_sigma: float,
    adaptive_block_size: int,
    adaptive_C: int,
    illum_kernel_size: int,
    morph_kernel_size: int,
) -> None:
    """
    Cleans the input image and saves the processed image to the output path.
    """
    logging.info(f"Starting cleanup of image '{input_path}'.")
    image = load_image(input_path)
    grayscale = convert_to_grayscale(image)

    # optional illumination normalization to reduce shadows
    if illum_kernel_size > 0:
        grayscale = normalize_illumination(grayscale, illum_kernel_size)

    blurred = apply_blur(grayscale)
    thresholded = threshold_image(
        blurred, method, canny_sigma, adaptive_block_size, adaptive_C
    )

    # morphological closing to remove small gaps/artifacts
    if morph_kernel_size >= 3 and morph_kernel_size % 2 == 1:
        logging.debug(
            f"Applying morphological closing with kernel size {morph_kernel_size}."
        )
        mker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, mker)
    else:
        logging.warning(
            f"Invalid morph-kernel-size {morph_kernel_size}; skipping closing."
        )

    if bg_color_color:
        processed_image = bg_color(thresholded, bg_color_color, line_color)
    else:
        if supports_transparency(output_path):
            processed_image = create_transparent_image(thresholded, line_color)
        else:
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
        method=args.method,
        canny_sigma=args.canny_sigma,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_C=args.adaptive_C,
        illum_kernel_size=args.illum_kernel_size,
        morph_kernel_size=args.morph_kernel_size,
    )


if __name__ == "__main__":
    main()
