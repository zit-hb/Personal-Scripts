#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_compression_artifacts.py
#
# Description:
# This script detects images that exhibit high compression artifacts, such as JPEG artifacts.
# It analyzes images using a blockiness-based metric to quantify the presence of compression artifacts.
# It supports batch processing and allows you to set a threshold for artifact detection.
#
# Usage:
# ./detect_compression_artifacts.py [image_file|image_directory] [options]
#
# Arguments:
#   - [image_file]: The path to the input image file.
#   - [image_directory]: The path to the input image directory (when using --batch/-B).
#
# Options:
#   -t THRESHOLD, --threshold THRESHOLD    Threshold for compression artifact detection (default: 1000.0).
#   -B, --batch                            Process a batch of images in a directory.
#   -o OUTPUT_FILE, --output OUTPUT_FILE   Output file to save the results.
#
# Returns: Exit code 0 if all images have low compression artifacts, 1 if any image has high artifacts.
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

import cv2
import numpy as np


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect images with high compression artifacts using a blockiness-based method."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="The path to the input image file or directory (use --batch/-B for directories).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1000.0,
        help="Threshold for compression artifact detection.",
    )
    parser.add_argument(
        "-B",
        "--batch",
        action="store_true",
        help="Process a batch of images in a directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file to save the results.",
    )
    return parser.parse_args()


def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def is_image_compressed(image_path, threshold):
    """
    Determines if an image has high compression artifacts using the blockiness-based method.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
        return None, None

    score = calculate_artifact_score(image)
    has_artifacts = score > threshold
    return has_artifacts, score


def process_image(image_path, threshold):
    """
    Processes a single image to determine if it has high compression artifacts.
    """
    has_artifacts, score = is_image_compressed(image_path, threshold)
    if has_artifacts is None:
        return None

    result = {"image": image_path, "score": score, "high_artifacts": has_artifacts}

    status = "has high artifacts" if has_artifacts else "low artifacts"
    logging.info(
        f"Image '{image_path}' {status}. (Score: {score:.2f}, Threshold: {threshold})"
    )
    return result


def main():
    args = parse_arguments()
    setup_logging()

    image_path = args.image_path
    threshold = args.threshold
    output_file = args.output

    if args.batch:
        if not os.path.isdir(image_path):
            logging.error(f"Input path '{image_path}' is not a directory.")
            sys.exit(1)
        image_files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]
        if not image_files:
            logging.error(f"No image files found in directory '{image_path}'.")
            sys.exit(1)
    else:
        if not os.path.isfile(image_path):
            logging.error(f"Input file '{image_path}' does not exist.")
            sys.exit(1)
        image_files = [image_path]

    results = []
    any_high_artifacts = False

    for img_file in image_files:
        result = process_image(img_file, threshold)
        if result:
            results.append(result)
            if result["high_artifacts"]:
                any_high_artifacts = True

    # Save results to output file if specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                for res in results:
                    status = (
                        "high_artifacts" if res["high_artifacts"] else "low_artifacts"
                    )
                    f.write(f"{res['image']},{res['score']:.2f},{status}\n")
            logging.info(f"Results saved to '{output_file}'.")
        except Exception as e:
            logging.error(f"Failed to write results to '{output_file}': {e}")

    # Exit with code 1 if any image has high compression artifacts
    sys.exit(1 if any_high_artifacts else 0)


if __name__ == "__main__":
    main()
