#!/usr/bin/env python3

# -------------------------------------------------------
# Script: find_best_images.py
#
# Description:
# This script processes images in a directory, computes quality metrics such as blurriness
# and compression artifacts, and selects the top images with the best quality.
#
# Usage:
# ./find_best_images.py [image_directory] [options]
#
# - [image_directory]: The path to the input image directory.
#
# Options:
# -n NUM, --num-images NUM      Number of top images to select (default: 10).
# -o DIR, --output-dir DIR      Output directory to copy the selected images.
# -v, --verbose                 Enable verbose output.
# -d, --debug                   Enable debug logging.
# -L WEIGHT, --laplacian-weight WEIGHT
#                               Weight for the laplacian (blurriness) score (default: 1.0).
# -A WEIGHT, --artifact-weight WEIGHT
#                               Weight for the artifact score (default: 1.0).
#
# Template: ubuntu22.04
#
# Requirements:
# - OpenCV (install via: apt-get install -y python3-opencv)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import shutil
from typing import List, Dict, Optional, Any

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Find the top X images with the best quality in a directory.'
    )
    parser.add_argument(
        'image_directory',
        type=str,
        help='The path to the input image directory.'
    )
    parser.add_argument(
        '-n', '--num-images',
        type=int,
        default=10,
        help='Number of top images to select.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Output directory to copy the selected images.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging.'
    )
    parser.add_argument(
        '-L', '--laplacian-weight',
        type=float,
        default=1.0,
        help='Weight for the laplacian (blurriness) score.'
    )
    parser.add_argument(
        '-A', '--artifact-weight',
        type=float,
        default=1.0,
        help='Weight for the artifact score.'
    )
    return parser.parse_args()


def setup_logging(verbose: bool, debug: bool) -> None:
    """
    Sets up the logging configuration based on the verbose and debug flags.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def get_image_files(image_directory: str) -> List[str]:
    """
    Retrieves a list of image file paths from the given directory.
    """
    if not os.path.isdir(image_directory):
        logging.error(f"Input path '{image_directory}' is not a directory.")
        sys.exit(1)

    image_files: List[str] = [
        os.path.join(image_directory, f)
        for f in os.listdir(image_directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    ]
    if not image_files:
        logging.error(f"No image files found in directory '{image_directory}'.")
        sys.exit(1)

    return image_files


def calculate_laplacian_variance(gray_image: np.ndarray) -> float:
    """
    Calculates the Laplacian variance of a grayscale image.
    This is a measure of blurriness: the higher the variance, the less blurry the image.
    """
    laplacian: np.ndarray = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance: float = float(laplacian.var())
    return variance


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


def process_image(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Processes a single image to compute quality scores.
    """
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
        return None

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate blurriness score
    laplacian_score: float = calculate_laplacian_variance(gray)

    # Calculate compression artifact score
    artifact_score: float = calculate_artifact_score(image)

    result: Dict[str, Any] = {
        'image': image_path,
        'laplacian_score': laplacian_score,
        'artifact_score': artifact_score,
    }

    logging.info(
        f"Processed '{image_path}': Laplacian = {laplacian_score:.2f}, "
        f"Artifact Score = {artifact_score:.2f}"
    )

    return result


def process_images(image_files: List[str]) -> List[Dict[str, Any]]:
    """
    Processes a list of image files and computes quality scores for each.
    """
    results: List[Dict[str, Any]] = []

    for img_file in image_files:
        result = process_image(img_file)
        if result:
            results.append(result)

    return results


def normalize_scores(scores: List[float], higher_better: bool = True) -> List[float]:
    """
    Normalizes a list of scores to the range [0, 1].
    If higher_better is True, higher scores will have higher normalized values.
    If higher_better is False, lower scores will have higher normalized values.
    """
    min_score: float = min(scores)
    max_score: float = max(scores)
    if max_score == min_score:
        # All scores are the same
        return [1.0] * len(scores)
    else:
        if higher_better:
            return [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            return [(max_score - s) / (max_score - min_score) for s in scores]


def collect_and_normalize_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collects scores from results, normalizes them, and updates the results with normalized scores.
    """
    # Collect scores
    laplacian_scores: List[float] = [res['laplacian_score'] for res in results]
    artifact_scores: List[float] = [res['artifact_score'] for res in results]

    # Normalize blurriness scores (higher is better)
    norm_laplacian_scores: List[float] = normalize_scores(laplacian_scores, higher_better=True)

    # Normalize artifact scores (lower is better, so invert them)
    norm_artifact_scores: List[float] = normalize_scores(artifact_scores, higher_better=False)

    # Update results with normalized scores
    for res, nl, na in zip(results, norm_laplacian_scores, norm_artifact_scores):
        res['norm_laplacian_score'] = nl
        res['norm_artifact_score'] = na

    return results


def compute_overall_scores(results: List[Dict[str, Any]], laplacian_weight: float, artifact_weight: float) -> List[Dict[str, Any]]:
    """
    Computes overall quality scores based on normalized laplacian and artifact scores,
    weighted by the given weights.
    """
    total_weight = laplacian_weight + artifact_weight
    if total_weight == 0:
        # Avoid division by zero
        total_weight = 1.0
        laplacian_weight = 1.0
        artifact_weight = 1.0

    for res in results:
        nl = res['norm_laplacian_score']
        na = res['norm_artifact_score']

        overall_score = ((nl * laplacian_weight) + (na * artifact_weight)) / total_weight
        res['overall_score'] = overall_score

    return results


def select_top_images(results: List[Dict[str, Any]], num_images: int) -> List[Dict[str, Any]]:
    """
    Sorts the results based on overall quality score and selects the top images.
    """
    # Sort results based on overall quality score
    results.sort(key=lambda x: x['overall_score'], reverse=True)

    # Select top images
    top_results: List[Dict[str, Any]] = results[:num_images]

    return top_results


def output_results(top_results: List[Dict[str, Any]], laplacian_weight: float, artifact_weight: float) -> None:
    """
    Outputs the top results to the console.
    """
    print(f"Using Weights: Laplacian={laplacian_weight}, Artifacts={artifact_weight}\n")
    for res in top_results:
        print(f"Image: {res['image']}")
        print(f"  Laplacian Score: {res['laplacian_score']:.2f} (Normalized: {res['norm_laplacian_score']:.2f})")
        print(f"  Artifact Score: {res['artifact_score']:.2f} (Normalized: {res['norm_artifact_score']:.2f})")
        print(f"  Overall Quality Score: {res['overall_score']:.2f}")
        print()


def copy_images(top_results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Copies the selected top images to the specified output directory,
    prepending the overall quality score to the filename.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory '{output_dir}'.")
        except Exception as e:
            logging.error(f"Could not create output directory '{output_dir}': {e}")
            sys.exit(1)
    for res in top_results:
        src: str = res['image']
        filename: str = os.path.basename(src)
        formatted_score: str = f"{res['overall_score']:.2f}"
        dst_filename: str = f"{formatted_score}_{filename}"
        dst: str = os.path.join(output_dir, dst_filename)
        try:
            shutil.copy2(src, dst)
            logging.info(f"Copied '{src}' to '{dst}'.")
        except Exception as e:
            logging.error(f"Failed to copy '{src}' to '{dst}': {e}")


def main() -> None:
    """
    Main function to orchestrate the image quality assessment process.
    """
    args: argparse.Namespace = parse_arguments()
    setup_logging(args.verbose, args.debug)

    image_files: List[str] = get_image_files(args.image_directory)
    if not image_files:
        logging.error("No valid images were found.")
        sys.exit(1)

    results: List[Dict[str, Any]] = process_images(image_files)
    if not results:
        logging.error("No valid images were processed.")
        sys.exit(1)

    results = collect_and_normalize_scores(results)
    results = compute_overall_scores(results, args.laplacian_weight, args.artifact_weight)
    top_results = select_top_images(results, args.num_images)

    output_results(top_results, args.laplacian_weight, args.artifact_weight)

    if args.output_dir:
        copy_images(top_results, args.output_dir)

    sys.exit(0)


if __name__ == '__main__':
    main()
