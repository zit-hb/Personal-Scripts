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
#
# Returns exit code 0 if successful, 1 if any error occurs.
#
# Requirements:
# - OpenCV (install via: sudo apt install python3-opencv)
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
        '-n',
        '--num-images',
        type=int,
        default=10,
        help='Number of top images to select.'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        help='Output directory to copy the selected images.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """
    Sets up the logging configuration based on the verbose flag.
    """
    if verbose:
        level = logging.INFO
    else:
        level = logging.ERROR  # Only show errors by default
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
    """
    laplacian: np.ndarray = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance: float = float(laplacian.var())
    return variance


def calculate_dct_variance(gray_image: np.ndarray) -> float:
    """
    Calculates the variance of high-frequency DCT coefficients of a grayscale image.
    """
    # Perform Discrete Cosine Transform
    dct: np.ndarray = cv2.dct(np.float32(gray_image))

    # Zero out low-frequency coefficients
    dct_high: np.ndarray = dct.copy()
    dct_high[:8, :8] = 0  # Assuming low frequencies are in the top-left 8x8 block

    # Calculate variance of the high-frequency components
    variance: float = float(np.var(dct_high))
    return variance


def process_image(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Processes a single image to compute quality scores.
    """
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
        return None

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate blurriness scores using different methods
    laplacian_score: float = calculate_laplacian_variance(gray)

    # Calculate compression artifact score
    artifact_score: float = calculate_dct_variance(gray)

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
    Collects scores from results, normalizes them, and updates the results with normalized and overall scores.
    """
    # Collect scores
    laplacian_scores: List[float] = [res['laplacian_score'] for res in results]
    artifact_scores: List[float] = [res['artifact_score'] for res in results]

    # Normalize blurriness scores (higher is better)
    norm_laplacian_scores: List[float] = normalize_scores(laplacian_scores, higher_better=True)

    # Normalize artifact scores (lower is better, so invert them)
    norm_artifact_scores: List[float] = normalize_scores(artifact_scores, higher_better=False)

    # Compute overall quality score by combining blurriness and artifact scores
    overall_scores: List[float] = [
        (nl + na) / 2
        for nl, na in zip(norm_laplacian_scores, norm_artifact_scores)
    ]

    # Update results with normalized and overall scores
    for res, nl, na, oscore in zip(
        results, norm_laplacian_scores, norm_artifact_scores, overall_scores
    ):
        res['norm_laplacian_score'] = nl
        res['norm_artifact_score'] = na
        res['overall_score'] = oscore

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


def output_results(top_results: List[Dict[str, Any]]) -> None:
    """
    Outputs the top results to the console.
    """
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
    setup_logging(args.verbose)

    image_files: List[str] = get_image_files(args.image_directory)
    if not image_files:
        logging.error("No valid images were found.")
        sys.exit(1)

    results: List[Dict[str, Any]] = process_images(image_files)
    if not results:
        logging.error("No valid images were processed.")
        sys.exit(1)

    results = collect_and_normalize_scores(results)
    top_results = select_top_images(results, args.num_images)

    output_results(top_results)

    if args.output_dir:
        copy_images(top_results, args.output_dir)

    sys.exit(0)


if __name__ == '__main__':
    main()
