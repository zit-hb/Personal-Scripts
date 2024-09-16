#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_blurriness.py
#
# Description:
# This script checks if an image or images in a directory are blurry by calculating
# a sharpness score using one of several methods:
# - Laplacian Variance
# - Sobel Variance
# - Tenengrad Variance
# It supports batch processing and allows you to set a threshold for blurriness.
#
# Usage:
# ./detect_blurriness.py [image_file|image_directory] [options]
#
# - [image_file]: The path to the input image file.
# - [image_directory]: The path to the input image directory (when using --batch).
#
# Options:
# -t THRESHOLD, --threshold THRESHOLD
#                           Threshold for blurriness detection (default depends on method).
# --batch                   Process a batch of images in a directory.
# --method METHOD           Method to use for blurriness detection (default: laplacian).
#                           Choices are "laplacian", "sobel", or "tenengrad".
# -o OUTPUT_FILE, --output OUTPUT_FILE
#                           Output file to save the results.
#
# Returns exit code 0 if all images are not blurry, 1 if any image is blurry.
#
# Requirements:
# - OpenCV (install via: sudo apt install python3-opencv)
#
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
        description='Detect blurry images using various methods.'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='The path to the input image file or directory (use --batch for directories).'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        help='Threshold for blurriness detection (default depends on method).'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process a batch of images in a directory.'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='laplacian',
        choices=['laplacian', 'sobel', 'tenengrad'],
        help='Method to use for blurriness detection (default: laplacian).'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output file to save the results.'
    )
    args = parser.parse_args()

    # Set default thresholds based on the method if not provided
    if args.threshold is None:
        if args.method == 'laplacian':
            args.threshold = 100.0
        elif args.method == 'sobel':
            args.threshold = 100.0
        elif args.method == 'tenengrad':
            args.threshold = 300.0  # Tenengrad usually has higher variance values

    return args


def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def calculate_laplacian_variance(gray_image):
    """
    Calculates the Laplacian variance of a grayscale image.
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def calculate_sobel_variance(gray_image):
    """
    Calculates the Sobel variance of a grayscale image.
    """
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    variance = sobel.var()
    return variance


def calculate_tenengrad_variance(gray_image):
    """
    Calculates the Tenengrad variance of a grayscale image.
    """
    # Use Sobel operator with ksize=3
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    variance = np.mean(gradient_magnitude ** 2)
    return variance


def is_image_blurry(image_path, threshold, method):
    """
    Determines if an image is blurry.

    Args:
        image_path (str): Path to the image file.
        threshold (float): Threshold for blurriness detection.
        method (str): Method to use for blurriness detection.

    Returns:
        tuple: (bool, float) indicating whether the image is blurry and the computed score.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'laplacian':
        score = calculate_laplacian_variance(gray)
    elif method == 'sobel':
        score = calculate_sobel_variance(gray)
    elif method == 'tenengrad':
        score = calculate_tenengrad_variance(gray)
    else:
        logging.error(f"Unknown method '{method}'.")
        return None, None

    is_blurry = score < threshold
    return is_blurry, score


def process_image(image_path, threshold, method):
    """
    Processes a single image to determine if it is blurry.

    Args:
        image_path (str): Path to the image file.
        threshold (float): Threshold for blurriness detection.
        method (str): Method to use for blurriness detection.

    Returns:
        dict: A dictionary containing the image path, score, and blurriness result.
    """
    is_blurry, score = is_image_blurry(image_path, threshold, method)
    if is_blurry is None:
        return None

    result = {
        'image': image_path,
        'score': score,
        'blurry': is_blurry
    }

    status = 'blurry' if is_blurry else 'sharp'
    logging.info(f"Image '{image_path}' is {status}. (Score: {score:.2f}, Threshold: {threshold}, Method: {method})")
    return result


def main():
    args = parse_arguments()
    setup_logging()

    image_path = args.image_path
    threshold = args.threshold
    method = args.method
    output_file = args.output

    if args.batch:
        if not os.path.isdir(image_path):
            logging.error(f"Input path '{image_path}' is not a directory.")
            sys.exit(1)
        image_files = [
            os.path.join(image_path, f) for f in os.listdir(image_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
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
    any_blurry = False

    for img_file in image_files:
        result = process_image(img_file, threshold, method)
        if result:
            results.append(result)
            if result['blurry']:
                any_blurry = True

    # Save results to output file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                for res in results:
                    status = 'blurry' if res['blurry'] else 'sharp'
                    f.write(f"{res['image']},{res['score']:.2f},{status}\n")
            logging.info(f"Results saved to '{output_file}'.")
        except Exception as e:
            logging.error(f"Failed to write results to '{output_file}': {e}")

    # Exit with code 1 if any image is blurry
    sys.exit(1 if any_blurry else 0)


if __name__ == '__main__':
    main()
