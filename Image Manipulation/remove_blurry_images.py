#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_blurry_images.py
#
# Description:
# This script scans through all images in a specified directory
# (optionally recursively) and uses the `detect_blurriness.py`
# script to determine the quality of each image. Images
# identified as blurry are removed from the directory.
#
# Usage:
# ./remove_blurry_images.py [directory] [options]
#
# Arguments:
# - [directory]               The path to the directory containing images.
#
# Options:
# -t THRESHOLD, --threshold THRESHOLD
#                           Threshold for blurriness detection (default depends on method).
# --method METHOD           Method to use for blurriness detection (default: laplacian).
#                           Choices are "laplacian", "sobel", or "tenengrad".
# --dry-run                 Simulate the removal of blurry images without deleting them.
# --recursive               Recursively traverse through subdirectories.
# -o OUTPUT_FILE, --output OUTPUT_FILE
#                           Output file to save the results.
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
import subprocess


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Remove blurry images from a directory using detect_blurriness.py.'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='The path to the directory containing images.'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        help='Threshold for blurriness detection (default depends on method).'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='laplacian',
        choices=['laplacian', 'sobel', 'tenengrad'],
        help='Method to use for blurriness detection (default: laplacian).'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the removal of blurry images without deleting them.'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively traverse through subdirectories.'
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )


def get_image_files(directory, recursive=False):
    """
    Retrieves a list of image file paths from the specified directory.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    image_files = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.lower().endswith(supported_extensions):
                image_files.append(os.path.join(directory, file))

    return image_files


def check_image_blurriness(image_path, threshold, method):
    """
    Determines if an image is blurry by invoking detect_blurriness.py.
    """
    detect_script = os.path.join(os.path.dirname(__file__), '..', 'Image Recognition', 'detect_blurriness.py')
    detect_command = [
        sys.executable, detect_script,
        image_path,
        '--threshold', str(threshold),
        '--method', method
    ]

    try:
        result = subprocess.run(
            detect_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        logging.error(f"Failed to execute detect_blurriness.py for '{image_path}': {e}")
        return False  # Assume not blurry if detection fails

    if result.returncode == 1:
        return True
    elif result.returncode == 0:
        return False
    else:
        logging.error(f"Error processing '{image_path}': {result.stderr.strip()}")
        return False  # Assume not blurry on unexpected return code


def remove_image(image_path, dry_run=False):
    """
    Removes the specified image file.
    """
    if dry_run:
        logging.info(f"[Dry Run] Would remove: {image_path}")
    else:
        try:
            os.remove(image_path)
            logging.info(f"Removed: {image_path}")
        except Exception as e:
            logging.error(f"Failed to remove '{image_path}': {e}")


def save_results(results, output_file):
    """
    Saves the blurriness results to the specified output file.
    """
    try:
        with open(output_file, 'w') as f:
            for image, is_blurry in results:
                status = 'blurry' if is_blurry else 'sharp'
                f.write(f"{image},{status}\n")
        logging.info(f"Results saved to '{output_file}'.")
    except Exception as e:
        logging.error(f"Failed to write results to '{output_file}': {e}")


def main():
    args = parse_arguments()
    setup_logging()

    directory = args.directory
    threshold = args.threshold
    method = args.method
    dry_run = args.dry_run
    recursive = args.recursive
    output_file = args.output

    if not os.path.isdir(directory):
        logging.error(f"Input path '{directory}' is not a directory.")
        sys.exit(1)

    image_files = get_image_files(directory, recursive)
    if not image_files:
        logging.warning(f"No image files found in directory '{directory}'.")
        sys.exit(0)

    logging.info(f"Processing {len(image_files)} image(s) in '{directory}' with method '{method}' and threshold {threshold}.")

    results = []
    any_removed = False

    for img_path in image_files:
        is_blurry = check_image_blurriness(img_path, threshold, method)
        results.append((img_path, is_blurry))
        if is_blurry:
            any_removed = True
            remove_image(img_path, dry_run)

    if output_file:
        save_results(results, output_file)

    if any_removed:
        if dry_run:
            logging.info("Some blurry images would be removed (dry run).")
        else:
            logging.info("Some blurry images were removed.")
    else:
        logging.info("No blurry images found.")


if __name__ == '__main__':
    main()
