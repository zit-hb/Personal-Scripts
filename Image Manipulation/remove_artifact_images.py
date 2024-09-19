#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_artifact_images.py
#
# Description:
# This script scans through all images in a specified directory
# (optionally recursively) and uses the `detect_compression_artifacts.py`
# script to determine the quality of each image. Images
# identified as having high compression artifacts are removed from the directory.
#
# Usage:
# ./remove_artifact_images.py [directory] [options]
#
# Arguments:
# - [directory]               The path to the directory containing images.
#
# Options:
# -t THRESHOLD, --threshold THRESHOLD
#                           Threshold for compression artifact detection (default: 1000.0).
# --dry-run                 Simulate the removal of images with high artifacts without deleting them.
# --recursive               Recursively traverse through subdirectories.
# -o OUTPUT_FILE, --output OUTPUT_FILE
#                           Output file to save the results.
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
import subprocess


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Remove images with high compression artifacts using detect_compression_artifacts.py.'
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
        default=1000.0,
        help='Threshold for compression artifact detection (default: 1000.0).'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the removal of images with high artifacts without deleting them.'
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
    return parser.parse_args()


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


def check_image_artifacts(image_path, threshold):
    """
    Determines if an image has high compression artifacts by invoking detect_compression_artifacts.py.
    """
    detect_script = os.path.join(os.path.dirname(__file__), '..', 'Image Recognition', 'detect_compression_artifacts.py')
    detect_command = [
        sys.executable, detect_script,
        image_path,
        '--threshold', str(threshold)
    ]

    try:
        result = subprocess.run(
            detect_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        logging.error(f"Failed to execute detect_compression_artifacts.py for '{image_path}': {e}")
        return False  # Assume not high artifacts if detection fails

    if result.returncode == 1:
        return True
    elif result.returncode == 0:
        return False
    else:
        logging.error(f"Error processing '{image_path}': {result.stderr.strip()}")
        return False  # Assume not high artifacts on unexpected return code


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
    Saves the artifact detection results to the specified output file.
    """
    try:
        with open(output_file, 'w') as f:
            for image, has_artifacts in results:
                status = 'high_artifacts' if has_artifacts else 'low_artifacts'
                f.write(f"{image},{status}\n")
        logging.info(f"Results saved to '{output_file}'.")
    except Exception as e:
        logging.error(f"Failed to write results to '{output_file}': {e}")


def main():
    args = parse_arguments()
    setup_logging()

    directory = args.directory
    threshold = args.threshold
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

    logging.info(f"Processing {len(image_files)} image(s) in '{directory}' with threshold {threshold}.")

    results = []
    any_removed = False

    for img_path in image_files:
        has_artifacts = check_image_artifacts(img_path, threshold)
        results.append((img_path, has_artifacts))
        if has_artifacts:
            any_removed = True
            remove_image(img_path, dry_run)

    if output_file:
        save_results(results, output_file)

    if any_removed:
        if dry_run:
            logging.info("Some images would be removed due to high compression artifacts (dry run).")
        else:
            logging.info("Some images were removed due to high compression artifacts.")
    else:
        logging.info("No images with high compression artifacts found.")


if __name__ == '__main__':
    main()
