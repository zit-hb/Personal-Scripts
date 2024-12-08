#!/usr/bin/env python3

# -------------------------------------------------------
# Script: merge_images_advanced.py
#
# Description:
# This script intelligently merges multiple images into a single image by aligning them
# using feature detection and applying advanced blending techniques. It leverages
# machine learning-based feature detectors to ensure accurate alignment and seamless
# blending, resulting in a clear and cohesive merged image.
#
# Usage:
# ./merge_images_advanced.py [input_path] [options]
#
# Arguments:
#   - [input_path]: The path to the input image file or directory.
#
# Options:
#   -r, --recursive            Process directories recursively.
#   --blend-mode BLEND_MODE    Mode to blend images.
#                              Choices: "average", "median", "max", "min".
#                              (default: "average")
#   --alignment-method METHOD  Feature detection method for alignment.
#                              Choices: "ORB", "SIFT", "SURF".
#                              (default: "ORB")
#   --resolution-mode RES_MODE
#                              Mode to determine output image resolution.
#                              Choices: "smallest", "biggest", "middle", "custom".
#                              (default: "middle")
#   --width WIDTH              Custom width for the output image (required if resolution-mode is "custom").
#   --height HEIGHT            Custom height for the output image (required if resolution-mode is "custom").
#   -o OUTPUT_FILE, --output OUTPUT_FILE
#                              Output file name for the merged image (default: "merged_image_advanced.png").
#
# Template: ubuntu22.04
#
# Requirements:
#   - OpenCV (install via: apt-get install -y python3-opencv opencv-data)
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
import cv2
import numpy as np


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Intelligently merge multiple images into a single image with alignment and advanced blending.'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input image file or directory.'
    )
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Process directories recursively.'
    )
    parser.add_argument(
        '--blend-mode',
        type=str,
        default='average',
        choices=['average', 'median', 'max', 'min'],
        help='Mode to blend images (default: average).'
    )
    parser.add_argument(
        '--alignment-method',
        type=str,
        default='ORB',
        choices=['ORB', 'SIFT', 'SURF'],
        help='Feature detection method for alignment (default: ORB).'
    )
    parser.add_argument(
        '--resolution-mode',
        type=str,
        default='middle',
        choices=['smallest', 'biggest', 'middle', 'custom'],
        help='Mode to determine output image resolution (default: middle).'
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Custom width for the output image (required if resolution-mode is "custom").'
    )
    parser.add_argument(
        '--height',
        type=int,
        help='Custom height for the output image (required if resolution-mode is "custom").'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='merged_image_advanced.png',
        help='Output file name for the merged image (default: merged_image_advanced.png).'
    )
    args = parser.parse_args()

    # Validate custom resolution arguments
    if args.resolution_mode == 'custom':
        if args.width is None or args.height is None:
            parser.error('--width and --height must be specified when resolution-mode is "custom".')

    return args


def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def collect_images(input_path, recursive):
    """
    Collects all image files from the input path.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_files = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith(supported_extensions):
            image_files.append(input_path)
        else:
            logging.error(f"File '{input_path}' is not a supported image format.")
    elif os.path.isdir(input_path):
        if recursive:
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_path):
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(input_path, file))
    else:
        logging.error(f"Input path '{input_path}' is neither a file nor a directory.")
        sys.exit(1)

    if not image_files:
        logging.error(f"No image files found in the specified path '{input_path}'.")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to merge.")
    return image_files


def determine_output_size(image_files, resolution_mode, custom_width=None, custom_height=None):
    """
    Determines the output image size based on the resolution mode.
    """
    sizes = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except Exception as e:
            logging.error(f"Failed to open image '{img_path}': {e}")
            sys.exit(1)

    if resolution_mode == 'smallest':
        width = min(size[0] for size in sizes)
        height = min(size[1] for size in sizes)
    elif resolution_mode == 'biggest':
        width = max(size[0] for size in sizes)
        height = max(size[1] for size in sizes)
    elif resolution_mode == 'middle':
        sorted_widths = sorted(size[0] for size in sizes)
        sorted_heights = sorted(size[1] for size in sizes)
        width = sorted_widths[len(sorted_widths) // 2]
        height = sorted_heights[len(sorted_heights) // 2]
    elif resolution_mode == 'custom':
        width = custom_width
        height = custom_height
    else:
        logging.error(f"Unknown resolution mode '{resolution_mode}'.")
        sys.exit(1)

    logging.info(f"Output image size set to: {width}x{height}")
    return width, height


def load_and_resize_images(image_files, target_size):
    """
    Loads images, resizes them to the target size, and returns them as a list of NumPy arrays.
    """
    resized_images = []
    for idx, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img_np = np.array(img)
                resized_images.append(img_np)
                logging.info(f"Loaded and resized image {idx + 1}/{len(image_files)}: '{img_path}'")
        except Exception as e:
            logging.error(f"Failed to process image '{img_path}': {e}")
            sys.exit(1)
    return resized_images


def align_images(images, alignment_method):
    """
    Aligns all images to the first image in the list using feature detection and homography.
    """
    if len(images) == 1:
        logging.info("Only one image provided. No alignment needed.")
        return images

    reference_image = images[0]
    aligned_images = [reference_image]

    # Initialize feature detector
    if alignment_method == 'SIFT':
        try:
            sift = cv2.SIFT_create()
            detector = sift
            logging.info("Using SIFT for feature detection.")
        except AttributeError:
            logging.error("SIFT is not available. Ensure that you have opencv-contrib-python installed.")
            sys.exit(1)
    elif alignment_method == 'SURF':
        try:
            surf = cv2.xfeatures2d.SURF_create()
            detector = surf
            logging.info("Using SURF for feature detection.")
        except AttributeError:
            logging.error("SURF is not available. Ensure that you have opencv-contrib-python installed.")
            sys.exit(1)
    else:
        orb = cv2.ORB_create(5000)
        detector = orb
        logging.info("Using ORB for feature detection.")

    # Define matcher
    if alignment_method in ['SIFT', 'SURF']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for idx, img in enumerate(images[1:], start=2):
        logging.info(f"Aligning image {idx}/{len(images)} to the reference image.")
        keypoints1, descriptors1 = detector.detectAndCompute(reference_image, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img, None)

        if descriptors1 is None or descriptors2 is None:
            logging.warning(f"Not enough descriptors found in image {idx}. Skipping alignment.")
            aligned_images.append(img)
            continue

        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            logging.warning(f"Not enough matches found in image {idx}. Skipping alignment.")
            aligned_images.append(img)
            continue

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        try:
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None:
                aligned_img = cv2.warpPerspective(img, M, (reference_image.shape[1], reference_image.shape[0]))
                aligned_images.append(aligned_img)
                logging.info(f"Image {idx} aligned successfully.")
            else:
                logging.warning(f"Homography could not be computed for image {idx}. Skipping alignment.")
                aligned_images.append(img)
        except cv2.error as e:
            logging.error(f"Error during alignment of image {idx}: {e}")
            aligned_images.append(img)

    return aligned_images


def blend_images(images, blend_mode):
    """
    Blends multiple aligned images into a single image using the specified blend mode.
    """
    logging.info(f"Blending images using '{blend_mode}' mode.")
    stacked_images = np.stack(images, axis=3)

    if blend_mode == 'average':
        blended = np.mean(stacked_images, axis=3).astype(np.uint8)
    elif blend_mode == 'median':
        blended = np.median(stacked_images, axis=3).astype(np.uint8)
    elif blend_mode == 'max':
        blended = np.max(stacked_images, axis=3).astype(np.uint8)
    elif blend_mode == 'min':
        blended = np.min(stacked_images, axis=3).astype(np.uint8)
    else:
        logging.error(f"Unknown blend mode '{blend_mode}'.")
        sys.exit(1)

    return blended


def save_image(image_np, output_path):
    """
    Saves a NumPy array as an image to the specified path.
    """
    try:
        image = Image.fromarray(image_np)
        image.save(output_path)
        logging.info(f"Merged image saved as '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save merged image '{output_path}': {e}")
        sys.exit(1)


def main():
    args = parse_arguments()
    setup_logging()

    input_path = args.input_path
    recursive = args.recursive
    blend_mode = args.blend_mode
    alignment_method = args.alignment_method
    resolution_mode = args.resolution_mode
    custom_width = args.width
    custom_height = args.height
    output_file = args.output

    image_files = collect_images(input_path, recursive)
    output_size = determine_output_size(
        image_files,
        resolution_mode,
        custom_width=custom_width,
        custom_height=custom_height
    )

    # Load and resize images
    images = load_and_resize_images(image_files, output_size)

    # Align images
    aligned_images = align_images(images, alignment_method)

    # Blend images
    merged_image_np = blend_images(aligned_images, blend_mode)

    # Save the merged image
    save_image(merged_image_np, output_file)


if __name__ == '__main__':
    main()
