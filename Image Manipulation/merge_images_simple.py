#!/usr/bin/env python3

# -------------------------------------------------------
# Script: merge_images_simple.py
#
# Description:
# This script merges all images in a specified directory into a single image by layering them
# with appropriate transparency. It supports recursive directory traversal and provides
# various options for handling images of different sizes and resolutions.
#
# Usage:
# ./merge_images_simple.py [input_path] [options]
#
# Arguments:
#   - [input_path]: The path to the input image file or directory.
#
# Options:
#   -r, --recursive            Process directories recursively.
#   --resize-mode RESIZE_MODE  Mode to handle differing image sizes.
#                              Choices: "center", "zoom", "tile", "stretch", "scale".
#                              (default: "center")
#   --resolution-mode RES_MODE
#                              Mode to determine output image resolution.
#                              Choices: "smallest", "biggest", "middle", "custom".
#                              (default: "middle")
#   --width WIDTH              Custom width for the output image (required if resolution-mode is "custom").
#   --height HEIGHT            Custom height for the output image (required if resolution-mode is "custom").
#   -o OUTPUT_FILE, --output OUTPUT_FILE
#                              Output file name for the merged image (default: "merged_image.png").
#
# Template: ubuntu22.04
#
# Requirements:
# - Pillow (install via: pip install Pillow==11.0.0)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from PIL import Image


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Merge multiple images into a single image with layered transparency.'
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
        '--resize-mode',
        type=str,
        default='center',
        choices=['center', 'zoom', 'tile', 'stretch', 'scale'],
        help='Mode to handle differing image sizes (default: center).'
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
        default='merged_image.png',
        help='Output file name for the merged image (default: merged_image.png).'
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


def resize_image(img, target_size, resize_mode):
    """
    Resizes an image based on the specified resize mode.
    """
    target_width, target_height = target_size
    img_width, img_height = img.size

    if resize_mode == 'center':
        # Create a transparent background
        new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
        # Calculate position to center the image
        paste_x = (target_width - img_width) // 2
        paste_y = (target_height - img_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img

    elif resize_mode == 'zoom':
        # Scale the image to fit within the target size while maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
        paste_x = (target_width - img.size[0]) // 2
        paste_y = (target_height - img.size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img

    elif resize_mode == 'tile':
        # Tile the image to fill the target size
        new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
        for x in range(0, target_width, img_width):
            for y in range(0, target_height, img_height):
                new_img.paste(img, (x, y))
        return new_img

    elif resize_mode == 'stretch':
        # Stretch the image to exactly match the target size
        return img.resize(target_size, Image.Resampling.LANCZOS)

    elif resize_mode == 'scale':
        # Scale the image to fit the target size while maintaining aspect ratio
        img_ratio = img_width / img_height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_ratio)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        return new_img

    else:
        logging.error(f"Unknown resize mode '{resize_mode}'.")
        sys.exit(1)


def merge_images(image_files, output_size, resize_mode):
    """
    Merges multiple images into a single image with layered transparency.
    """
    merged_image = Image.new('RGBA', output_size, (0, 0, 0, 0))
    num_images = len(image_files)
    alpha = int(255 / num_images)  # Calculate transparency per layer

    for idx, img_path in enumerate(image_files):
        try:
            with Image.open(img_path).convert('RGBA') as img:
                resized_img = resize_image(img, output_size, resize_mode)
                # Apply transparency
                alpha_layer = resized_img.split()[3].point(lambda p: p * (alpha / 255))
                resized_img.putalpha(alpha_layer)
                merged_image = Image.alpha_composite(merged_image, resized_img)
                logging.info(f"Layered image {idx + 1}/{num_images}: '{img_path}'")
        except Exception as e:
            logging.error(f"Failed to process image '{img_path}': {e}")
            sys.exit(1)

    return merged_image


def main():
    args = parse_arguments()
    setup_logging()

    input_path = args.input_path
    recursive = args.recursive
    resize_mode = args.resize_mode
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

    merged_image = merge_images(image_files, output_size, resize_mode)

    try:
        merged_image.save(output_file)
        logging.info(f"Merged image saved as '{output_file}'.")
    except Exception as e:
        logging.error(f"Failed to save merged image '{output_file}': {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
