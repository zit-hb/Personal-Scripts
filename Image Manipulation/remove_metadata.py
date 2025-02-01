#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_metadata.py
#
# Description:
# This script removes removable metadata from images.
# By default, it overwrites the original images with
# newly-created versions that contain only the pixel data
# necessary to display the image correctly. It also provides
# an optional feature to attempt the removal or reduction
# of embedded watermarks with minimal visible changes to
# the human eye, using a naive median filter approach.
#
# Usage:
# ./remove_metadata.py [options] [directory|image ...]
#
# Arguments:
#   - [directory|image ...]: One or more paths to images or
#                            directories containing images.
#
# Options:
#   -o DIR, --output-dir DIR   Output directory for the images
#                              with metadata removed. If not
#                              specified, the images are
#                              overwritten in-place.
#   -r, --recursive            Process directories recursively.
#   -w, --remove-watermark     Attempt to remove or reduce
#                              embedded watermarks with minimal
#                              visible changes (naive approach).
#   -f FORMAT, --format FORMAT Force output image format (e.g. png, jpeg, ...).
#                              By default, the image is saved in its original format.
#   -v, --verbose              Enable verbose output.
#   -h, --help                 Display this help message.
#
# Template: ubuntu22.04
#
# Requirements:
#   - Pillow (install via: pip install Pillow==11.0.0)
#   - piexif (install via: pip install piexif==1.1.3)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List

from PIL import Image, ImageFilter
import piexif

# A simple map from forced format strings to recommended extensions
FORMAT_EXTENSION_MAP = {
    "jpg": ".jpg",
    "jpeg": ".jpeg",
    "png": ".png",
    "tiff": ".tiff",
    "bmp": ".bmp",
    "gif": ".gif",
    "webp": ".webp",
}


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove (almost) all metadata from images, optionally also reducing watermarks."
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for images with metadata removed.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively.",
    )
    parser.add_argument(
        "-w",
        "--remove-watermark",
        action="store_true",
        help="Attempt to remove/reduce embedded watermarks with minimal visible changes.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        help="Force output image format (e.g., png, jpeg, tiff, ...). "
        "If not specified, use the original file extension.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to image(s) or directory(ies) to process.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool):
    """Sets up the logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_images(paths: List[str], recursive: bool) -> List[str]:
    """
    Collects image files from the given paths.
    If a path is a directory and recursive is True,
    it will traverse the directory structure.
    """
    supported_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp")
    image_files = []

    for path in paths:
        if os.path.isfile(path):
            if path.lower().endswith(supported_extensions):
                image_files.append(path)
            else:
                logging.warning(
                    f"File '{path}' is not a supported image format, skipping."
                )
        elif os.path.isdir(path):
            if recursive:
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(supported_extensions):
                            image_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(path):
                    if file.lower().endswith(supported_extensions):
                        image_files.append(os.path.join(path, file))
        else:
            logging.warning(
                f"Path '{path}' is neither a file nor a directory, skipping."
            )

    if not image_files:
        logging.error("No image files found in the specified path(s).")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to process.")
    return image_files


def remove_metadata(img: Image.Image, img_path: str, verbose: bool) -> Image.Image:
    """
    Removes as much metadata as possible from the given Pillow image.
    Returns a new Pillow Image object that contains only the pixel data.
    """
    if verbose:
        # Log the 'info' dictionary (often includes IPTC, XMP, etc.).
        for k, v in img.info.items():
            logging.debug(f"Metadata in 'img.info' -> Key: {k}, Value: {v}")

        # If there's EXIF data, parse with piexif to list the tags.
        exif_data = img.info.get("exif")
        if exif_data:
            try:
                exif_dict = piexif.load(exif_data)
                for ifd_name in exif_dict:
                    if isinstance(exif_dict[ifd_name], dict):
                        for tag_id, value in exif_dict[ifd_name].items():
                            tag_name = (
                                piexif.TAGS[ifd_name]
                                .get(tag_id, {})
                                .get("name", tag_id)
                            )
                            logging.debug(
                                f"EXIF Tag [{ifd_name}]: {tag_name} -> {value}"
                            )
            except Exception as e:
                logging.debug(f"Could not parse EXIF data for '{img_path}': {e}")

    # Create a new image from pixel data only
    data = img.getdata()
    mode = img.mode
    size = img.size

    new_img = Image.new(mode, size)
    new_img.putdata(data)
    return new_img


def remove_embedded_watermark(img: Image.Image, verbose: bool) -> Image.Image:
    """
    Tries to remove or reduce embedded watermarks with minimal visible changes
    to the human eye. This naive approach applies a mild median filter (size=3)
    to the entire image to blur out subtle text or lines.
    """
    if verbose:
        logging.debug(
            "Applying a minimal median filter (size=3) to reduce potential watermarks."
        )
    filtered_img = img.filter(ImageFilter.MedianFilter(size=3))
    return filtered_img


def determine_output_path(
    img_path: str, output_dir: str, force_format: str, verbose: bool
) -> str:
    """
    Determine the output file path based on whether a format is forced
    or an output directory is specified. By default, we overwrite the original
    file with the same extension. If a format is forced, we use the mapped
    extension (if known) or fallback to '.' + forced_format.
    """
    # Get the directory, base name, extension of the input
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    original_ext = os.path.splitext(os.path.basename(img_path))[1]  # e.g. ".jpg"
    original_dir = os.path.dirname(img_path)

    # Decide on extension
    if force_format:
        # Use the extension map, or fallback if unknown
        ext = FORMAT_EXTENSION_MAP.get(force_format.lower(), "." + force_format.lower())
    else:
        # Keep exactly the same extension
        ext = original_ext

    output_name = base_name + ext

    if output_dir:
        # Keep directory structure relative to input
        abs_input_dir = os.path.abspath(original_dir)
        abs_base_dir = os.path.commonpath([os.getcwd(), abs_input_dir])
        rel_path = os.path.relpath(abs_input_dir, abs_base_dir)
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, output_name)
    else:
        # Overwrite in-place (or same directory but potentially different extension if forced)
        output_path = os.path.join(original_dir, output_name)

    if verbose:
        action = "Overwriting" if output_dir is None else "Saving to"
        logging.debug(f"{action} path determined as '{output_path}'")

    return output_path


def process_image(
    img_path: str,
    output_dir: str,
    force_format: str,
    remove_watermark_flag: bool,
    verbose: bool,
):
    """
    Process a single image: remove metadata (and optionally watermarks),
    then save to either output_dir or overwrite the original path.
    """
    try:
        with Image.open(img_path) as img:
            original_format = img.format
            if verbose:
                logging.info(f"Processing '{img_path}' (format: {original_format})")

            # Remove all metadata
            clean_img = remove_metadata(img, img_path, verbose)

            # Optionally try to reduce watermarks
            if remove_watermark_flag:
                clean_img = remove_embedded_watermark(clean_img, verbose)

            # Determine the save format
            if force_format:
                save_format = force_format.upper()
            else:
                # Use original format (as recognized by Pillow) if available, else default 'PNG'
                save_format = (original_format or "PNG").upper()

            # Determine the output path (might overwrite the original)
            output_path = determine_output_path(
                img_path=img_path,
                output_dir=output_dir,
                force_format=force_format,
                verbose=verbose,
            )

            # Provide an empty EXIF block to ensure old EXIF isn't preserved
            exif_bytes = piexif.dump({})

            # If saving as JPEG, we may need to ensure 'RGB' mode to avoid errors with 'RGBA' or 'P'
            if save_format in ["JPEG", "JPG"] and clean_img.mode not in ["RGB", "L"]:
                clean_img = clean_img.convert("RGB")

            # Save the final image
            clean_img.save(output_path, format=save_format, exif=exif_bytes)
            if verbose:
                logging.info(f"Saved cleaned image to '{output_path}'")

    except Exception as e:
        logging.error(f"Error processing '{img_path}': {e}")


def main():
    """Main function that orchestrates removing metadata from images."""
    args = parse_arguments()
    setup_logging(args.verbose)

    # Collect all image files from arguments
    image_files = collect_images(args.paths, args.recursive)

    # Process each image
    for img_path in image_files:
        process_image(
            img_path=img_path,
            output_dir=args.output_dir,
            force_format=args.format,
            remove_watermark_flag=args.remove_watermark,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
