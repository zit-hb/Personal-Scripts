#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_metadata.py
#
# Description:
# This script detects and displays the metadata (EXIF, IPTC, XMP, etc.) from
# images. If there is no metadata, no output will be produced for that file.
#
# Usage:
#   ./detect_metadata.py [options] [directory|image ...]
#
# Arguments:
#   - [directory|image ...]: One or more paths to images or directories
#                            containing images.
#
# Options:
#   -r, --recursive    Process directories recursively.
#   -v, --verbose      Enable verbose logging (warnings, errors, debug
#                      information).
#   -h, --help         Display this help message.
#
# Template: ubuntu24.04
#
# Requirements:
#   - Pillow (install via: pip install Pillow==11.1.0)
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

from PIL import Image
import piexif


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect and display metadata (EXIF, IPTC, XMP, etc.) from images."
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
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
    if not verbose:
        level = logging.CRITICAL

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
                logging.debug(f"Skipping non-image file '{path}'.")
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
            logging.debug(f"Path '{path}' is neither a file nor a directory, skipping.")

    return image_files


def detect_metadata(img_path: str):
    """
    Detects and prints any available metadata for a single image.
    If no metadata is found, no output is produced for that image.
    """
    try:
        with Image.open(img_path) as img:
            # Attempt to gather both 'info' metadata and EXIF metadata
            metadata_found = False
            metadata_dict = {}

            # 'info' dictionary metadata (like IPTC/XMP stored by Pillow)
            # Pillow's img.info can contain various fields, e.g. 'icc_profile', 'exif', 'dpi'
            # that might hold metadata. We only display them if they're not empty or trivial.
            info_data = {}
            for key, val in img.info.items():
                # Not interesting
                if key.lower().startswith("jfif"):
                    continue
                # We skip 'exif' here because we handle that separately below
                if key.lower() == "exif":
                    continue

                info_data[key] = val

            if info_data:
                metadata_found = True
                metadata_dict["info"] = info_data

            # EXIF data, if present
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                try:
                    exif_dict = piexif.load(exif_bytes)
                    exif_readable = {}
                    for ifd_name in exif_dict:
                        # exif_dict[ifd_name] can be a dict of tag_id->value
                        if isinstance(exif_dict[ifd_name], dict):
                            tag_pairs = {}
                            for tag_id, value in exif_dict[ifd_name].items():
                                tag_name = (
                                    piexif.TAGS[ifd_name]
                                    .get(tag_id, {})
                                    .get("name", tag_id)
                                )
                                # Convert bytes to string for user-friendly output if possible
                                if isinstance(value, bytes):
                                    try:
                                        value_str = value.decode(
                                            "utf-8", errors="replace"
                                        )
                                        value = value_str
                                    except:
                                        pass
                                tag_pairs[tag_name] = value
                            if tag_pairs:
                                exif_readable[ifd_name] = tag_pairs

                    if exif_readable:
                        metadata_found = True
                        metadata_dict["exif"] = exif_readable
                except Exception as e:
                    logging.debug(f"Could not parse EXIF data for '{img_path}': {e}")

            if metadata_found:
                print(f"--- Metadata for '{img_path}' ---")
                for section_key, section_val in metadata_dict.items():
                    if isinstance(section_val, dict):
                        for k, v in section_val.items():
                            # If it's a dict inside, e.g. exif sub-IFDs
                            if isinstance(v, dict):
                                print(f"  {k}:")
                                for sub_tag, sub_val in v.items():
                                    print(f"    {sub_tag}: {sub_val}")
                            else:
                                print(f"  {k}: {v}")
                    else:
                        print(f"  {section_val}")
                print()

    except Exception as e:
        logging.debug(f"Error reading '{img_path}': {e}")


def main():
    """Main function that orchestrates detecting metadata from images."""
    args = parse_arguments()
    setup_logging(args.verbose)

    image_files = collect_images(args.paths, args.recursive)

    if not image_files:
        logging.debug("No image files found to analyze.")
        sys.exit(0)

    for img_path in image_files:
        detect_metadata(img_path)


if __name__ == "__main__":
    main()
