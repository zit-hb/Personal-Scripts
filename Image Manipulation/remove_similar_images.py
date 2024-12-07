#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_similar_images.py
#
# Description:
# This script searches for images and removes those that are
# extremely visually similar, keeping only one copy.
#
# Usage:
# ./remove_similar_images.py [options] [directory|image]...
#
# Arguments:
#   - [directory|image]: The image or directory to scan for images.
#
# Options:
#   -t THRESHOLD, --threshold THRESHOLD   The similarity threshold for comparison.
#                                         Lower values = stricter comparison.
#                                         Defaults to 5 if not provided.
#   -d, --directory-only                  Limit comparisons to images within the same directory.
#   -n, --dry-run                         Show what would be done without making any changes.
#   -v, --verbose                         Enable verbose output.
#   -k KEEP, --keep KEEP                  Criteria for keeping a file. (choices: "oldest", "newest", "biggest", "smallest") (default: "biggest")
#   -h, --help                            Display this help message.
#
# Template: ubuntu22.04
#
# Requirements:
# - Pillow (install via: pip install Pillow==11.0.0)
# - imagehash (install via: pip install imagehash==4.3.1)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import os
import sys
from typing import List, Dict, Optional
from PIL import Image
import imagehash


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='This script searches for images and removes those that are '
                    'extremely visually similar, keeping only one copy.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='The image or directory to scan for images.'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=5,
        help='The similarity threshold for comparison. Lower values = stricter comparison.'
    )
    parser.add_argument(
        '-d', '--directory-only',
        action='store_true',
        help='Limit comparisons to images within the same directory.'
    )
    parser.add_argument(
        '-n', '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '-k', '--keep',
        type=str,
        default='biggest',
        choices=['oldest', 'newest', 'biggest', 'smallest'],
        help='Criteria for keeping a file.'
    )
    return parser.parse_args()


def collect_images(paths: List[str]) -> List[str]:
    """Collects image files from the specified paths."""
    images = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(supported_extensions):
                        images.append(os.path.join(root, file))
        elif os.path.isfile(path):
            if path.endswith(supported_extensions):
                images.append(path)
            else:
                print(f"Warning: '{path}' is not a supported image file, skipping.")
        else:
            print(f"Warning: '{path}' is not a valid file or directory, skipping.")
    return images


def group_images_by_directory(images: List[str]) -> Dict[str, List[str]]:
    """Groups images by their directory path."""
    dir_to_images = {}
    for image in images:
        dir_path = os.path.dirname(image)
        dir_to_images.setdefault(dir_path, []).append(image)
    return dir_to_images


def compute_image_hash(image_path: str) -> Optional[imagehash.ImageHash]:
    """Computes the average hash of an image."""
    try:
        with Image.open(image_path) as img:
            return imagehash.average_hash(img)
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None


def select_image_to_keep(image1: str, image2: str, criteria: str) -> str:
    """Selects one image to keep from two images based on the specified criteria."""
    if criteria == 'oldest':
        time1 = os.path.getmtime(image1)
        time2 = os.path.getmtime(image2)
        return image1 if time1 < time2 else image2
    elif criteria == 'newest':
        time1 = os.path.getmtime(image1)
        time2 = os.path.getmtime(image2)
        return image1 if time1 > time2 else image2
    elif criteria == 'biggest':
        size1 = os.path.getsize(image1)
        size2 = os.path.getsize(image2)
        return image1 if size1 > size2 else image2
    elif criteria == 'smallest':
        size1 = os.path.getsize(image1)
        size2 = os.path.getsize(image2)
        return image1 if size1 < size2 else image2
    else:
        return image1


def compare_and_remove_images(images: List[str], args: argparse.Namespace) -> None:
    """Compares images and removes those that are extremely visually similar."""
    hash_to_image = {}
    removed_images = set()
    for image in images:
        if image in removed_images:
            continue
        img_hash = compute_image_hash(image)
        if img_hash is None:
            continue
        duplicate_found = False
        for existing_hash, existing_image in hash_to_image.items():
            if existing_image in removed_images:
                continue
            difference = img_hash - existing_hash
            if difference < args.threshold:
                duplicate_found = True
                # Decide which image to keep
                image_to_keep = select_image_to_keep(image, existing_image, args.keep)
                image_to_remove = existing_image if image_to_keep == image else image
                if args.verbose or args.dry_run:
                    print(f"Removing similar image: '{image_to_remove}' (Difference: {difference})")
                if not args.dry_run:
                    try:
                        os.remove(image_to_remove)
                    except Exception as e:
                        print(f"Error deleting image '{image_to_remove}': {e}")
                removed_images.add(image_to_remove)
                # Update hash_to_image
                if image_to_remove == existing_image:
                    hash_to_image[existing_hash] = image_to_keep
                break
        if not duplicate_found:
            hash_to_image[img_hash] = image


def main() -> None:
    """Main function that orchestrates the image removal process."""
    args = parse_arguments()
    images = collect_images(args.paths)
    if not images:
        print("No images found to process.")
        sys.exit(0)
    if args.verbose:
        print(f"Total images to process: {len(images)}")
    if args.directory_only:
        dir_to_images = group_images_by_directory(images)
        for dir_path, dir_images in dir_to_images.items():
            if args.verbose:
                print(f"Processing directory: '{dir_path}'")
            compare_and_remove_images(dir_images, args)
    else:
        compare_and_remove_images(images, args)


if __name__ == "__main__":
    main()
