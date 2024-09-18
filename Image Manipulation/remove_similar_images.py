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
# Options:
#   -t THRESHOLD, --threshold THRESHOLD   The similarity threshold for comparison.
#                                         Lower values = stricter comparison.
#                                         Defaults to 5 if not provided.
#   -d, --directory-only                  Limit comparisons to images within the same directory.
#   -n, --dry-run                         Show what would be done without making any changes.
#   -v, --verbose                         Enable verbose output.
#   -h, --help                            Display this help message.
#
# Arguments:
#   [directory|image]: The image or directory to scan for images.
#
# Requirements:
# - Python 3.x
# - Pillow (install via: pip install Pillow)
# - imagehash (install via: pip install imagehash)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import os
import sys
from typing import List, Dict
from PIL import Image
import imagehash


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='This script searches for images and removes those that are '
                    'extremely visually similar, keeping only one copy.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Examples:
  ./remove_similar_images.py -n -v /path/to/images
  ./remove_similar_images.py -t 10 -d /path/to/images
''')
    parser.add_argument('paths', nargs='+', help='The image or directory to scan for images.')
    parser.add_argument('-t', '--threshold', type=int, default=5,
                        help='The similarity threshold for comparison.\n'
                             'Lower values = stricter comparison.\n'
                             'Defaults to 5 if not provided.')
    parser.add_argument('-d', '--directory-only', action='store_true',
                        help='Limit comparisons to images within the same directory.')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='Show what would be done without making any changes.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output.')
    return parser.parse_args()


def check_dependencies():
    try:
        import PIL
        import imagehash
    except ImportError as e:
        print(f"Error: Required module '{e.name}' not found. Please install it using 'pip install {e.name}'")
        sys.exit(1)


def collect_images(paths: List[str]) -> List[str]:
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
    dir_to_images = {}
    for image in images:
        dir_path = os.path.dirname(image)
        dir_to_images.setdefault(dir_path, []).append(image)
    return dir_to_images


def compute_image_hash(image_path: str):
    try:
        with Image.open(image_path) as img:
            return imagehash.average_hash(img)
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None


def compare_and_remove_images(images: List[str], args):
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
            difference = img_hash - existing_hash
            if difference < args.threshold:
                duplicate_found = True
                if args.verbose or args.dry_run:
                    print(f"Removing similar image: '{image}' (Difference: {difference})")
                if not args.dry_run:
                    try:
                        os.remove(image)
                    except Exception as e:
                        print(f"Error deleting image '{image}': {e}")
                removed_images.add(image)
                break
        if not duplicate_found:
            hash_to_image[img_hash] = image


def main():
    args = parse_arguments()
    check_dependencies()
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
