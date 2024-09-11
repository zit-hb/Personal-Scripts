#!/bin/bash

# -------------------------------------------------------
# Script: crop_images.sh
#
# Description:
# This script searches a given directory (recursively) for
# image files (JPEG, PNG, WEBP) and crops a specified number
# of pixels from the top, bottom, left, or right of each image.
# If no crop is specified for a particular side, that side will
# remain unchanged.
#
# Usage:
# ./crop_images.sh [directory] [-t TOP] [-b BOTTOM] [-l LEFT] [-r RIGHT]
#
# - [directory]: The directory to scan for images.
#                Defaults to the current directory if not provided.
# - [-t TOP, --top TOP]: Number of pixels to crop from the top.
# - [-b BOTTOM, --bottom BOTTOM]: Number of pixels to crop from the bottom.
# - [-l LEFT, --left LEFT]: Number of pixels to crop from the left.
# - [-r RIGHT, --right RIGHT]: Number of pixels to crop from the right.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Directory to search for images (default is current directory)
DIRECTORY=${1:-.}

# Default crop values (0 pixels from all sides if not specified)
CROP_TOP=0
CROP_BOTTOM=0
CROP_LEFT=0
CROP_RIGHT=0

# Parse optional crop values
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--top) CROP_TOP="$2"; shift ;;
        -b|--bottom) CROP_BOTTOM="$2"; shift ;;
        -l|--left) CROP_LEFT="$2"; shift ;;
        -r|--right) CROP_RIGHT="$2"; shift ;;
        *) DIRECTORY="$1" ;;
    esac
    shift
done

# Ensure ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is required but not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Recursively find all image files in the directory
find "$DIRECTORY" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | while read -r image; do
    # Get the dimensions of the image
    width=$(identify -format "%w" "$image")
    height=$(identify -format "%h" "$image")

    # Calculate new dimensions and offset for cropping
    new_width=$((width - CROP_LEFT - CROP_RIGHT))
    new_height=$((height - CROP_TOP - CROP_BOTTOM))
    offset_x=$CROP_LEFT
    offset_y=$CROP_TOP

    # Ensure new dimensions are valid (non-negative)
    if (( new_width <= 0 || new_height <= 0 )); then
        echo "Error: Crop values too large for $image, skipping."
        continue
    fi

    # Perform the crop operation
    echo "Cropping $image (Original size: ${width}x${height}, New size: ${new_width}x${new_height})"
    convert "$image" -crop "${new_width}x${new_height}+$offset_x+$offset_y" +repage "$image"
done
