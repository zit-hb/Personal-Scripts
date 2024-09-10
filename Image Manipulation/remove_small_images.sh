#!/bin/bash

# -------------------------------------------------------
# Script: remove_small_images.sh
# 
# Description:
# This script searches a given directory (recursively) for
# images and removes those that have a width or height
# smaller than a specified minimum dimension. The script
# uses ImageMagick's `identify` tool to check the image
# dimensions.
#
# Usage:
# ./delete_small_images.sh [directory] [min_dimension]
#
# - [directory]: The directory to scan for images. 
#                If not provided, the current directory is used.
# - [min_dimension]: The minimum allowed width or height in pixels.
#                    Images smaller than this will be deleted.
#                    Defaults to 600 pixels if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Directory to search for images (default is current directory)
DIRECTORY=${1:-.}

# Minimum allowed width or height (default is 600 pixels)
MIN_DIMENSION=${2:-600}

# Ensure ImageMagick is installed
if ! command -v identify &> /dev/null; then
    echo "ImageMagick is required but not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Recursively find all image files in the directory
find "$DIRECTORY" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | while read -r image; do
    # Get the dimensions of the image
    dimensions=$(identify -format "%w %h" "$image")
    width=$(echo "$dimensions" | awk '{print $1}')
    height=$(echo "$dimensions" | awk '{print $2}')

    # Check if the image is smaller than the minimum dimension
    if (( width < MIN_DIMENSION || height < MIN_DIMENSION )); then
        echo "Deleting $image (Size: ${width}x${height})"
        rm "$image"
    fi
done
