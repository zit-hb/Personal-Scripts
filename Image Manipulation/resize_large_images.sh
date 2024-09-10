#!/bin/bash

# -------------------------------------------------------
# Script: resize_large_images.sh
# 
# Description:
# This script searches a given directory (recursively) for
# image files (JPEG, PNG, WEBP) and ensures that no image
# exceeds a maximum width or height. If an image's width or 
# height exceeds the specified maximum dimension, the image 
# is resized proportionally to fit within that limit, while 
# maintaining its aspect ratio. If the image's dimensions are 
# already within the limit, it is skipped.
#
# Usage:
# ./resize_large_images.sh [directory] [max_dimension]
#
# - [directory]: The directory to scan for images.
#                Defaults to the current directory if not provided.
# - [max_dimension]: The maximum width/height of the images.
#                    Defaults to 2048 pixels if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Directory to search for images (default is current directory)
DIRECTORY=${1:-.}

# Maximum allowed width or height (default is 2048 pixels)
MAX_DIMENSION=${2:-2048}

# Ensure ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is required but not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Recursively find all image files in the directory
find "$DIRECTORY" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | while read -r image; do
    # Get the dimensions of the image
    dimensions=$(identify -format "%w %h" "$image")
    width=$(echo "$dimensions" | awk '{print $1}')
    height=$(echo "$dimensions" | awk '{print $2}')

    # Check if the image exceeds the maximum dimension
    if (( width > MAX_DIMENSION || height > MAX_DIMENSION )); then
        echo "Resizing $image (Original size: ${width}x${height})"
        # Resize the image to fit within the max dimension, maintaining aspect ratio
        convert "$image" -resize ${MAX_DIMENSION}x${MAX_DIMENSION}\> "$image"
    else
        echo "Skipping $image (Size: ${width}x${height})"
    fi
done
