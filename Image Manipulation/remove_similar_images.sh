#!/bin/bash

# -------------------------------------------------------
# Script: remove_similar_images.sh
# 
# Description:
# This script searches a given directory (recursively) for
# images and removes those that are extremely visually 
# similar, based on a configurable similarity threshold.
# It uses ImageMagick's `compare` tool to calculate 
# the visual similarity between images. If two images 
# are found to be similar below the specified threshold, 
# one of them is deleted.
#
# Usage:
# ./remove_similar_images.sh [directory] [threshold]
#
# - [directory]: The directory to scan for images. 
#                If not provided, the current directory is used.
# - [threshold]: The similarity threshold for comparison. 
#                Lower values = stricter comparison. 
#                Defaults to 5 if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Directory to search for images (default is current directory)
DIRECTORY=${1:-.}

# Similarity threshold (default to 5 if not provided)
THRESHOLD=${2:-5}

# Ensure ImageMagick is installed
if ! command -v compare &> /dev/null; then
    echo "ImageMagick is required but not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Find all image files in the directory
find "$DIRECTORY" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | while read -r base_image; do
    # Loop through other images in the directory
    find "$DIRECTORY" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | while read -r other_image; do
        # Skip comparing the image with itself
        if [ "$base_image" != "$other_image" ]; then
            # Compare the two images and calculate the difference
            difference=$(compare -metric RMSE "$base_image" "$other_image" null: 2>&1 | awk '{print $1}')
            
            # Remove the image if the difference is below the threshold
            if (( $(echo "$difference < $THRESHOLD" | bc -l) )); then
                echo "Removing similar image: $other_image (Difference: $difference)"
                rm "$other_image"
            fi
        fi
    done
done
