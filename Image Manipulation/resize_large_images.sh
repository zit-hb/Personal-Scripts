#!/bin/bash

# -------------------------------------------------------
# Script: resize_large_images.sh
#
# Description:
# This script searches for image files (JPEG, PNG, WEBP) and ensures
# that no image exceeds a maximum width or height. If an image's width
# or height exceeds the specified maximum dimension, the image is resized
# proportionally to fit within that limit, while maintaining its aspect ratio.
#
# Usage:
# ./resize_large_images.sh [directory|image]... [-w MAX_WIDTH] [-h MAX_HEIGHT]
#
# - [directory|image]: The image or directory to scan for images.
# - [-w MAX_WIDTH, --max-width MAX_WIDTH]: The maximum width of the images.
#                Defaults to 2048 pixels if not provided.
# - [-h MAX_HEIGHT, --max-height MAX_HEIGHT]: The maximum height of the images.
#                 Defaults to 2048 pixels if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Default maximum allowed width and height
MAX_WIDTH=2048
MAX_HEIGHT=2048

# Arrays to hold options and files
options=()
files=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -w|--max-width) MAX_WIDTH="$2"; shift 2 ;;
        -h|--max-height) MAX_HEIGHT="$2"; shift 2 ;;
        *) files+=("$1"); shift ;;
    esac
done

# Ensure ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is required but not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Function to process input arguments and extract images
process_inputs() {
    local inputs=("$@")
    local images=()

    for input in "${inputs[@]}"; do
        if [ -d "$input" ]; then
            while IFS= read -r -d $'\0' file; do
                images+=("$file")
            done < <(find "$input" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) -print0)
        elif [ -f "$input" ]; then
            images+=("$input")
        else
            echo "Warning: $input is not a valid file or directory, skipping."
        fi
    done

    echo "${images[@]}"
}

# Process input arguments
images=($(process_inputs "${files[@]}"))

# Resize large images
for image in "${images[@]}"; do
    dimensions=$(identify -format "%w %h" "$image")
    width=$(echo "$dimensions" | awk '{print $1}')
    height=$(echo "$dimensions" | awk '{print $2}')

    if (( width > MAX_WIDTH || height > MAX_HEIGHT )); then
        echo "Resizing $image (Original size: ${width}x${height})"
        convert "$image" -resize ${MAX_WIDTH}x${MAX_HEIGHT}\> "$image"
    else
        echo "Skipping $image (Size: ${width}x${height})"
    fi
done
