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
# ./resize_large_images.sh [directory|image]... [max_width] [max_height]
#
# - [directory|image]: The image or directory to scan for images.
# - [max_width]: The maximum width of the images.
#                Defaults to 2048 pixels if not provided.
# - [max_height]: The maximum height of the images.
#                 Defaults to 2048 pixels if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Maximum allowed width (default is 2048 pixels)
MAX_WIDTH=${!#-1:-2048}

# Maximum allowed height (default is 2048 pixels)
MAX_HEIGHT=${!#:-2048}

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
images=($(process_inputs "${@:1:$#-2}"))

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
