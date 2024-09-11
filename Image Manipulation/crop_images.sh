#!/bin/bash

# -------------------------------------------------------
# Script: crop_images.sh
#
# Description:
# This script searches for image files (JPEG, PNG, WEBP) and crops
# a specified number of pixels from the top, bottom, left, or right
# of each image.
#
# Usage:
# ./crop_images.sh [directory|image]... [-t TOP] [-b BOTTOM] [-l LEFT] [-r RIGHT]
#
# - [directory|image]: The image or directory to scan for images.
# - [-t TOP, --top TOP]: Number of pixels to crop from the top.
# - [-b BOTTOM, --bottom BOTTOM]: Number of pixels to crop from the bottom.
# - [-l LEFT, --left LEFT]: Number of pixels to crop from the left.
# - [-r RIGHT, --right RIGHT]: Number of pixels to crop from the right.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Default crop values (0 pixels from all sides if not specified)
CROP_TOP=0
CROP_BOTTOM=0
CROP_LEFT=0
CROP_RIGHT=0

# Arrays to hold options and files
options=()
files=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--top) CROP_TOP="$2"; shift 2 ;;
        -b|--bottom) CROP_BOTTOM="$2"; shift 2 ;;
        -l|--left) CROP_LEFT="$2"; shift 2 ;;
        -r|--right) CROP_RIGHT="$2"; shift 2 ;;
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

# Crop images
for image in "${images[@]}"; do
    width=$(identify -format "%w" "$image")
    height=$(identify -format "%h" "$image")

    new_width=$((width - CROP_LEFT - CROP_RIGHT))
    new_height=$((height - CROP_TOP - CROP_BOTTOM))
    offset_x=$CROP_LEFT
    offset_y=$CROP_TOP

    if (( new_width <= 0 || new_height <= 0 )); then
        echo "Error: Crop values too large for $image, skipping."
        continue
    fi

    echo "Cropping $image (Original size: ${width}x${height}, New size: ${new_width}x${new_height})"
    convert "$image" -crop "${new_width}x${new_height}+$offset_x+$offset_y" +repage "$image"
done
