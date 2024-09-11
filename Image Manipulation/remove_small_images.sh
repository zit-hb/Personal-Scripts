#!/bin/bash

# -------------------------------------------------------
# Script: remove_small_images.sh
#
# Description:
# This script searches for images and removes those that have
# a width or height smaller than a specified minimum dimension.
#
# Usage:
# ./remove_small_images.sh [directory|image]... [min_dimension]
#
# - [directory|image]: The image or directory to scan for images.
# - [min_dimension]: The minimum allowed width or height in pixels.
#                    Images smaller than this will be deleted.
#                    Defaults to 600 pixels if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Minimum allowed width or height (default is 600 pixels)
MIN_DIMENSION=600

# Arrays to hold options and files
options=()
files=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        MIN_DIMENSION="$1"
    else
        files+=("$1")
    fi
    shift
done

# Ensure ImageMagick is installed
if ! command -v identify &> /dev/null; then
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

# Remove small images
for image in "${images[@]}"; do
    dimensions=$(identify -format "%w %h" "$image")
    width=$(echo "$dimensions" | awk '{print $1}')
    height=$(echo "$dimensions" | awk '{print $2}')

    if (( width < MIN_DIMENSION || height < MIN_DIMENSION )); then
        echo "Deleting $image (Size: ${width}x${height})"
        rm "$image"
    fi
done
