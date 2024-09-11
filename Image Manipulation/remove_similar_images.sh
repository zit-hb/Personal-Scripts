#!/bin/bash

# -------------------------------------------------------
# Script: remove_similar_images.sh
#
# Description:
# This script searches for images and removes those that are
# extremely visually similar, based on a configurable similarity
# threshold.
#
# Usage:
# ./remove_similar_images.sh [directory|image]... [threshold]
#
# - [directory|image]: The image or directory to scan for images.
# - [threshold]: The similarity threshold for comparison.
#                Lower values = stricter comparison.
#                Defaults to 5 if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Similarity threshold (default to 5 if not provided)
THRESHOLD=5

# Arrays to hold options and files
options=()
files=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        THRESHOLD="$1"
    else
        files+=("$1")
    fi
    shift
done

# Ensure ImageMagick is installed
if ! command -v compare &> /dev/null; then
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

# Remove similar images
for base_image in "${images[@]}"; do
    for other_image in "${images[@]}"; do
        if [ "$base_image" != "$other_image" ]; then
            difference=$(compare -metric RMSE "$base_image" "$other_image" null: 2>&1 | awk '{print $1}')

            if (( $(echo "$difference < $THRESHOLD" | bc -l) )); then
                echo "Removing similar image: $other_image (Difference: $difference)"
                rm "$other_image"
            fi
        fi
    done
done
