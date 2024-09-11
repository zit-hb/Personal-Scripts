#!/bin/bash

# -------------------------------------------------------
# Script: convert_webp_to_jpg.sh
#
# Description:
# This script searches for WEBP image files and converts them to JPG format.
#
# Usage:
# ./convert_webp_to_jpg.sh [directory|image]... [-o OUTPUT_DIR]
#
# - [directory|image]: The image or directory to scan for WEBP images.
# - [-o OUTPUT_DIR, --output-dir OUTPUT_DIR]: The directory to save converted images.
#                Defaults to the current working directory if not provided.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

# Default output directory
OUTPUT_DIR=$(pwd)

# Arrays to hold options and files
options=()
files=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) files+=("$1"); shift ;;
    esac
done

# Ensure ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "ImageMagick is required but not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Function to process input arguments and extract WEBP images
process_inputs() {
    local inputs=("$@")
    local images=()

    for input in "${inputs[@]}"; do
        if [ -d "$input" ]; then
            while IFS= read -r -d $'\0' file; do
                images+=("$file")
            done < <(find "$input" -type f -iname '*.webp' -print0)
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

# Convert WEBP images to JPG
for image in "${images[@]}"; do
    output="${OUTPUT_DIR}/$(basename "${image%.webp}.jpg")"
    echo "Converting $image to $output"
    convert "$image" "$output"
done
