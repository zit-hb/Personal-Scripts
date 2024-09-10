#!/bin/bash

# -------------------------------------------------------
# Script: convert_webp_to_jpg.sh
# 
# Description:
# This script converts all .webp files in a specified 
# directory (or the current directory if none is provided)
# to .jpg format. It utilizes ImageMagick's 'convert' tool
# to handle the conversion process.
#
# Usage:
# ./convert_webp_to_jpg.sh [directory]
#
# - [directory]: The directory containing .webp files. 
#                If not provided, the current directory 
#                is used.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# Notes:
# - The script does not overwrite existing .jpg files; 
#   it will create new files with the same name but a 
#   different extension.
#
# -------------------------------------------------------

# Directory containing the .webp files
DIRECTORY=${1:-.}

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null
then
    echo "ImageMagick is required but it's not installed. Install it using: sudo apt install imagemagick"
    exit 1
fi

# Loop through all .webp files and convert them to .jpg
for file in "$DIRECTORY"/*.webp; do
    if [ -f "$file" ]; then
        # Get the filename without the extension
        filename="${file%.webp}"
        # Convert to jpg
        convert "$file" "${filename}.jpg"
        echo "Converted $file to ${filename}.jpg"
        rm "$file"
    fi
done
