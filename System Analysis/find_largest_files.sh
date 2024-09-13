#!/bin/bash

# -------------------------------------------------------
# Script: find_largest_files.sh
#
# Description:
# This script searches for the X largest files in a specified
# directory, recursively, and prints their details in a human-readable
# format by default. Optionally, it can output just the file paths.
#
# Usage:
# ./find_largest_files.sh [directory] [-n NUM_FILES] [--paths-only]
#
# - [directory]: The directory to search for files. Defaults to the current directory if not provided.
# - [-n NUM_FILES, --num-files NUM_FILES]: The number of largest files to display. Defaults to 10.
# - [--paths-only]: Output only the file paths, for easy use in other scripts.
#
# -------------------------------------------------------

# Default number of files to display and directory
NUM_FILES=10
DIRECTORY="."
PATHS_ONLY=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num-files) NUM_FILES="$2"; shift 2 ;;
        --paths-only) PATHS_ONLY=true; shift ;;
        *) DIRECTORY="$1"; shift ;;
    esac
done

# Resolve the directory to its absolute path
DIRECTORY=$(realpath "$DIRECTORY")

# Ensure the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Find the X largest files in the specified directory
if $PATHS_ONLY; then
    # Output only file paths
    find "$DIRECTORY" -type f -exec du -b {} + | sort -rh | head -n "$NUM_FILES" | cut -f2-
else
    # Output human-readable file information (size, path, creation and modification dates)
    echo "Finding the $NUM_FILES largest files in $DIRECTORY"
    echo "---------------------------------------"
    find "$DIRECTORY" -type f -exec du -b {} + | sort -rh | head -n "$NUM_FILES" | while read -r size file; do
        # Retrieve file metadata: size, path, creation, and modification dates (down to minute)
        creation_date=$(stat --format='%w' "$file" | cut -d'.' -f1)
        mod_date=$(stat --format='%y' "$file" | cut -d'.' -f1)
        if [[ "$creation_date" == "-" ]]; then
            creation_date="Unavailable"
        fi
        echo "Size: $(du -h "$file" | cut -f1), File: $file"
        echo "Creation Date: $creation_date, Modification Date: $mod_date"
        echo "---------------------------------------"
    done
fi
