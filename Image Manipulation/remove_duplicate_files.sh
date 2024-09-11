#!/bin/bash

# -------------------------------------------------------
# Script: remove_duplicate_files.sh
#
# Description:
# This script searches for files and removes duplicate files
# based on their file hash (SHA-256). Only one copy of each file
# is retained.
#
# Usage:
# ./remove_duplicate_files.sh [directory|file]...
#
# - [directory|file]: The file or directory to scan for duplicates.
#
# -------------------------------------------------------

# Declare an associative array to track file hashes
declare -A file_hashes

# Arrays to hold options and files
options=()
files=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    files+=("$1")
    shift
done

# Function to process input arguments and extract files
process_inputs() {
    local inputs=("$@")
    local files=()

    for input in "${inputs[@]}"; do
        if [ -d "$input" ]; then
            while IFS= read -r -d $'\0' file; do
                files+=("$file")
            done < <(find "$input" -type f -print0)
        elif [ -f "$input" ]; then
            files+=("$input")
        else
            echo "Warning: $input is not a valid file or directory, skipping."
        fi
    done

    echo "${files[@]}"
}

# Process input arguments
files=($(process_inputs "${files[@]}"))

# Remove duplicate files
for file in "${files[@]}"; do
    hash=$(sha256sum "$file" | awk '{print $1}')

    if [[ -n "${file_hashes[$hash]}" ]]; then
        echo "Removing duplicate: $file"
        rm "$file"
    else
        file_hashes[$hash]="$file"
    fi
done
