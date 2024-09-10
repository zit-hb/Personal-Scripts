#!/bin/bash

# -------------------------------------------------------
# Script: remove_duplicate_files.sh
#
# Description:
# This script recursively searches a given directory for 
# files and removes duplicate files based on their file 
# hash (SHA-256). Only one copy of each file is retained.
#
# Usage:
# ./remove_duplicate_files.sh [directory]
#
# - [directory]: The directory to scan for duplicates. 
#                If not provided, the current directory is used.
#
# Requirements:
# - This script relies on the 'sha256sum' command to compute 
#   file hashes and 'find' to recursively search directories.
#
# -------------------------------------------------------

# Directory to search for duplicates (default is current directory)
DIRECTORY=${1:-.}

# Declare an associative array to track file hashes
declare -A file_hashes

# Recursively find all files in the directory
find "$DIRECTORY" -type f | while read -r file; do
    # Compute the SHA-256 hash of the file
    hash=$(sha256sum "$file" | awk '{print $1}')
    
    # Check if the hash already exists
    if [[ -n "${file_hashes[$hash]}" ]]; then
        # If the hash exists, delete the duplicate file
        echo "Removing duplicate: $file"
        rm "$file"
    else
        # If not, store the hash and the file
        file_hashes[$hash]="$file"
    fi
done
