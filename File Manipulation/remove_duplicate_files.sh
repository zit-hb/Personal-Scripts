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
# ./remove_duplicate_files.sh [options] [directory|file]...
#
# Options:
#   -n, --dry-run   Show what would be done without making any changes.
#   -v, --verbose   Enable verbose output.
#   -h, --help      Display this help message.
#
# - [directory|file]: The file or directory to scan for duplicates.
#
# -------------------------------------------------------

set -euo pipefail

# Initialize variables
declare -A file_hashes
declare -a files_to_process=()
dry_run=false
verbose=false

# Function to display usage
usage() {
    grep '^#' "$0" | cut -c 4-
    exit 1
}

# Parse options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            dry_run=true
            shift
            ;;
        -v|--verbose)
            verbose=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            files_to_process+=("$1")
            shift
            ;;
    esac
done

# Check if at least one file or directory is provided
if [[ ${#files_to_process[@]} -eq 0 ]]; then
    echo "Error: No files or directories provided."
    usage
fi

# Function to check dependencies
check_dependencies() {
    local deps=("sha256sum" "awk" "find")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            echo "Error: Required command '$dep' not found."
            exit 1
        fi
    done
}

# Function to process input arguments and extract files
process_inputs() {
    local inputs=("$@")
    local file_list=()

    for input in "${inputs[@]}"; do
        if [[ -d "$input" ]]; then
            while IFS= read -r -d '' file; do
                file_list+=("$file")
            done < <(find "$input" -type f -print0)
        elif [[ -f "$input" ]]; then
            file_list+=("$input")
        else
            echo "Warning: '$input' is not a valid file or directory, skipping."
        fi
    done

    # Output files via printf with null terminators
    for file in "${file_list[@]}"; do
        printf '%s\0' "$file"
    done
}

# Main function
main() {
    check_dependencies

    # Get list of files to process
    mapfile -d '' files < <(process_inputs "${files_to_process[@]}")

    # Remove duplicate files
    for file in "${files[@]}"; do
        if [[ -r "$file" ]]; then
            hash=$(sha256sum "$file" | awk '{print $1}')
            if [[ -n "${file_hashes[$hash]:-}" ]]; then
                if $verbose || $dry_run; then
                    echo "Duplicate found: '$file' (duplicate of '${file_hashes[$hash]}')"
                fi
                if ! $dry_run; then
                    rm "$file"
                fi
            else
                file_hashes["$hash"]="$file"
            fi
        else
            echo "Warning: Cannot read file '$file', skipping."
        fi
    done
}

main
