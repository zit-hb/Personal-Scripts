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
# ./find_largest_files.sh [options] [directory]
#
# Options:
#   -n NUM_FILES, --num-files NUM_FILES   The number of largest files to display. Defaults to 10.
#   --paths-only                          Output only the file paths, for easy use in other scripts.
#   -v, --verbose                         Enable verbose output.
#   -d, --dry-run                         Show what would be done without making any changes.
#   -h, --help                            Display this help message.
#
# - [directory]: The directory to search for files. Defaults to the current directory if not provided.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

set -euo pipefail

# Default values
NUM_FILES=10
DIRECTORY="."
PATHS_ONLY=false
VERBOSE=false
DRY_RUN=false

# Function to display usage
usage() {
    sed -n '
        1d
        /^#/! q
        s/^# \{0,1\}//
        p
    ' "$0"
    exit 1
}

# Function to check dependencies
check_dependencies() {
    local deps=("find" "du" "sort" "head" "stat" "realpath")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            echo "Error: Required command '$dep' not found."
            exit 1
        fi
    done
}

# Parse arguments
parse_arguments() {
    local directory_set=false

    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -n|--num-files)
                if [[ -n "${2:-}" ]]; then
                    NUM_FILES="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            --paths-only)
                PATHS_ONLY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
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
                if ! $directory_set; then
                    DIRECTORY="$1"
                    directory_set=true
                    shift
                else
                    echo "Error: Multiple directories provided."
                    usage
                fi
                ;;
        esac
    done
}

# Main function
main() {
    parse_arguments "$@"
    check_dependencies

    # Resolve the directory to its absolute path
    DIRECTORY=$(realpath "$DIRECTORY")

    # Ensure the directory exists
    if [[ ! -d "$DIRECTORY" ]]; then
        echo "Error: Directory '$DIRECTORY' does not exist."
        exit 1
    fi

    if $VERBOSE || $DRY_RUN; then
        echo "Searching in directory: '$DIRECTORY'"
        echo "Number of files to display: $NUM_FILES"
        $PATHS_ONLY && echo "Outputting paths only." || echo "Outputting detailed information."
        $DRY_RUN && echo "Dry-run mode enabled."
    fi

    # Dry-run mode: skip execution
    if $DRY_RUN; then
        echo "Dry-run: No files will be listed or modified."
        exit 0
    fi

    # Find the X largest files in the specified directory
    if $PATHS_ONLY; then
        # Output only file paths
        find "$DIRECTORY" -type f -print0 | xargs -0 du -b | sort -rh | head -n "$NUM_FILES" | cut -f2-
    else
        # Output human-readable file information (size, path, creation and modification dates)
        echo "Finding the $NUM_FILES largest files in '$DIRECTORY'"
        echo "---------------------------------------"
        find "$DIRECTORY" -type f -print0 | xargs -0 du -b | sort -rh | head -n "$NUM_FILES" | while IFS= read -r line; do
            size=$(echo "$line" | cut -f1)
            file=$(echo "$line" | cut -f2-)
            if [[ ! -e "$file" ]]; then
                echo "Warning: File '$file' does not exist, skipping."
                continue
            fi
            # Retrieve file metadata: size, path, creation, and modification dates
            creation_date=$(stat -c '%w' "$file" 2>/dev/null | cut -d'.' -f1)
            mod_date=$(stat -c '%y' "$file" 2>/dev/null | cut -d'.' -f1)
            if [[ "$creation_date" == "-" || -z "$creation_date" ]]; then
                creation_date="Unavailable"
            fi
            human_size=$(du -h "$file" | cut -f1)
            echo "Size: $human_size, File: $file"
            echo "Creation Date: $creation_date, Modification Date: $mod_date"
            echo "---------------------------------------"
        done
    fi
}

main "$@"
