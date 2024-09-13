#!/bin/bash

# -------------------------------------------------------
# Script: convert_webp_to_jpg.sh
#
# Description:
# This script searches for WEBP image files and converts them to JPG format.
#
# Usage:
# ./convert_webp_to_jpg.sh [options] [directory|image]...
#
# Options:
#   -o OUTPUT_DIR, --output-dir OUTPUT_DIR   The directory to save converted images.
#                                            Defaults to the current working directory if not provided.
#   -n, --dry-run                            Show what would be done without making any changes.
#   -v, --verbose                            Enable verbose output.
#   -h, --help                               Display this help message.
#
# - [directory|image]: The image or directory to scan for WEBP images.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

set -euo pipefail

# Default output directory
OUTPUT_DIR=$(pwd)
dry_run=false
verbose=false
declare -a files=()

# Function to display usage
usage() {
    grep '^#' "$0" | cut -c 4-
    exit 1
}

# Function to check dependencies
check_dependencies() {
    local deps=("convert" "find")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            echo "Error: Required command '$dep' not found."
            exit 1
        fi
    done
}

# Function to process input arguments and extract WEBP images
process_inputs() {
    local inputs=("$@")
    local images=()

    for input in "${inputs[@]}"; do
        if [[ -d "$input" ]]; then
            while IFS= read -r -d '' file; do
                images+=("$file")
            done < <(find "$input" -type f \( -iname '*.webp' \) -print0)
        elif [[ -f "$input" ]]; then
            case "$input" in
                *.webp|*.WEBP)
                    images+=("$input")
                    ;;
                *)
                    echo "Warning: '$input' is not a WEBP file, skipping."
                    ;;
            esac
        else
            echo "Warning: '$input' is not a valid file or directory, skipping."
        fi
    done

    # Output images via printf with null terminators
    for img in "${images[@]}"; do
        printf '%s\0' "$img"
    done
}

# Main function
main() {
    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -o|--output-dir)
                if [[ -n "${2:-}" ]]; then
                    OUTPUT_DIR="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
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
                files+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Error: No files or directories provided."
        usage
    fi

    check_dependencies

    # Process input arguments
    mapfile -d '' images < <(process_inputs "${files[@]}")

    if [[ ${#images[@]} -eq 0 ]]; then
        echo "No WEBP images found to convert."
        exit 0
    fi

    # Convert WEBP images to JPG
    for image in "${images[@]}"; do
        output="${OUTPUT_DIR}/$(basename "${image%.*}.jpg")"
        if $verbose || $dry_run; then
            echo "Converting '$image' to '$output'"
        fi
        if ! $dry_run; then
            # Ensure output directory exists
            mkdir -p "$OUTPUT_DIR"
            convert "$image" "$output"
        fi
    done
}

main "$@"
