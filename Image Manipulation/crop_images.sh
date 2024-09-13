#!/bin/bash
# -------------------------------------------------------
# Script: crop_images.sh
#
# Description:
# This script searches for image files (JPEG, PNG, WEBP) and crops
# a specified number of pixels from the top, bottom, left, or right
# of each image.
#
# Usage:
# ./crop_images.sh [options] [directory|image]...
#
# Options:
#   -t TOP, --top TOP            Number of pixels to crop from the top.
#   -b BOTTOM, --bottom BOTTOM   Number of pixels to crop from the bottom.
#   -l LEFT, --left LEFT         Number of pixels to crop from the left.
#   -r RIGHT, --right RIGHT      Number of pixels to crop from the right.
#   -n, --dry-run                Show what would be done without making any changes.
#   -v, --verbose                Enable verbose output.
#   -h, --help                   Display this help message.
#
# - [directory|image]: The image or directory to scan for images.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

set -euo pipefail

# Default crop values (0 pixels from all sides if not specified)
CROP_TOP=0
CROP_BOTTOM=0
CROP_LEFT=0
CROP_RIGHT=0

dry_run=false
verbose=false
declare -a files=()

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
    local deps=("convert" "identify" "find")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            echo "Error: Required command '$dep' not found."
            exit 1
        fi
    done
}

# Function to process input arguments and extract images
process_inputs() {
    local inputs=("$@")
    local images=()

    for input in "${inputs[@]}"; do
        if [[ -d "$input" ]]; then
            while IFS= read -r -d '' file; do
                images+=("$file")
            done < <(find "$input" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) -print0)
        elif [[ -f "$input" ]]; then
            case "$input" in
                *.jpg|*.jpeg|*.png|*.webp|*.JPG|*.JPEG|*.PNG|*.WEBP)
                    images+=("$input")
                    ;;
                *)
                    echo "Warning: '$input' is not a supported image file, skipping."
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
            -t|--top)
                if [[ -n "${2:-}" ]]; then
                    CROP_TOP="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -b|--bottom)
                if [[ -n "${2:-}" ]]; then
                    CROP_BOTTOM="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -l|--left)
                if [[ -n "${2:-}" ]]; then
                    CROP_LEFT="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -r|--right)
                if [[ -n "${2:-}" ]]; then
                    CROP_RIGHT="$2"
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
        echo "No images found to crop."
        exit 0
    fi

    # Crop images
    for image in "${images[@]}"; do
        if [[ ! -r "$image" ]]; then
            echo "Warning: Cannot read file '$image', skipping."
            continue
        fi

        width=$(identify -format "%w" "$image")
        height=$(identify -format "%h" "$image")

        new_width=$((width - CROP_LEFT - CROP_RIGHT))
        new_height=$((height - CROP_TOP - CROP_BOTTOM))
        offset_x=$CROP_LEFT
        offset_y=$CROP_TOP

        if (( new_width <= 0 || new_height <= 0 )); then
            echo "Error: Crop values too large for '$image', skipping."
            continue
        fi

        if $verbose || $dry_run; then
            echo "Cropping '$image' (Original size: ${width}x${height}, New size: ${new_width}x${new_height})"
        fi

        if ! $dry_run; then
            convert "$image" -crop "${new_width}x${new_height}+${offset_x}+${offset_y}" +repage "$image"
        fi
    done
}

main "$@"
