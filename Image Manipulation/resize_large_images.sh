#!/bin/bash
# -------------------------------------------------------
# Script: resize_large_images.sh
#
# Description:
# This script searches for image files (JPEG, PNG, WEBP) and ensures
# that no image exceeds a maximum width or height. If an image's width
# or height exceeds the specified maximum dimension, the image is resized
# proportionally to fit within that limit, while maintaining its aspect ratio.
#
# Usage:
# ./resize_large_images.sh [options] [directory|image]...
#
# Options:
#   -w MAX_WIDTH, --max-width MAX_WIDTH    The maximum width of the images.
#                                          Defaults to 2048 pixels if not provided.
#   -h MAX_HEIGHT, --max-height MAX_HEIGHT The maximum height of the images.
#                                          Defaults to 2048 pixels if not provided.
#   -n, --dry-run                          Show what would be done without making any changes.
#   -v, --verbose                          Enable verbose output.
#   --help                                 Display this help message.
#
# - [directory|image]: The image or directory to scan for images.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
#
# -------------------------------------------------------

set -euo pipefail

# Default maximum allowed width and height
MAX_WIDTH=2048
MAX_HEIGHT=2048

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
            -w|--max-width)
                if [[ -n "${2:-}" ]]; then
                    MAX_WIDTH="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -h|--max-height)
                if [[ -n "${2:-}" ]]; then
                    MAX_HEIGHT="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            --help)
                usage
                ;;
            -n|--dry-run)
                dry_run=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
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
        echo "No images found to resize."
        exit 0
    fi

    # Resize large images
    for image in "${images[@]}"; do
        if [[ ! -r "$image" ]]; then
            echo "Warning: Cannot read file '$image', skipping."
            continue
        fi

        dimensions=$(identify -format "%w %h" "$image")
        width=$(echo "$dimensions" | awk '{print $1}')
        height=$(echo "$dimensions" | awk '{print $2}')

        if (( width > MAX_WIDTH || height > MAX_HEIGHT )); then
            if $verbose || $dry_run; then
                echo "Resizing '$image' (Original size: ${width}x${height})"
            fi
            if ! $dry_run; then
                convert "$image" -resize "${MAX_WIDTH}x${MAX_HEIGHT}>" "$image"
            fi
        else
            if $verbose; then
                echo "Skipping '$image' (Size: ${width}x${height})"
            fi
        fi
    done
}

main "$@"
