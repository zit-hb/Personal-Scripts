#!/bin/bash
# -------------------------------------------------------
# Script: remove_small_images.sh
#
# Description:
# This script searches for images and removes those that have
# a width or height smaller than specified minimum dimensions.
#
# Usage:
# ./remove_small_images.sh [options] [directory|image]...
#
# Options:
#   -w MIN_WIDTH, --min-width MIN_WIDTH    The minimum allowed width in pixels.
#                                          Images with a width smaller than this will be deleted.
#                                          Defaults to 600 pixels if not provided.
#   -h MIN_HEIGHT, --min-height MIN_HEIGHT The minimum allowed height in pixels.
#                                          Images with a height smaller than this will be deleted.
#                                          Defaults to 600 pixels if not provided.
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

# Default minimum allowed width and height
MIN_WIDTH=600
MIN_HEIGHT=600

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
    local deps=("identify" "find" "awk")
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
            -w|--min-width)
                if [[ -n "${2:-}" ]]; then
                    MIN_WIDTH="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -h|--min-height)
                if [[ -n "${2:-}" ]]; then
                    MIN_HEIGHT="$2"
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
            --help)
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
        echo "No images found to process."
        exit 0
    fi

    # Remove small images
    for image in "${images[@]}"; do
        if [[ ! -r "$image" ]]; then
            echo "Warning: Cannot read file '$image', skipping."
            continue
        fi

        dimensions=$(identify -format "%w %h" "$image" 2>/dev/null || true)
        if [[ -z "$dimensions" ]]; then
            echo "Warning: Failed to get dimensions for '$image', skipping."
            continue
        fi

        width=$(echo "$dimensions" | awk '{print $1}')
        height=$(echo "$dimensions" | awk '{print $2}')

        if (( width < MIN_WIDTH || height < MIN_HEIGHT )); then
            if $verbose || $dry_run; then
                echo "Deleting '$image' (Size: ${width}x${height})"
            fi
            if ! $dry_run; then
                rm "$image"
            fi
        else
            if $verbose; then
                echo "Keeping '$image' (Size: ${width}x${height})"
            fi
        fi
    done
}

main "$@"
