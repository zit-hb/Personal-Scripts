#!/bin/bash
# -------------------------------------------------------
# Script: remove_similar_images.sh
#
# Description:
# This script searches for images and removes those that are
# extremely visually similar, based on a configurable similarity
# threshold.
#
# Usage:
# ./remove_similar_images.sh [options] [directory|image]...
#
# Options:
#   -t THRESHOLD, --threshold THRESHOLD   The similarity threshold for comparison.
#                                         Lower values = stricter comparison.
#                                         Defaults to 5 if not provided.
#   -d, --directory-only                  Limit comparisons to images within the same directory.
#   -n, --dry-run                         Show what would be done without making any changes.
#   -v, --verbose                         Enable verbose output.
#   -h, --help                            Display this help message.
#
# - [directory|image]: The image or directory to scan for images.
#
# Requirements:
# - ImageMagick (install via: sudo apt install imagemagick)
# - bc (for floating point comparisons)
#
# -------------------------------------------------------

set -euo pipefail

# Default similarity threshold
THRESHOLD=5
DIRECTORY_ONLY=false
dry_run=false
verbose=false

declare -a files=()
declare -A hash_to_images=()

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
    local deps=("compare" "convert" "find" "bc")
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

# Function to compute image hash using average hashing
compute_image_hash() {
    local image="$1"
    local hash

    # Convert image to 8x8 grayscale and get pixel values
    local pixels
    pixels=$(convert "$image" -resize 8x8! -colorspace Gray -depth 8 txt:- | \
             grep -Eo '#[0-9A-F]{2}' | \
             sed 's/#//')

    # Compute average pixel value
    local sum=0
    local count=0
    for pixel in $pixels; do
        sum=$((sum + 0x$pixel))
        count=$((count + 1))
    done
    local avg=$((sum / count))

    # Build hash
    hash=""
    for pixel in $pixels; do
        if (( 0x$pixel > avg )); then
            hash="${hash}1"
        else
            hash="${hash}0"
        fi
    done

    echo "$hash"
}

# Main function
main() {
    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -t|--threshold)
                if [[ -n "${2:-}" ]]; then
                    THRESHOLD="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -d|--directory-only)
                DIRECTORY_ONLY=true
                shift
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
        echo "No images found to process."
        exit 0
    fi

    # Remove similar images
    for image in "${images[@]}"; do
        if [[ ! -r "$image" ]]; then
            echo "Warning: Cannot read file '$image', skipping."
            continue
        fi

        # Compute image hash
        hash=$(compute_image_hash "$image")

        # Get directory if DIRECTORY_ONLY is true
        local dir=""
        if $DIRECTORY_ONLY; then
            dir=$(dirname "$image")
        fi

        key="$hash|$dir"

        if [[ -n "${hash_to_images[$key]:-}" ]]; then
            # Found a similar image
            existing_image="${hash_to_images[$key]}"

            # Compare images to get actual difference
            difference=$(compare -metric RMSE "$image" "$existing_image" null: 2>&1 | awk '{print $1}')
            if (( $(echo "$difference < $THRESHOLD" | bc -l) )); then
                if $verbose || $dry_run; then
                    echo "Removing similar image: '$image' (Difference: $difference)"
                fi
                if ! $dry_run; then
                    rm "$image"
                fi
                continue
            fi
        else
            hash_to_images["$key"]="$image"
        fi
    done
}

main "$@"
