#!/bin/bash
# -------------------------------------------------------
# Script: extract_faces_from_video.sh
#
# Description:
# This script extracts all frames (images) from a given video
# file and uses the "contains_faces.py" script and the
# "detect_blurriness.py" script to filter and keep only
# the frames that contain faces and are not blurry.
#
# Usage:
# ./extract_faces_from_video.sh [options] [video_file]
#
# Options:
#   -o OUTPUT_DIR, --output-dir OUTPUT_DIR
#       The directory to save the extracted frames.
#       Defaults to a temporary directory if not provided.
#   -s FRAME_RATE, --frame-rate FRAME_RATE
#       Frame extraction rate (frames per second). Default is to extract every frame.
#   -p CONTAINS_FACES_PATH, --contains-faces-path CONTAINS_FACES_PATH
#       Path to the "contains_faces.py" script. Defaults to "../Image Recognition/contains_faces.py".
#   -b BLURRY_THRESHOLD, --blurry-threshold BLURRY_THRESHOLD
#       Threshold for detecting blurry images. Defaults to 100.
#   -d DETECT_BLURRY_PATH, --detect-blurry-path DETECT_BLURRY_PATH
#       Path to the "detect_blurriness.py" script. Defaults to "../Image Recognition/detect_blurriness.py".
#   -n, --dry-run
#       Show what would be done without making any changes.
#   -v, --verbose
#       Enable verbose output.
#   -h, --help
#       Display this help message.
#
# Requirements:
# - FFmpeg (install via: sudo apt install ffmpeg)
# - Python with OpenCV (install via: sudo apt install python3-opencv opencv-data)
#
# -------------------------------------------------------

set -euo pipefail

# Default paths for the Python scripts
CONTAINS_FACES_PATH="../Image Recognition/contains_faces.py"
DETECT_BLURRY_PATH="../Image Recognition/detect_blurriness.py"

BLURRY_THRESHOLD=100  # Default blurriness threshold
FRAME_RATE=""
OUTPUT_DIR=""
DRY_RUN=false
VERBOSE=false

VIDEO_FILE=""

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
    local deps=("ffmpeg" "python3")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" >/dev/null 2>&1; then
            echo "Error: Required command '$dep' not found."
            exit 1
        fi
    done

    # Check if the Python scripts exist
    if [[ ! -x "$CONTAINS_FACES_PATH" ]]; then
        echo "Error: contains_faces.py script not found or not executable at '$CONTAINS_FACES_PATH'."
        exit 1
    fi

    if [[ ! -x "$DETECT_BLURRY_PATH" ]]; then
        echo "Error: detect_blurriness.py script not found or not executable at '$DETECT_BLURRY_PATH'."
        exit 1
    fi
}

# Function to parse arguments
parse_arguments() {
    if [[ $# -eq 0 ]]; then
        echo "Error: No video file provided."
        usage
    fi

    POSITIONAL=()
    while [[ $# -gt 0 ]]; do
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
            -s|--frame-rate)
                if [[ -n "${2:-}" ]]; then
                    FRAME_RATE="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -p|--contains-faces-path)
                if [[ -n "${2:-}" ]]; then
                    CONTAINS_FACES_PATH="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -b|--blurry-threshold)
                if [[ -n "${2:-}" ]]; then
                    BLURRY_THRESHOLD="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -d|--detect-blurry-path)
                if [[ -n "${2:-}" ]]; then
                    DETECT_BLURRY_PATH="$2"
                    shift 2
                else
                    echo "Error: Missing argument for $1"
                    usage
                fi
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
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
                POSITIONAL+=("$1")
                shift
                ;;
        esac
    done

    set -- "${POSITIONAL[@]}" # restore positional parameters

    if [[ ${#POSITIONAL[@]} -eq 0 ]]; then
        echo "Error: No video file provided."
        usage
    elif [[ ${#POSITIONAL[@]} -gt 1 ]]; then
        echo "Error: Multiple video files provided. Please provide only one video file."
        usage
    else
        VIDEO_FILE="${POSITIONAL[0]}"
    fi

    if [[ ! -f "$VIDEO_FILE" ]]; then
        echo "Error: Video file '$VIDEO_FILE' not found."
        exit 1
    fi
}

# Main function
main() {
    parse_arguments "$@"
    check_dependencies

    # Create a temporary directory if OUTPUT_DIR is not provided
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR=$(mktemp -d)
        if $VERBOSE || $DRY_RUN; then
            echo "No output directory specified. Using temporary directory: '$OUTPUT_DIR'"
        fi
    fi

    # Create the output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Extract frames from the video using FFmpeg
    if $VERBOSE || $DRY_RUN; then
        echo "Extracting frames from video..."
    fi

    if [[ -z "$FRAME_RATE" ]]; then
        # Extract every frame if no frame rate is provided
        FF_COMMAND=("ffmpeg" -i "$VIDEO_FILE" -q:v 1 "$OUTPUT_DIR/frame_%04d.jpg")
    else
        # Extract frames at the specified frame rate
        FF_COMMAND=("ffmpeg" -i "$VIDEO_FILE" -q:v 1 -vf "fps=$FRAME_RATE" "$OUTPUT_DIR/frame_%04d.jpg")
    fi

    if $VERBOSE || $DRY_RUN; then
        echo "Running command: ${FF_COMMAND[*]}"
    fi

    if ! $DRY_RUN; then
        "${FF_COMMAND[@]}"
    fi

    # Check if any frames were extracted
    shopt -s nullglob
    images=("$OUTPUT_DIR"/*.jpg)
    shopt -u nullglob

    if [[ ${#images[@]} -eq 0 ]]; then
        echo "No frames extracted from video."
        exit 1
    fi

    # Loop through extracted frames, filter blurry images, and check face count
    for image in "${images[@]}"; do
        if [[ -f "$image" ]]; then
            if $VERBOSE || $DRY_RUN; then
                echo "Processing '$image'..."
            fi

            # Check if the image is blurry
            DETECT_BLURRY_CMD=("$DETECT_BLURRY_PATH" "$image" -t "$BLURRY_THRESHOLD")
            if $VERBOSE || $DRY_RUN; then
                echo "Running command: ${DETECT_BLURRY_CMD[*]}"
            fi
            if ! $DRY_RUN; then
                if ! "${DETECT_BLURRY_CMD[@]}"; then
                    echo "Removing '$image' (blurry)"
                    rm "$image"
                    continue
                fi
            fi

            # Check for the number of faces
            CONTAINS_FACES_CMD=("$CONTAINS_FACES_PATH" "$image" -g "0")
            if $VERBOSE || $DRY_RUN; then
                echo "Running command: ${CONTAINS_FACES_CMD[*]}"
            fi
            if ! $DRY_RUN; then
                if ! "${CONTAINS_FACES_CMD[@]}"; then
                    echo "Removing '$image' (does not meet face conditions)"
                    rm "$image"
                fi
            fi
        fi
    done

    echo "Processing complete."
    if $VERBOSE || $DRY_RUN; then
        echo "Final images are in '$OUTPUT_DIR'"
    fi
}

main "$@"
