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
# ./extract_faces_from_video.sh [video_file] [-o OUTPUT_DIR] [-s FRAME_RATE] [-p CONTAINS_FACES_PATH] [-b BLURRY_THRESHOLD] [-d DETECT_BLURRY_PATH]
#
# - [video_file]: The path to the input video file.
# - [-o OUTPUT_DIR, --output-dir OUTPUT_DIR]: The directory to save the extracted frames.
#                Defaults to a temporary directory if not provided.
# - [-s FRAME_RATE, --frame-rate FRAME_RATE]: Frame extraction rate (frames per second). Default is to extract every frame.
# - [-p CONTAINS_FACES_PATH, --contains-faces-path CONTAINS_FACES_PATH]: Path to the "contains_faces.py" script. Defaults to "../Image Recognition/contains_faces.py".
# - [-b BLURRY_THRESHOLD, --blurry-threshold BLURRY_THRESHOLD]: Threshold for detecting blurry images. Defaults to 100.
# - [-d DETECT_BLURRY_PATH, --detect-blurry-path DETECT_BLURRY_PATH]: Path to the "detect_blurriness.py" script. Defaults to "../Image Recognition/detect_blurriness.py".
#
# Requirements:
# - FFmpeg (install via: sudo apt install ffmpeg)
# - Python with OpenCV (install via: sudo apt install python3-opencv opencv-data)
#
# -------------------------------------------------------

# Default paths for the Python scripts
CONTAINS_FACES_PATH="../Image Recognition/contains_faces.py"
DETECT_BLURRY_PATH="../Image Recognition/detect_blurriness.py"

BLURRY_THRESHOLD=100  # Default blurriness threshold
FRAME_RATE=""
OUTPUT_DIR=""

# Parse arguments
VIDEO_FILE="$1"
shift

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -s|--frame-rate) FRAME_RATE="$2"; shift 2 ;;
        -p|--contains-faces-path) CONTAINS_FACES_PATH="$2"; shift 2 ;;
        -b|--blurry-threshold) BLURRY_THRESHOLD="$2"; shift 2 ;;
        -d|--detect-blurry-path) DETECT_BLURRY_PATH="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Check if required arguments are provided
if [ -z "$VIDEO_FILE" ]; then
    echo "Usage: ./extract_faces_from_video.sh [video_file] [-o OUTPUT_DIR] [-s FRAME_RATE] [-p CONTAINS_FACES_PATH] [-b BLURRY_THRESHOLD] [-d DETECT_BLURRY_PATH]"
    exit 1
fi

# Create a temporary directory if OUTPUT_DIR is not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(mktemp -d)
    echo "No output directory specified. Using temporary directory: $OUTPUT_DIR"
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract frames from the video using FFmpeg
echo "Extracting frames from video..."
if [ -z "$FRAME_RATE" ]; then
    # Extract every frame if no frame rate is provided
    ffmpeg -i "$VIDEO_FILE" -q:v 1 "$OUTPUT_DIR/frame_%04d.jpg"
else
    # Extract frames at the specified frame rate
    ffmpeg -i "$VIDEO_FILE" -q:v 1 -vf "fps=$FRAME_RATE" "$OUTPUT_DIR/frame_%04d.jpg"
fi

# Loop through extracted frames, filter blurry images, and check face count
for image in "$OUTPUT_DIR"/*.jpg; do
    if [ -f "$image" ]; then
        echo "Processing $image..."

        # Check if the image is blurry
        "$DETECT_BLURRY_PATH" "$image" -t "$BLURRY_THRESHOLD"
        if [ $? -ne 0 ]; then
            echo "Removing $image (blurry)"
            rm "$image"
            continue
        fi

        # Check for the number of faces
        "$CONTAINS_FACES_PATH" "$image" -g "0"

        # Remove the image if it doesn't contain faces (exit code 1)
        if [ $? -ne 0 ]; then
            echo "Removing $image (does not meet face conditions)"
            rm "$image"
        fi
    fi
done

echo "Processing complete."
