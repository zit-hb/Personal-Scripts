#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_blurriness.py
#
# Description:
# This script checks if an image is blurry by calculating
# the variance of the Laplacian (a measure of sharpness).
# Images with a low variance are considered blurry.
#
# Usage:
# ./detect_blurriness.py [image_file] [-t THRESHOLD]
#
# - [image_file]: The path to the image file to check for blurriness.
# - [-t THRESHOLD, --threshold THRESHOLD]: The threshold for blurriness.
#                                          Default is 100 (lower values = blurrier).
#
# Returns exit code 0 if the image is not blurry, 1 if it is blurry.
#
# Requirements:
# - Python with OpenCV (install via: sudo apt install python3-opencv)
#
# -------------------------------------------------------

import cv2
import argparse
import sys

# Set up argument parser
parser = argparse.ArgumentParser(description='Detect blurry images using Laplacian variance.')
parser.add_argument('image_file', type=str, help='The path to the image file to check for blurriness.')
parser.add_argument('-t', '--threshold', type=float, default=100.0, help='Threshold for blurriness detection. Default is 100.')

args = parser.parse_args()

# Load the image
image = cv2.imread(args.image_file)
if image is None:
    print(f"Error: Could not load image {args.image_file}.")
    sys.exit(2)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the Laplacian variance (a measure of sharpness)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# Print the variance result
print(f"Image sharpness (Laplacian variance): {laplacian_var}")

# Determine if the image is blurry
if laplacian_var < args.threshold:
    print(f"Image {args.image_file} is blurry (Threshold: {args.threshold})")
    sys.exit(1)  # Exit code 1 for blurry image
else:
    print(f"Image {args.image_file} is sharp enough (Threshold: {args.threshold})")
    sys.exit(0)  # Exit code 0 for sharp image
