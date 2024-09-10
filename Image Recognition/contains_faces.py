#!/usr/bin/env python3

# -------------------------------------------------------
# Script: contains_faces.py
#
# Description:
# This script checks if a given image file contains one or
# more faces using OpenCV's `detectMultiScale` method. It prints
# the number of faces detected and returns a special exit code
# based on the result: 0 if faces are found, 1 if no faces are found.
#
# Usage:
# ./contains_faces.py [image_file] [-s SCALE] [-n NEIGHBORS] [-m WIDTH,HEIGHT]
#
# - [image_file]: The path to the image file to check for faces.
# - [-s SCALE, --scale-factor SCALE]: Parameter specifying how much the image size is reduced at each image scale (default: 1.1).
# - [-n NEIGHBORS, --min-neighbors NEIGHBORS]: Parameter specifying how many neighbors each candidate rectangle should have to retain it (default: 5).
# - [-m WIDTH,HEIGHT, --min-size WIDTH,HEIGHT]: Minimum possible object size. Objects smaller than that are ignored (default: 30,30).
#
# Requirements:
# - Python with OpenCV (install via: sudo apt install python3-opencv opencv-data)
#
# -------------------------------------------------------

import sys
import cv2
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Detect faces in an image using OpenCV.')
parser.add_argument('image_file', type=str, help='The path to the image file to check for faces.')
parser.add_argument('-s', '--scale-factor', type=float, default=1.1, help='Parameter specifying how much the image size is reduced at each image scale.')
parser.add_argument('-n', '--min-neighbors', type=int, default=5, help='Parameter specifying how many neighbors each candidate rectangle should have to retain it.')
parser.add_argument('-m', '--min-size', type=str, default='30,30', help='Minimum possible object size. Objects smaller than that are ignored. Format: width,height')

args = parser.parse_args()

# Parse min-size argument
min_size = tuple(map(int, args.min_size.split(',')))

# Load the image
image = cv2.imread(args.image_file)
if image is None:
    print("Error: Could not load image.")
    sys.exit(2)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar cascade for face detection
haar_cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

# Check if the Haar cascade file exists
if not os.path.exists(haar_cascade_path):
    print(f"Error: Haar cascade file not found at {haar_cascade_path}")
    sys.exit(3)

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

if face_cascade.empty():
    print("Error: Could not load Haar cascade.")
    sys.exit(3)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors, minSize=min_size)

# Print the number of faces found
num_faces = len(faces)
print(f"Number of faces found: {num_faces}")

# Exit with code 0 if faces are found, 1 if none are found
sys.exit(0 if num_faces > 0 else 1)
