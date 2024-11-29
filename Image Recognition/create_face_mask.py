#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_face_mask.py
#
# Description:
# This script detects all faces in an image, traces the contour
# of the detected faces, and generates a mask where the
# faces are black and the rest of the image is white.
#
# Usage:
# ./create_face_mask.py [image_file] [output_file] [-c CONFIDENCE] [-p SHAPE_PREDICTOR_PATH]
#
# - [image_file]: The path to the input image file.
# - [output_file]: The path to save the output mask image.
# - [-c CONFIDENCE, --confidence CONFIDENCE]: Minimum confidence for face detection (default: 0.5).
# - [-p SHAPE_PREDICTOR_PATH, --shape-predictor-path SHAPE_PREDICTOR_PATH]: Path to the shape predictor file.
#
# Template: ubuntu22.04
#
# Requirements:
# - CMake (install via: apt install cmake)
# - OpenCV (install via: apt install python3-opencv opencv-data)
# - Dlib (install via: pip install dlib)
# - Shape predictor (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import cv2
import dlib
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Create a face mask with exact face contour from an image using Dlib.')
parser.add_argument('image_file', type=str, help='The path to the input image file.')
parser.add_argument('output_file', type=str, help='The path to save the output mask image.')
parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum confidence for face detection (default: 0.5).')
parser.add_argument('-p', '--shape-predictor-path', type=str, required=True, help='Path to the shape predictor file.')

args = parser.parse_args()

# Load the shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor_path)

# Load the image
image = cv2.imread(args.image_file)
(h, w) = image.shape[:2]

# Create a white mask (all pixels initially white)
mask = np.ones((h, w), dtype="uint8") * 255

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces, scores, _ = detector.run(gray, 1, -1)

# Loop over the face detections
for i, face in enumerate(faces):
    if scores[i] >= args.confidence:
        # Get the landmarks/parts for the face
        shape = predictor(gray, face)
        points = []
        for j in range(0, 68):
            points.append((shape.part(j).x, shape.part(j).y))

        # Create a convex hull around the face
        hull = cv2.convexHull(np.array(points))

        # Draw the face region on the mask
        cv2.fillConvexPoly(mask, hull, 0)

# Save the resulting mask to the specified output file
cv2.imwrite(args.output_file, mask)

print(f"Mask saved to {args.output_file}")
