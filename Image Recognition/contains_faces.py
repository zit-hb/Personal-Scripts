#!/usr/bin/env python3

# -------------------------------------------------------
# Script: contains_faces.py
#
# Description:
# This script checks if a given image file contains one or
# more faces using OpenCV's `detectMultiScale` method. It prints
# the number of faces detected and returns a special exit code
# based on the result: 0 if the specified conditions for the number
# of faces are met, 1 otherwise.
#
# Usage:
# ./contains_faces.py [image_file] [-s SCALE] [-n NEIGHBORS] [-m WIDTH,HEIGHT] [-e NUM] [-l NUM] [-g NUM] [-c CASCADE_PATH]
#
# - [image_file]: The path to the image file to check for faces.
# - [-s SCALE, --scale-factor SCALE]: Parameter specifying how much the image size is reduced at each image scale (default: 1.1).
# - [-n NEIGHBORS, --min-neighbors NEIGHBORS]: Parameter specifying how many neighbors each candidate rectangle should have to retain it (default: 5).
# - [-m WIDTH,HEIGHT, --min-size WIDTH,HEIGHT]: Minimum possible object size. Objects smaller than that are ignored (default: 30,30).
# - [-e NUM, --exact NUM]: Exit successfully if exactly NUM faces are found.
# - [-l NUM, --less-than NUM]: Exit successfully if less than NUM faces are found.
# - [-g NUM, --more-than NUM]: Exit successfully if more than NUM faces are found.
# - [-c CASCADE_PATH, --cascade-path CASCADE_PATH]: Path to the Haar cascade file (default: /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml).
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
parser.add_argument('-e', '--exact', type=int, help='Exit successfully if exactly NUM faces are found.')
parser.add_argument('-l', '--less-than', type=int, help='Exit successfully if less than NUM faces are found.')
parser.add_argument('-g', '--more-than', type=int, help='Exit successfully if more than NUM faces are found.')
parser.add_argument('-c', '--cascade-path', type=str, default='/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml', help='Path to the Haar cascade file.')

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

# Check if the Haar cascade file exists
if not os.path.exists(args.cascade_path):
    print(f"Error: Haar cascade file not found at {args.cascade_path}")
    sys.exit(3)

face_cascade = cv2.CascadeClassifier(args.cascade_path)

if face_cascade.empty():
    print("Error: Could not load Haar cascade.")
    sys.exit(3)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors, minSize=min_size)

# Print the number of faces found
num_faces = len(faces)
print(f"Number of faces found: {num_faces}")

# Determine exit code based on the number of faces found
conditions_met = True

if args.exact is not None and num_faces != args.exact:
    conditions_met = False
if args.less_than is not None and num_faces >= args.less_than:
    conditions_met = False
if args.more_than is not None and num_faces <= args.more_than:
    conditions_met = False

if conditions_met:
    sys.exit(0)
else:
    sys.exit(1)
