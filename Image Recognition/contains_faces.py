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
# ./contains_faces.py [options] [image_file]
#
# Arguments:
#   - [image_file]: The path to the image file to check for faces.
#
# Options:
#   -s SCALE, --scale-factor SCALE      Parameter specifying how much the image size is reduced at each image scale (default: 1.1).
#   -n NEIGHBORS, --min-neighbors NEIGHBORS
#                                       Parameter specifying how many neighbors each candidate rectangle should have to retain it (default: 5).
#   -m WIDTH,HEIGHT, --min-size WIDTH,HEIGHT
#                                       Minimum possible object size. Objects smaller than that are ignored (default: 30,30).
#   -e NUM, --exact NUM                 Exit successfully if exactly NUM faces are found.
#   -l NUM, --less-than NUM             Exit successfully if less than NUM faces are found.
#   -g NUM, --more-than NUM             Exit successfully if more than NUM faces are found.
#   -c CASCADE_PATH, --cascade-path CASCADE_PATH
#                                       Path to the Haar cascade file (default: /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml).
#   -o OUTPUT_PATH, --output-path OUTPUT_PATH
#                                       Path to save the output image with detected faces drawn.
#   -f, --face-coordinates              Output the coordinates of detected faces.
#   -F FORMAT, --output-format FORMAT   Specify the output format for face coordinates: text (default), json, csv.
#   -r WIDTH,HEIGHT, --resize WIDTH,HEIGHT
#                                       Resize the input image before processing.
#   -R ANGLE, --rotate ANGLE            Rotate the image before processing. Specify angle in degrees.
#   -d, --display                       Display the image with detected faces.
#   -v, --verbose                       Enable verbose logging.
#   -q, --quiet                         Suppress non-error output.
#   -h, --help                          Display this help message.
#
# Template: ubuntu22.04
#
# Requirements:
#   - OpenCV (install via: apt-get install -y python3-opencv opencv-data)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import sys
import cv2
import os
import argparse
import numpy as np
import logging
import json
import csv


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect faces in an image using OpenCV."
    )
    parser.add_argument(
        "image_file",
        type=str,
        help="The path to the image file to check for faces.",
    )
    parser.add_argument(
        "-s",
        "--scale-factor",
        type=float,
        default=1.1,
        help="Parameter specifying how much the image size is reduced at each image scale.",
    )
    parser.add_argument(
        "-n",
        "--min-neighbors",
        type=int,
        default=5,
        help="Parameter specifying how many neighbors each candidate rectangle should have to retain it.",
    )
    parser.add_argument(
        "-m",
        "--min-size",
        type=str,
        default="30,30",
        help="Minimum possible object size. Objects smaller than that are ignored. Format: width,height",
    )
    parser.add_argument(
        "-e",
        "--exact",
        type=int,
        help="Exit successfully if exactly NUM faces are found.",
    )
    parser.add_argument(
        "-l",
        "--less-than",
        type=int,
        help="Exit successfully if less than NUM faces are found.",
    )
    parser.add_argument(
        "-g",
        "--more-than",
        type=int,
        help="Exit successfully if more than NUM faces are found.",
    )
    parser.add_argument(
        "-c",
        "--cascade-path",
        type=str,
        default="/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        help="Path to the Haar cascade file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to save the output image with detected faces drawn.",
    )
    parser.add_argument(
        "-f",
        "--face-coordinates",
        action="store_true",
        help="Output the coordinates of detected faces.",
    )
    parser.add_argument(
        "-F",
        "--output-format",
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Specify the output format for face coordinates.",
    )
    parser.add_argument(
        "-r",
        "--resize",
        type=str,
        help="Resize the input image before processing. Format: width,height",
    )
    parser.add_argument(
        "-R",
        "--rotate",
        type=float,
        help="Rotate the image before processing. Specify angle in degrees.",
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display the image with detected faces.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool, quiet: bool):
    """
    Sets up the logging configuration.
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def output_face_coordinates(faces, output_format):
    """
    Outputs the coordinates of detected faces in the specified format.
    """
    if output_format == "text":
        for x, y, w, h in faces:
            print(f"Face at x:{x}, y:{y}, width:{w}, height:{h}")
    elif output_format == "json":
        faces_list = [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            for (x, y, w, h) in faces
        ]
        print(json.dumps(faces_list))
    elif output_format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["x", "y", "width", "height"])
        for x, y, w, h in faces:
            writer.writerow([x, y, w, h])


def main():
    """
    Main function of the script.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.quiet)

    # Load the image
    image = cv2.imread(args.image_file)
    if image is None:
        logging.error("Could not load image.")
        sys.exit(2)

    # If resize is specified, resize the image
    if args.resize:
        try:
            resize_dims = tuple(map(int, args.resize.split(",")))
            image = cv2.resize(image, resize_dims)
            logging.info(f"Image resized to {resize_dims}")
        except Exception as e:
            logging.error(f"Failed to resize image: {e}")
            sys.exit(4)

    # If rotate is specified, rotate the image
    if args.rotate:
        try:
            angle = args.rotate
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(
                image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            logging.info(f"Image rotated by {angle} degrees")
        except Exception as e:
            logging.error(f"Failed to rotate image: {e}")
            sys.exit(5)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if the Haar cascade file exists
    if not os.path.exists(args.cascade_path):
        logging.error(f"Haar cascade file not found at {args.cascade_path}")
        sys.exit(3)

    face_cascade = cv2.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        logging.error("Could not load Haar cascade.")
        sys.exit(3)

    # Parse min-size argument
    try:
        min_size = tuple(map(int, args.min_size.split(",")))
    except Exception as e:
        logging.error(f"Invalid format for min-size: {e}")
        sys.exit(6)

    # Detect faces in the image
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=min_size,
        )
    except Exception as e:
        logging.error(f"Error during face detection: {e}")
        sys.exit(7)

    num_faces = len(faces)
    logging.info(f"Number of faces found: {num_faces}")
    if not args.quiet:
        print(f"Number of faces found: {num_faces}")

    # Output face coordinates if requested
    if args.face_coordinates:
        output_face_coordinates(faces, args.output_format)

    # If output_path is specified, draw rectangles and save image
    if args.output_path or args.display:
        # Draw rectangles around faces
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if args.output_path:
            cv2.imwrite(args.output_path, image)
            logging.info(f"Output image saved to {args.output_path}")

    # If display is True, show the image
    if args.display:
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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


if __name__ == "__main__":
    main()
