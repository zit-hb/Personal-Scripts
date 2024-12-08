#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_face_mask.py
#
# Description:
# This script detects all faces in an image using Dlib, determines their
# contours using a facial landmark predictor, and generates a binary mask
# where faces are black and the rest of the image is white.
#
# Additionally, if debug mode is enabled, a debug image showing the detected
# faces, their confidence values, the face contours, and individual landmark
# points will be saved as well.
#
# Usage:
# ./create_face_mask.py [input_path] [output_path] [options]
#
# Arguments:
#   - [input_path]:  The path to the input image file.
#   - [output_path]: The path to save the output mask image.
#
# Options:
#   -c, --confidence CONFIDENCE    Minimum confidence for face detection (default: 0.5).
#   -v, --verbose                  Enable verbose logging (INFO level).
#   -vv, --debug                   Enable debug logging (DEBUG level).
#   -d, --debug-image              Save a debug image showing face contours and landmarks.
#
# Template: ubuntu22.04
#
# Requirements:
#   - CMake (install via: apt-get install -y cmake)
#   - OpenCV (install via: apt-get install -y python3-opencv opencv-data)
#   - Dlib (install via: pip install dlib==19.24.6)
#   - NumPy (install via: pip install numpy==1.21.5)
#   - requests (install via: pip install requests==2.32.3)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import bz2
import cv2
import dlib
import logging
import numpy as np
import os
import requests
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Constants
DEFAULT_CONFIDENCE = 0.5
SHAPE_PREDICTOR_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
CACHE_DIR = Path.home() / '.cache' / 'create_face_mask'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTOR_PATH = CACHE_DIR / 'shape_predictor_68_face_landmarks.dat'


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Create a binary face mask from an image using Dlib face detection.'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input image file.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='The path to save the output mask image.'
    )
    parser.add_argument(
        '-c',
        '--confidence',
        type=float,
        default=DEFAULT_CONFIDENCE,
        help='Minimum confidence for face detection (default: 0.5).'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    parser.add_argument(
        '-vv',
        '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )
    parser.add_argument(
        '-d',
        '--debug-image',
        action='store_true',
        help='Save a debug image showing face contours and landmarks.'
    )
    args = parser.parse_args()

    # Validate input image
    if not os.path.isfile(args.input_path):
        parser.error(f'Input image "{args.input_path}" does not exist.')

    return args


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR

    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def download_and_extract_shape_predictor(destination: Path) -> None:
    """
    Download and extract the shape predictor file if not present.
    """
    if destination.is_file():
        logging.info(f'Shape predictor already available at {destination}.')
        return

    logging.info(f'Downloading shape predictor from {SHAPE_PREDICTOR_URL}.')
    try:
        response = requests.get(SHAPE_PREDICTOR_URL, stream=True)
        response.raise_for_status()
        compressed_path = destination.with_suffix('.dat.bz2')

        with open(compressed_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f'Downloaded compressed shape predictor to {compressed_path}.')

        # Extract bz2 file
        logging.info('Extracting shape predictor file.')
        with bz2.open(compressed_path, 'rb') as f_in, open(destination, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        logging.info(f'Shape predictor file extracted to {destination}.')
        compressed_path.unlink()

    except Exception as e:
        logging.error(f'Failed to download or extract shape predictor: {e}')
        sys.exit(1)


def load_dlib_models(predictor_path: Path) -> Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
    """
    Load the Dlib face detector and shape predictor models.
    """
    logging.info('Loading Dlib models...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    logging.info('Dlib models loaded successfully.')
    return detector, predictor


def detect_faces(
    image: np.ndarray,
    detector: dlib.fhog_object_detector,
    confidence_threshold: float
) -> Tuple[List[dlib.rectangle], List[float]]:
    """
    Detect faces in an image using Dlib's get_frontal_face_detector.
    Returns faces and their corresponding confidence scores above the given threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, scores, _ = detector.run(gray, 1, -1)

    filtered_faces = []
    filtered_scores = []
    for face, score in zip(faces, scores):
        if score >= confidence_threshold:
            filtered_faces.append(face)
            filtered_scores.append(score)

    logging.debug(f'Detected {len(filtered_faces)} faces above confidence {confidence_threshold}.')
    return filtered_faces, filtered_scores


def get_face_landmarks(
    image: np.ndarray,
    predictor: dlib.shape_predictor,
    face_rect: dlib.rectangle
) -> List[Tuple[int, int]]:
    """
    Get the 68-point facial landmarks for a given face rectangle using the predictor.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face_rect)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return points


def create_mask(
    image_shape: Tuple[int, int],
    all_points: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """
    Create a mask with faces filled in black and the background white.
    all_points is a list of lists of (x,y) landmarks for each face.
    """
    (h, w) = image_shape
    mask = (255 * (np.ones((h, w), dtype='uint8'))).copy()
    for points in all_points:
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 0)
    return mask


def create_debug_image(
    original_image: np.ndarray,
    faces: List[dlib.rectangle],
    scores: List[float],
    all_points: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """
    Create a debug image with bounding boxes, confidence values, face contours,
    and landmark points overlaid on the original image.
    """
    debug_img = original_image.copy()

    for face, score, points in zip(faces, scores, all_points):
        # Draw bounding box
        cv2.rectangle(
            debug_img,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (0, 255, 0), 2
        )
        # Draw confidence text
        text = f'{score*100:.1f}%'
        cv2.putText(
            debug_img,
            text,
            (face.left(), face.top() - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        # Draw face contour
        hull = cv2.convexHull(np.array(points))
        cv2.polylines(debug_img, [hull], True, (255, 0, 0), 2)

        # Draw landmark points
        for (x, y) in points:
            cv2.circle(debug_img, (x, y), 1, (0, 0, 255), -1)

    return debug_img


def main() -> None:
    """
    Main function to orchestrate the face mask creation process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)
    logging.info('Starting face mask creation.')

    # Ensure shape predictor file is available
    download_and_extract_shape_predictor(PREDICTOR_PATH)

    # Load the image
    image = cv2.imread(args.input_path)
    if image is None:
        logging.error(f'Failed to load image "{args.input_path}".')
        sys.exit(1)

    (h, w) = image.shape[:2]

    # Load models
    detector, predictor = load_dlib_models(PREDICTOR_PATH)

    # Detect faces
    faces, scores = detect_faces(image, detector, args.confidence)

    # If no faces detected
    if not faces:
        if args.debug_image:
            # If debug_image is enabled, just produce a debug image with no faces.
            # The debug image in this case would look like the original image as there are no faces.
            cv2.imwrite(args.output_path, image)
            logging.info(f'No faces detected. Debug image saved to "{args.output_path}"')
        else:
            # Otherwise, produce a white mask
            mask = 255 * np.ones((h, w), dtype='uint8')
            cv2.imwrite(args.output_path, mask)
            logging.info('No faces detected. Saved a white mask.')
        sys.exit(0)

    # Extract face landmarks
    all_points = [get_face_landmarks(image, predictor, face) for face in faces]

    if args.debug_image:
        # If debug_image is enabled, only save the debug image
        debug_img = create_debug_image(image, faces, scores, all_points)
        cv2.imwrite(args.output_path, debug_img)
        logging.info(f'Debug image saved to "{args.output_path}"')
    else:
        # Otherwise, save the mask
        mask = create_mask((h, w), all_points)
        cv2.imwrite(args.output_path, mask)
        logging.info(f'Mask saved to "{args.output_path}"')


if __name__ == '__main__':
    main()
