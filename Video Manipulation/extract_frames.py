#!/usr/bin/env python3

# -------------------------------------------------------
# Script: extract_frames.py
#
# Description:
# This script extracts frames from given video files or directories containing video files,
# filters them based on blurriness, compression artifacts, and content matching using YOLOv8,
# and saves the resulting frames to an output directory.
#
# Usage:
# ./extract_frames.py [options] [input_paths ...]
#
# Arguments:
#   - [input_paths ...]: The path(s) to input video file(s) or directory(s) containing video files.
#
# Options:
#   -o OUTPUT_DIR, --output-dir OUTPUT_DIR
#       The directory to save the extracted frames.
#       Defaults to the current working directory if not provided.
#   -s FRAME_RATE, --frame-rate FRAME_RATE
#       Frame extraction rate (frames per second). Default is to extract every frame.
#   -b BLURRY_THRESHOLD, --blurry-threshold BLURRY_THRESHOLD
#       Threshold for detecting blurry images.
#   -a COMPRESSION_THRESHOLD, --compression-threshold COMPRESSION_THRESHOLD
#       Threshold for compression artifacts detection.
#   -c CLASS_NAMES, --class CLASS_NAMES
#       The object class names to detect within the frames (e.g., "person", "car").
#       Multiple --class (-c) arguments can be provided. Any matching class will be accepted.
#   -C NEGATIVE_CLASS_NAMES, --negative-class NEGATIVE_CLASS_NAMES
#       The object class names that should NOT be present in the frames.
#       Multiple --negative-class (-C) arguments can be provided. None should match.
#   -S START_TIME, --start START_TIME
#       Start time of the video segment to process. Supports formats like "ss", "mm:ss", "hh:mm:ss".
#   -E END_TIME, --end END_TIME
#       End time of the video segment to process. Supports formats like "ss", "mm:ss", "hh:mm:ss".
#   -v, --verbose
#       Enable verbose output.
#   -d, --dry-run
#       Show what would be done without making any changes.
#   -r, --recursive
#       Recursively search for video files in the specified directory(s).
#   -h, --help
#       Display this help message.
#
# Template: ubuntu22.04
#
# Requirements:
#   - OpenCV (install via: apt-get install -y python3-opencv opencv-data)
#   - Ultralytics YOLOv8 (install via: pip install ultralytics==8.3.39)
#   - Pillow (install via: pip install pillow==11.0.0)
#   - ImageHash (install via: pip install ImageHash==4.3.1)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import cv2
import os
import sys
import logging
import numpy as np
from typing import Optional, Tuple, List
from ultralytics import YOLO
import urllib.request
import imagehash
from PIL import Image
from collections import OrderedDict

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'extract_frames')
MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt'
MODEL_FILENAME = 'yolov8m.pt'
MAX_CACHE_ENTRIES = 1000

# List of video file extensions to consider
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.m4v']


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Extract frames from videos, filter by blurriness, compression artifacts, and content matching using YOLOv8.'
    )
    parser.add_argument(
        'input_paths',
        nargs='+',
        type=str,
        help='The path(s) to input video file(s) or directory(s) containing video files.'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        help='The directory to save the extracted frames. Defaults to the current working directory.'
    )
    parser.add_argument(
        '-s',
        '--frame-rate',
        type=float,
        help='Frame extraction rate (frames per second). Default is to extract every frame.'
    )
    parser.add_argument(
        '-b',
        '--blurry-threshold',
        type=float,
        default=0,
        help='Threshold for detecting blurry images.'
    )
    parser.add_argument(
        '-a',
        '--compression-threshold',
        type=float,
        default=0,
        help='Threshold for compression artifacts detection.'
    )
    parser.add_argument(
        '-c',
        '--class',
        dest='class_names',
        type=str,
        action='append',
        help='The object class names to detect within the frames (e.g., "person", "car"). '
             'Multiple --class (-c) arguments can be provided. Any matching class will be accepted.'
    )
    parser.add_argument(
        '-C',
        '--negative-class',
        dest='negative_class_names',
        type=str,
        action='append',
        help='The object class names that should NOT be present in the frames. '
             'Multiple --negative-class (-C) arguments can be provided. None should match.'
    )
    parser.add_argument(
        '-S',
        '--start',
        type=str,
        help='Start time of the video segment to process. Supports formats like "ss", "mm:ss", "hh:mm:ss".'
    )
    parser.add_argument(
        '-E',
        '--end',
        type=str,
        help='End time of the video segment to process. Supports formats like "ss", "mm:ss", "hh:mm:ss".'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '-d',
        '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes.'
    )
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Recursively search for video files in the specified directory(s).'
    )
    args = parser.parse_args()
    return args


def setup_logging(verbose: bool) -> None:
    """
    Sets up the logging configuration.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def is_video_file(file_path: str) -> bool:
    """
    Checks if the given file path has a video file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def get_video_files(input_paths: List[str], recursive: bool) -> List[str]:
    """
    Retrieves a list of video files from the input paths.
    """
    video_files = []
    for path in input_paths:
        if os.path.isfile(path):
            if is_video_file(path):
                video_files.append(path)
            else:
                logging.warning(f"File '{path}' is not a supported video format. Skipping.")
        elif os.path.isdir(path):
            if recursive:
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if is_video_file(file_path):
                            video_files.append(file_path)
            else:
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path) and is_video_file(file_path):
                        video_files.append(file_path)
        else:
            logging.warning(f"Path '{path}' is not a file or directory. Skipping.")
    return video_files


def validate_video_file(video_file: str) -> None:
    """
    Validates that the video file exists and is accessible.
    """
    if not os.path.isfile(video_file):
        logging.error(f"Video file '{video_file}' not found.")
        sys.exit(1)


def create_output_directory(base_output_dir: Optional[str], video_file: str, dry_run: bool) -> str:
    """
    Creates the output directory for the video if it does not exist.
    The directory is named after the video file (without extension) inside the base output directory.
    """
    if base_output_dir is None:
        base_output_dir = os.getcwd()
        logging.info(f"No output directory specified. Using current working directory: '{base_output_dir}'")

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(base_output_dir, video_name)

    if not os.path.exists(output_dir):
        if dry_run:
            logging.info(f"Would create output directory '{output_dir}'")
        else:
            os.makedirs(output_dir)
            logging.info(f"Created output directory '{output_dir}'")

    return output_dir


def download_model(model_url: str, model_path: str) -> None:
    """
    Downloads the YOLOv8 model from the specified URL to the given path.
    """
    try:
        logging.info(f"Downloading YOLOv8 model from '{model_url}' to '{model_path}'...")
        urllib.request.urlretrieve(model_url, model_path)
        logging.info("Model downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        sys.exit(1)


def load_yolov8_model() -> Tuple[YOLO, List[str]]:
    """
    Loads the YOLOv8 model, downloads it if necessary, and retrieves class names.
    Returns the model and the list of class names.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logging.info(f"Created cache directory '{CACHE_DIR}'")

    model_path = os.path.join(CACHE_DIR, MODEL_FILENAME)

    # Download the model if it does not exist
    if not os.path.isfile(model_path):
        download_model(MODEL_URL, model_path)
    else:
        logging.info(f"YOLOv8 model already exists at '{model_path}'")

    # Load the YOLOv8 model
    try:
        model = YOLO(model_path)
        logging.info("YOLOv8 model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        sys.exit(1)

    # Retrieve class names directly from the model
    try:
        model_classes = model.names
        class_names = list(model_classes.values())
        logging.info("Retrieved class names from YOLOv8 model.")
    except Exception as e:
        logging.error(f"Failed to retrieve class names from YOLOv8 model: {e}")
        sys.exit(1)

    return model, class_names


def validate_class_names(specified_classes: Optional[List[str]], available_classes: List[str]) -> None:
    """
    Validates that the specified class names are available in the model's class list.
    """
    if not specified_classes:
        return

    invalid_classes = [cls for cls in specified_classes if cls not in available_classes]
    if invalid_classes:
        logging.error(f"The following specified classes are not recognized by YOLOv8: {invalid_classes}")
        logging.error(f"Available classes are: {available_classes}")
        sys.exit(1)
    else:
        logging.info("All specified classes are valid.")


def open_video_capture(video_file: str) -> cv2.VideoCapture:
    """
    Opens the video file for processing.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Could not open video file '{video_file}'")
        sys.exit(5)
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> Tuple[float, int]:
    """
    Retrieves properties from the video capture.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    logging.info(f"Video FPS: {fps}")
    logging.info(f"Total frames: {total_frames}")
    logging.info(f"Duration: {duration:.2f} seconds")

    return fps, total_frames


def calculate_frame_interval(fps: float, frame_rate: Optional[float]) -> int:
    """
    Calculates the frame interval based on desired frame rate.
    """
    if frame_rate and frame_rate > 0:
        frame_interval = int(round(fps / frame_rate))
        if frame_interval == 0:
            frame_interval = 1
    else:
        frame_interval = 1  # Process every frame

    logging.info(f"Processing every {frame_interval} frames")
    return frame_interval


def parse_timecode(time_str: Optional[str]) -> Optional[float]:
    """
    Parses a time string into seconds.
    Supports formats like "ss", "mm:ss", "hh:mm:ss".
    """
    if time_str is None:
        return None
    try:
        parts = time_str.strip().split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 1:
            # ss
            seconds = parts[0]
        elif len(parts) == 2:
            # mm:ss
            seconds = parts[0]*60 + parts[1]
        elif len(parts) == 3:
            # hh:mm:ss
            seconds = parts[0]*3600 + parts[1]*60 + parts[2]
        else:
            logging.error(f"Invalid time format: '{time_str}'")
            sys.exit(1)
        return seconds
    except ValueError:
        logging.error(f"Invalid time value: '{time_str}'")
        sys.exit(1)


def convert_times_to_frames(start_time: Optional[float], end_time: Optional[float], fps: float, total_frames: int) -> Tuple[int, int]:
    """
    Converts start and end times to frame numbers.
    """
    if start_time is not None:
        start_frame = int(start_time * fps)
    else:
        start_frame = 0

    if end_time is not None:
        end_frame = int(end_time * fps)
    else:
        end_frame = total_frames - 1

    return start_frame, end_frame


def validate_frame_numbers(start_frame: int, end_frame: int, total_frames: int, start_time: Optional[float], end_time: Optional[float]) -> None:
    """
    Validates the frame numbers.
    """
    if start_frame < 0 or start_frame >= total_frames:
        logging.error(f"Start time {start_time} seconds is out of bounds.")
        sys.exit(1)

    if end_frame < 0 or end_frame >= total_frames:
        logging.error(f"End time {end_time} seconds is out of bounds.")
        sys.exit(1)

    if start_frame > end_frame:
        logging.error("Start time is after end time.")
        sys.exit(1)

    logging.info(f"Processing frames from {start_frame} to {end_frame}")


def get_cached_result(cache: OrderedDict, hash_value: str) -> Optional[List[str]]:
    """
    Retrieves the cached result for the given hash.
    Returns the list of detected classes or None if not cached.
    """
    if hash_value in cache:
        cache.move_to_end(hash_value)  # Mark as recently used
        logging.debug(f"Cache hit for hash {hash_value}")
        return cache[hash_value]
    logging.debug(f"Cache miss for hash {hash_value}")
    return None


def insert_cache(cache: OrderedDict, hash_value: str, result: List[str], max_entries: int = MAX_CACHE_ENTRIES) -> None:
    """
    Inserts the result into the cache.
    """
    cache[hash_value] = result
    cache.move_to_end(hash_value)  # Mark as recently used
    if len(cache) > max_entries:
        evicted = cache.popitem(last=False)  # Remove least recently used
        logging.debug(f"Cache full. Evicted oldest cache entry: {evicted[0]}")
    logging.debug(f"Inserted hash {hash_value} into cache")


def initialize_in_memory_cache() -> OrderedDict:
    """
    Initializes the in-memory cache.
    """
    return OrderedDict()


def process_video_frames(
    cap: cv2.VideoCapture,
    args: argparse.Namespace,
    output_dir: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    model: YOLO,
    available_classes: List[str],
    cache: OrderedDict
) -> None:
    """
    Processes frames from the video capture.
    """
    frame_interval = calculate_frame_interval(fps, args.frame_rate)
    saved_frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate class names
    validate_class_names(args.class_names, available_classes)
    validate_class_names(args.negative_class_names, available_classes)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_number = start_frame

    while frame_number <= end_frame:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame {frame_number}. Skipping.")
            frame_number += 1
            continue

        if (frame_number - start_frame) % frame_interval != 0:
            frame_number += 1
            continue  # Skip this frame

        logging.info(f"Processing frame {frame_number}/{total_frames}")

        process_result = process_frame(
            frame, frame_number, args, output_dir, model, available_classes, cache)
        if process_result:
            saved_frame_count += 1

        frame_number += 1

    cap.release()
    logging.info(f"Processing complete. Saved {saved_frame_count} frames to '{output_dir}'")


def process_frame(
    frame: np.ndarray,
    frame_number: int,
    args: argparse.Namespace,
    output_dir: str,
    model: YOLO,
    available_classes: List[str],
    cache: OrderedDict
) -> bool:
    """
    Processes a single frame: checks blurriness, compression artifacts, content matching,
    and saves the frame if conditions are met.
    Utilizes caching for YOLOv8 results to improve performance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check blurriness if threshold > 0
    if args.blurry_threshold > 0:
        variance = calculate_laplacian_variance(gray)
        is_blurry = variance < args.blurry_threshold

        if is_blurry:
            logging.info(f"Frame {frame_number} is blurry (variance: {variance:.2f})")
            return False

    # Check compression artifacts if threshold > 0
    if args.compression_threshold > 0:
        artifact_score = calculate_artifact_score(frame)
        has_artifacts = artifact_score > args.compression_threshold
        if has_artifacts:
            logging.info(f"Frame {frame_number} has high compression artifacts (score: {artifact_score:.2f})")
            return False

    # Content matching using YOLOv8 if class names are provided
    detected_classes = []
    if args.class_names or args.negative_class_names:
        # Compute image hash
        hash_value = compute_image_hash(frame)

        # Check cache
        cached_result = get_cached_result(cache, hash_value)
        if cached_result is not None:
            detected_classes = cached_result
        else:
            # Perform inference
            results = model(frame, verbose=False)  # Suppress model's own logging

            # Assume single image inference
            result = results[0]

            # Extract detected class names
            detected_classes = [available_classes[int(cls_id)] for cls_id in result.boxes.cls.tolist()]

            # Insert into cache
            insert_cache(cache, hash_value, detected_classes)

        # Check for positive class matches
        if args.class_names:
            positive_match = any(cls in detected_classes for cls in args.class_names)
            if not positive_match:
                logging.info(f"Frame {frame_number} does not contain desired classes.")
                return False

        # Check for negative class matches
        if args.negative_class_names:
            negative_match = any(cls in detected_classes for cls in args.negative_class_names)
            if negative_match:
                logging.info(f"Frame {frame_number} contains undesired classes.")
                return False

    # Save frame
    frame_filename = f"frame_{frame_number:04d}.jpg"
    frame_path = os.path.join(output_dir, frame_filename)
    if args.dry_run:
        logging.info(f"Would save frame {frame_number} to '{frame_path}'")
    else:
        success = cv2.imwrite(frame_path, frame)
        if success:
            logging.info(f"Saved frame {frame_number} to '{frame_path}'")
            return True
        else:
            logging.warning(f"Failed to save frame {frame_number} to '{frame_path}'")
            return False

    return True


def calculate_laplacian_variance(gray_image: np.ndarray) -> float:
    """
    Calculates the Laplacian variance of a grayscale image.
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def calculate_artifact_score(image: np.ndarray) -> float:
    """
    Calculates a compression artifact score based on blockiness detection.
    A common compression artifact in JPEG images is blockiness caused by the 8x8 DCT blocks.

    Approach:
    - Convert the image to grayscale.
    - JPEG compression typically operates on 8x8 pixel blocks. Compression artifacts often manifest
      as discontinuities along block boundaries.
    - We measure blockiness by summing the absolute differences in intensity across the vertical and horizontal
      boundaries that align with 8-pixel multiples.

    Steps:
    1. Convert the image to grayscale.
    2. For every vertical boundary at columns x = 8, 16, 24, ...:
       - Compute the absolute difference between pixels at column x-1 and x for all rows.
       - Sum these differences to get the vertical blockiness contribution.
    3. For every horizontal boundary at rows y = 8, 16, 24, ...:
       - Compute the absolute difference between pixels at row y-1 and y for all columns.
       - Sum these differences to get the horizontal blockiness contribution.
    4. The final artifact score is the sum of vertical and horizontal blockiness values.

    This metric will be higher for images with more pronounced block boundaries, which often correlate
    with high compression artifacts.

    Note:
    - This is a heuristic that focuses primarily on blockiness, a common JPEG artifact.
    - In practice, other factors may influence perceived compression quality, but this provides a
      logical, block-based approach for measuring common JPEG-like artifacts.
    """
    # Convert to grayscale
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    block_size: int = 8

    vertical_blockiness: float = 0.0
    # Check vertical boundaries
    for x in range(block_size, w, block_size):
        if x >= w:
            break
        col_diff: np.ndarray = np.abs(gray[:, x] - gray[:, x - 1])
        vertical_blockiness += float(np.sum(col_diff))

    horizontal_blockiness: float = 0.0
    # Check horizontal boundaries
    for y in range(block_size, h, block_size):
        if y >= h:
            break
        row_diff: np.ndarray = np.abs(gray[y, :] - gray[y - 1, :])
        horizontal_blockiness += float(np.sum(row_diff))

    artifact_score: float = vertical_blockiness + horizontal_blockiness
    return artifact_score


def compute_image_hash(frame: np.ndarray) -> str:
    """
    Computes the perceptual hash of the given frame using imagehash.
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hash_value = imagehash.phash(image)
    return str(hash_value)


def main() -> None:
    """
    Main function to orchestrate the frame extraction process.
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    # Get list of video files
    video_files = get_video_files(args.input_paths, args.recursive)
    if not video_files:
        logging.error("No valid video files found.")
        sys.exit(1)

    # Load YOLOv8 model
    model, available_classes = load_yolov8_model()

    # Initialize cache
    cache = initialize_in_memory_cache()

    for video_file in video_files:
        logging.info(f"Processing video file '{video_file}'")
        validate_video_file(video_file)
        output_dir = create_output_directory(args.output_dir, video_file, args.dry_run)
        cap = open_video_capture(video_file)
        fps, total_frames = get_video_properties(cap)

        start_time = parse_timecode(args.start)
        end_time = parse_timecode(args.end)
        start_frame, end_frame = convert_times_to_frames(start_time, end_time, fps, total_frames)
        validate_frame_numbers(start_frame, end_frame, total_frames, start_time, end_time)

        try:
            process_video_frames(cap, args, output_dir, start_frame, end_frame, fps, model, available_classes, cache)
        finally:
            cap.release()


if __name__ == '__main__':
    main()
