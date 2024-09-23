#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_object_mask.py
#
# Description:
# This script detects objects in an image using a pre-trained model
# from the Detectron2 library and creates masks for each detected object type.
# It can list available object types, process multiple object types,
# and handle batch processing of images.
#
# Usage:
# ./create_object_mask.py [image_file|image_directory] [output_file|output_directory] [options]
#
# - [image_file]: The path to the input image file.
# - [image_directory]: The path to the input image directory (when using --batch).
# - [output_file]: The path to save the output mask image(s).
# - [output_directory]: The path to save the output mask images (when using --batch).
#
# Options:
# -o OBJECT_TYPE [OBJECT_TYPE ...], --object_type OBJECT_TYPE [OBJECT_TYPE ...]
#                               The type(s) of object(s) to detect (e.g., "person", "cat").
#                               If omitted, masks for all detected object types are created.
# -t THRESHOLD, --threshold THRESHOLD
#                               Confidence threshold for object detection (default: 0.5).
# -m MODEL, --model MODEL       Model name to use (default: COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml).
# --invert                      Invert the masks (objects will be white and background black).
# --batch                       Process a batch of images in a directory.
# --cpu                         Force the model to run on CPU.
# --list-objects                List all detectable object classes by the model and exit.
#
# Requirements:
# - Torch & Torchvision
# - Detectron2
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


def get_predictor(model_name: str, threshold: float, use_cpu: bool = False) -> DefaultPredictor:
    """
    Initializes and returns a Detectron2 predictor.
    """
    if not use_cpu and not torch.cuda.is_available():
        logging.warning("CUDA is not available. The model will run on CPU.")
        use_cpu = True

    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        cfg.MODEL.DEVICE = 'cpu' if use_cpu else 'cuda'
        predictor = DefaultPredictor(cfg)
        return predictor
    except Exception as e:
        logging.error(f"Failed to initialize model '{model_name}': {e}")
        sys.exit(1)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Loads an image from the given path.
    """
    if not os.path.isfile(image_path):
        logging.error(f"Image file '{image_path}' does not exist.")
        return None
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image '{image_path}'.")
    return image


def list_object_classes(metadata):
    """
    Prints all detectable object classes by the model.
    """
    print("Detectable object classes:")
    for idx, class_name in enumerate(metadata.thing_classes):
        print(f"{idx}: {class_name}")
    sys.exit(0)


def get_object_classes(object_types: List[str], metadata) -> List[int]:
    """
    Maps object types to class indices based on the model's metadata.
    """
    object_classes = []
    for obj_type in object_types:
        found = False
        for idx, class_name in enumerate(metadata.thing_classes):
            if class_name.lower() == obj_type.lower():
                object_classes.append(idx)
                found = True
                break
        if not found:
            logging.warning(f"Object type '{obj_type}' not found in the model's classes.")
    return object_classes


def create_masks(
    image: np.ndarray,
    predictor: DefaultPredictor,
    object_classes: Optional[List[int]],
    invert: bool = False
) -> Optional[dict]:
    """
    Creates masks for each detected object type in the image.
    """
    try:
        outputs = predictor(image)
        instances = outputs["instances"]
        if not instances.has("pred_classes") or not instances.has("pred_masks"):
            logging.warning("No instances detected in the image.")
            return None
        pred_classes = instances.pred_classes
        pred_masks = instances.pred_masks
    except Exception as e:
        logging.error(f"Model prediction failed: {e}")
        return None

    # Initialize a dictionary to hold masks for each object type
    masks = {}

    # Process each detected instance
    for i, pred_class in enumerate(pred_classes):
        class_idx = pred_class.item()
        class_name = predictor.metadata.thing_classes[class_idx]
        if object_classes is not None and class_idx not in object_classes:
            continue  # Skip if not in the specified object classes

        # Create a mask for the object type if it doesn't exist
        if class_name not in masks:
            # Initialize the mask
            mask_shape = image.shape[:2]
            masks[class_name] = np.zeros(mask_shape, dtype="uint8") if invert else np.ones(mask_shape, dtype="uint8") * 255

        # Update the mask for the object type
        object_mask = pred_masks[i].cpu().numpy()
        masks[class_name][object_mask] = 255 if invert else 0

    return masks


def save_masks(masks: dict, output_path: str, base_name: str):
    """
    Saves the masks to the specified output path.
    """
    if not masks:
        logging.warning("No masks to save.")
        return

    for class_name, mask in masks.items():
        output_filename = f"{base_name}_{class_name}.png"
        output_file = os.path.join(output_path, output_filename)
        success = cv2.imwrite(output_file, mask)
        if success:
            logging.info(f"Mask for '{class_name}' saved to '{output_file}'.")
        else:
            logging.error(f"Failed to save mask for '{class_name}' to '{output_file}'.")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create object masks from an image for specific object types.'
    )
    parser.add_argument(
        'image_file',
        type=str,
        help='The path to the input image file or directory (use --batch for directories).'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='The path to save the output mask image(s) or directory (use --batch for directories).'
    )
    parser.add_argument(
        '-o',
        '--object_type',
        type=str,
        nargs='+',
        help='The type(s) of object(s) to detect (e.g., person, cat). If omitted, all detected object types are used.'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for object detection (default: 0.5).'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        help='Model name to use (default: COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml).'
    )
    parser.add_argument(
        '--invert',
        action='store_true',
        help='Invert the masks (objects will be white and background black).'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process a batch of images in a directory.'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force the model to run on CPU.'
    )
    parser.add_argument(
        '--list-objects',
        action='store_true',
        help='List all detectable object classes by the model and exit.'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Initialize predictor
    predictor = get_predictor(args.model, args.threshold, args.cpu)

    # Get model metadata
    cfg = predictor.cfg
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    predictor.metadata = metadata  # Attach metadata to predictor for easy access

    # List object classes if requested
    if args.list_objects:
        list_object_classes(metadata)

    # Map object types to class indices
    object_classes = None
    if args.object_type:
        object_classes = get_object_classes(args.object_type, metadata)
        if not object_classes:
            logging.error("None of the specified object types were found in the model's classes.")
            sys.exit(1)

    # Validate and prepare paths
    if args.batch:
        if not os.path.isdir(args.image_file):
            logging.error(f"Input path '{args.image_file}' is not a directory.")
            sys.exit(1)
        if not os.path.isdir(args.output_file):
            os.makedirs(args.output_file, exist_ok=True)
        image_files = [
            os.path.join(args.image_file, f) for f in os.listdir(args.image_file)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
        output_dir = args.output_file
    else:
        if not os.path.isfile(args.image_file):
            logging.error(f"Input file '{args.image_file}' does not exist.")
            sys.exit(1)
        image_files = [args.image_file]
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for image_path in image_files:
        image = load_image(image_path)
        if image is None:
            continue
        masks = create_masks(image, predictor, object_classes, invert=args.invert)
        if masks is None:
            logging.warning(f"No objects detected in image '{image_path}'.")
            continue

        # Determine base name for output files
        if args.batch:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = os.path.splitext(os.path.basename(args.output_file))[0]

        # Determine output directory
        if args.batch:
            output_path = output_dir
        else:
            output_path = output_dir if output_dir else '.'

        # Save the masks
        save_masks(masks, output_path, base_name)

    logging.info("Processing completed.")


if __name__ == '__main__':
    main()
