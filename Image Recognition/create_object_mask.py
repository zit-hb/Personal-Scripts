#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_object_mask.py
#
# Description:
# This script detects specified objects in an image using a pre-trained model
# from a library like Detectron2 and creates a mask where the detected objects
# are black, and the background is white.
#
# Usage:
# ./create_object_mask.py [image_file] [output_file] [object_type] [-t THRESHOLD] [-m MODEL]
#
# - [image_file]: The path to the input image file.
# - [output_file]: The path to save the output mask image.
# - [object_type]: The type of object to mask (e.g., "person", "cat", etc.)
# - [-t THRESHOLD, --threshold THRESHOLD]: Confidence threshold for object detection (default: 0.5).
# - [-m MODEL, --model MODEL]: Model name to use (default: COCO-InstanceSegmentation).
#
# Requirements:
# - Torch & Torchvision (install via: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117)
# - Detectron2 (install via: pip install 'git+https://github.com/facebookresearch/detectron2.git')
#
# -------------------------------------------------------

import cv2
import numpy as np
import argparse
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

# Set up argument parser
parser = argparse.ArgumentParser(description='Create an object mask from an image for a specific object type.')
parser.add_argument('image_file', type=str, help='The path to the input image file.')
parser.add_argument('output_file', type=str, help='The path to save the output mask image.')
parser.add_argument('object_type', type=str, help='The type of object to detect (e.g., person, cat, etc.).')
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Confidence threshold for object detection (default: 0.5).')
parser.add_argument('-m', '--model', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', help='Model name to use (default: COCO-InstanceSegmentation).')

args = parser.parse_args()

# Initialize Detectron2 model config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(args.model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold  # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model)

# Create a predictor object from the Detectron2 config
predictor = DefaultPredictor(cfg)

# Load the image
image = cv2.imread(args.image_file)
if image is None:
    print("Error: Could not load image.")
    exit(1)

# Make predictions
outputs = predictor(image)

# Get class metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Retrieve the class ID corresponding to the specified object type
object_class = None
for idx, class_name in enumerate(metadata.thing_classes):
    if class_name.lower() == args.object_type.lower():
        object_class = idx
        break

if object_class is None:
    print(f"Error: Object type '{args.object_type}' not found in the model.")
    exit(1)

# Extract predictions and masks for the specified object type
instances = outputs["instances"]
pred_classes = instances.pred_classes
pred_masks = instances.pred_masks

# Create a white mask (all pixels initially white)
mask = np.ones(image.shape[:2], dtype="uint8") * 255

# Loop over all detected objects and apply masks for the specified class
for i, pred_class in enumerate(pred_classes):
    if pred_class == object_class:
        # Convert mask to binary and apply it (set object regions to black)
        object_mask = pred_masks[i].cpu().numpy()
        mask[object_mask] = 0

# Save the resulting mask to the specified output file
cv2.imwrite(args.output_file, mask)

print(f"Object mask for '{args.object_type}' saved to {args.output_file}")
