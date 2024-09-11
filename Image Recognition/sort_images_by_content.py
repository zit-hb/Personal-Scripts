#!/usr/bin/env python3

# -------------------------------------------------------
# Script: sort_images_by_content.py
#
# Description:
# This script sorts images into directories based on their content using image
# recognition. Users can specify a directory depth and confidence threshold.
#
# Usage:
# ./sort_images_by_content.py [source_directory] [-l LEVEL] [-m MODEL] [-t CONFIDENCE_THRESHOLD] [-f FALLBACK] [-c CLASS_NAMES] [-v]
#
# - [source_directory]: The directory containing the images to be sorted.
# - [-l LEVEL, --level LEVEL]: The directory depth level for sorting (default: 1).
# - [-m MODEL, --model MODEL]: Pre-trained model to use for image recognition (default: CLIP).
# - [-t CONFIDENCE_THRESHOLD, --threshold CONFIDENCE_THRESHOLD]: Confidence threshold for sorting (default: 0.5).
# - [-f FALLBACK, --fallback FALLBACK]: Fallback directory for unrecognized or low-confidence images (default: unknown).
# - [-c CLASS_NAMES, --class-names CLASS_NAMES]: Comma-separated list of class names for image recognition.
# - [-v, --verbose]: Verbose mode to show detailed output.
#
# Requirements:
# - Torch & Torchvision (install via: pip install torch torchvision transformers)
#
# -------------------------------------------------------

import os
import shutil
import argparse
import tempfile
import torch
import warnings
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Set up argument parser
parser = argparse.ArgumentParser(description='Sort images by content into directories based on recognized objects.')
parser.add_argument('source_directory', type=str, help='The directory containing the images to be sorted.')
parser.add_argument('-l', '--level', type=int, default=1, help='Directory depth level for sorting (default: 1).')
parser.add_argument('-m', '--model', type=str, default='CLIP', help='Model name to use for image recognition (default: CLIP).')
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Confidence threshold for image recognition (default: 0.5).')
parser.add_argument('-f', '--fallback', type=str, default='unknown', help='Fallback directory for unrecognized or low-confidence images (default: unknown).')
parser.add_argument('-c', '--class-names', type=str, help='Comma-separated list of class names for image recognition.')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')

args = parser.parse_args()

# Load class names
if args.class_names:
    class_names = [name.strip() for name in args.class_names.split(",")]
else:
    class_names = [
        "person", "group", "animal", "car", "object", "landscape", "building", "document"
    ]

# Ensure the source directory exists
if not os.path.exists(args.source_directory):
    print(f"Error: Source directory {args.source_directory} does not exist.")
    exit(1)

# Set up the CLIP model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to predict the class of an image with confidence
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=[f"a photo of a {name}" for name in class_names], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Get the top predictions and their confidence
    top_probs, top_classes = probs[0].topk(args.level)
    return [(class_names[top_classes[i].item()], top_probs[i].item()) for i in range(args.level)]

# Function to create directory structure based on prediction and level
def get_target_directory(predictions, level):
    parts = [pred[0] for pred in predictions[:level]]
    target_path = os.path.join(*parts)
    return target_path

# Process each image in the source directory
for image_file in os.listdir(args.source_directory):
    image_path = os.path.join(args.source_directory, image_file)

    if os.path.isfile(image_path):
        try:
            # Get predictions and confidence
            predictions = predict_image(image_path)

            # Check if any prediction meets the confidence threshold
            target_directory = args.fallback
            for prediction, confidence in predictions:
                if confidence >= args.threshold:
                    target_directory = get_target_directory(predictions, args.level)
                    break

            # Create target directory if it doesn't exist
            target_path = os.path.join(args.source_directory, target_directory)
            os.makedirs(target_path, exist_ok=True)

            # Move the image to the target directory
            shutil.move(image_path, os.path.join(target_path, image_file))
            if args.verbose:
                print(f"Moved {image_file} to {target_path} with predictions {predictions}.")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print("Image sorting completed.")
