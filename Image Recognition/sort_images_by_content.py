#!/usr/bin/env python3

# -------------------------------------------------------
# Script: sort_images_by_content.py
#
# Description:
# This script organizes images into directories based on
# keywords extracted from their captions. It builds a hierarchical
# in-memory grouping of images up to the specified depth, ensuring
# that each level provides finer granularity with different labels.
# After grouping, it moves images to their final destinations.
#
# Usage:
# ./sort_images_by_content.py [source_directory] [options]
#
# - [source_directory]: The directory containing the images to be organized.
#
# Options:
# -l LEVEL, --level LEVEL         Maximum directory depth level for sorting (default: 1).
# -k NUM_KEYWORDS, --num-keywords NUM_KEYWORDS
#                                 Number of keywords to extract per image at each level (default: 5).
# -n MAX_DIRS, --max-dirs MAX_DIRS
#                                 Maximum number of directories to create at each level (default: 5).
# -v, --verbose                   Enable verbose output.
#
# Requirements:
# - PyTorch (install via: pip install torch torchvision)
# - Transformers (install via: pip install transformers)
# - Pillow (install via: pip install pillow)
# - spaCy (install via: pip install spacy)
# - spaCy English model (install via: python -m spacy download en_core_web_sm)
#
# -------------------------------------------------------

import argparse
import logging
import os
import shutil
import sys
from typing import List, Dict, Any
import warnings

import torch
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
from collections import Counter, defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Recursively organize images based on keywords extracted from their captions.'
    )
    parser.add_argument(
        'source_directory',
        type=str,
        help='The directory containing the images to be organized.'
    )
    parser.add_argument(
        '-l',
        '--level',
        type=int,
        default=1,
        help='Maximum directory depth level for sorting (default: 1).'
    )
    parser.add_argument(
        '-k',
        '--num-keywords',
        type=int,
        default=5,
        help='Number of keywords to extract per image at each level (default: 5).'
    )
    parser.add_argument(
        '-n',
        '--max-dirs',
        type=int,
        default=5,
        help='Maximum number of directories to create at each level (default: 5).'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    args = parser.parse_args()
    return args

def setup_logging(verbose: bool):
    """
    Sets up the logging configuration.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def load_models():
    """
    Loads the image captioning model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image captioning model
    caption_model_name = 'Salesforce/blip-image-captioning-base'
    caption_processor = BlipProcessor.from_pretrained(caption_model_name)
    caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)
    caption_model.eval()
    caption_model.to(device)

    return caption_processor, caption_model, device

def preprocess_image(image_path: str) -> Image.Image:
    """
    Loads and preprocesses an image.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logging.error(f"Error loading image '{image_path}': {e}")
        return None

def generate_caption(caption_model, caption_processor, device, image: Image.Image) -> str:
    """
    Generates a caption for the image.
    """
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = caption_model.generate(**inputs, max_new_tokens=20)
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return caption

def extract_keywords(caption: str, nlp, exclude_words: set) -> List[str]:
    """
    Extracts significant keywords from the caption using spaCy, excluding specified words.
    """
    doc = nlp(caption)
    # Extract nouns and proper nouns, excluding words in exclude_words
    keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.lemma_ not in exclude_words]
    return keywords

def build_hierarchy(
    images: List[str],
    captions: Dict[str, str],
    nlp,
    level: int,
    max_level: int,
    num_keywords: int,
    max_dirs: int,
    parent_keywords: set
) -> Dict[str, Any]:
    """
    Builds a hierarchical grouping of images based on keywords.
    """
    if level > max_level or not images:
        return {'images': images}

    # Extract keywords for each image, excluding parent keywords
    image_keywords = {}
    for image_path in images:
        caption = captions[image_path]
        keywords = extract_keywords(caption, nlp, parent_keywords)
        image_keywords[image_path] = keywords

    # Collect all keywords
    all_keywords = [kw for kws in image_keywords.values() for kw in kws]
    keyword_counts = Counter(all_keywords)
    most_common_keywords = [kw for kw, _ in keyword_counts.most_common(max_dirs)]

    # Assign images to groups based on keywords
    groups = defaultdict(list)
    for image_path, keywords in image_keywords.items():
        assigned = False
        for kw in most_common_keywords:
            if kw in keywords:
                groups[kw].append(image_path)
                assigned = True
                break
        if not assigned:
            groups['other'].append(image_path)

    # Build hierarchy recursively for each group
    hierarchy = {}
    for group_name, group_images in groups.items():
        new_parent_keywords = parent_keywords.union({group_name})
        subgroup = build_hierarchy(
            group_images,
            captions,
            nlp,
            level + 1,
            max_level,
            num_keywords,
            max_dirs,
            new_parent_keywords
        )
        hierarchy[group_name] = subgroup

    return hierarchy

def move_images(
    hierarchy: Dict[str, Any],
    base_dir: str,
    path_parts: List[str]
):
    """
    Moves images to their final destinations based on the hierarchy.
    """
    for key, value in hierarchy.items():
        if key == 'images':
            # Move images to the current path
            target_dir = os.path.join(base_dir, *path_parts)
            os.makedirs(target_dir, exist_ok=True)
            for image_path in value:
                image_name = os.path.basename(image_path)
                target_path = os.path.join(target_dir, image_name)
                try:
                    shutil.move(image_path, target_path)
                    logging.debug(f"Moved '{image_name}' to '{target_dir}'")
                except Exception as e:
                    logging.error(f"Failed to move '{image_name}' to '{target_dir}': {e}")
        else:
            # Recurse into subgroups
            move_images(value, base_dir, path_parts + [key.replace(' ', '_')])

def organize_images(args):
    """
    Main function to organize images based on keywords.
    """
    source_dir = args.source_directory
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Load models
    caption_processor, caption_model, device = load_models()
    nlp = spacy.load('en_core_web_sm')

    # Collect image paths
    image_files = [
        os.path.join(source_dir, f) for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
    ]

    if not image_files:
        logging.error(f"No image files found in directory '{source_dir}'.")
        sys.exit(1)

    logging.info(f"Generating captions for {len(image_files)} images...")

    # Generate captions for all images
    captions = {}
    for image_path in image_files:
        image = preprocess_image(image_path)
        if image is None:
            continue
        caption = generate_caption(caption_model, caption_processor, device, image)
        captions[image_path] = caption

    logging.info("Building hierarchical grouping...")

    # Build the hierarchy
    hierarchy = build_hierarchy(
        images=image_files,
        captions=captions,
        nlp=nlp,
        level=1,
        max_level=args.level,
        num_keywords=args.num_keywords,
        max_dirs=args.max_dirs,
        parent_keywords=set()
    )

    logging.info("Moving images to final destinations...")

    # Move images based on the hierarchy
    move_images(hierarchy, source_dir, [])

    logging.info("Image organization completed.")

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    organize_images(args)

if __name__ == '__main__':
    main()
