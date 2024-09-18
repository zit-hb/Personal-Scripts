#!/usr/bin/env python3

# -------------------------------------------------------
# Script: search_images_by_content.py
#
# Description:
# This script searches for images within a specified directory that contain specific content.
# It utilizes the CLIP (Contrastive Language–Image Pretraining) model to compute the similarity
# between image features and the provided text query. By default, it outputs human-readable
# information about each matched image, including file details, similarity scores, and relevant labels.
# An option is available to output only the file names of matched images for further processing.
# The output is sorted by similarity, with the most similar images listed first.
#
# Usage:
# ./search_images_by_content.py [source_directory] [search_term] [options]
#
# - [source_directory]: The directory containing the images to be searched.
# - [search_term]: The content to search for within the images (e.g., "cat").
#
# Options:
# -t THRESHOLD, --threshold THRESHOLD       Similarity threshold for matching images (default: 0.25).
# -n, --names-only                          Output only the file names of matched images.
# --verbose                                 Enable verbose output.
#
# Requirements:
# - Python 3.7 or higher
# - PyTorch (install via: pip install torch torchvision)
# - transformers (install via: pip install transformers)
# - Pillow (install via: pip install pillow)
# - tqdm (install via: pip install tqdm)
# - nltk (install via: pip install nltk)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List, Tuple
from datetime import datetime

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

import warnings
import nltk
from nltk.corpus import brown

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch._utils')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# Ensure NLTK data is downloaded
nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Search for images containing specific content using CLIP.'
    )
    parser.add_argument(
        'source_directory',
        type=str,
        help='The directory containing the images to be searched.'
    )
    parser.add_argument(
        'search_term',
        type=str,
        help='The content to search for within the images (e.g., "cat").'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.25,
        help='Similarity threshold for matching images (default: 0.25).'
    )
    parser.add_argument(
        '-n', '--names-only',
        action='store_true',
        help='Output only the file names of matched images.'
    )
    parser.add_argument(
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

def load_clip_model(device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Loads the CLIP model and processor.
    """
    logging.info("Loading CLIP model...")
    model_name = 'openai/clip-vit-base-patch32'
    try:
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        logging.info("CLIP model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load CLIP model '{model_name}': {e}")
        sys.exit(1)
    return clip_model, clip_processor

def get_image_paths(source_dir: str) -> List[str]:
    """
    Retrieves a list of image file paths from the source directory.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_paths = [
        os.path.join(source_dir, f) for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(supported_extensions)
    ]
    if not image_paths:
        logging.error(f"No image files found in directory '{source_dir}'.")
        sys.exit(1)
    logging.info(f"Found {len(image_paths)} image(s) to process.")
    return image_paths

def get_common_nouns(top_n: int = 1000) -> List[str]:
    """
    Retrieves a list of the most common English nouns from the Brown Corpus.
    """
    logging.info("Generating a list of common nouns from the Brown Corpus...")
    noun_counts = {}
    for word, tag in brown.tagged_words(tagset='universal'):
        if tag == 'NOUN':
            word = word.lower()
            noun_counts[word] = noun_counts.get(word, 0) + 1

    # Sort nouns by frequency
    sorted_nouns = sorted(noun_counts.items(), key=lambda x: x[1], reverse=True)
    common_nouns = [word for word, count in sorted_nouns[:top_n]]
    logging.info(f"Selected top {len(common_nouns)} common nouns.")
    return common_nouns

def compute_label_embeddings(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    labels: List[str]
) -> torch.Tensor:
    """
    Computes CLIP text embeddings for a list of labels.
    """
    logging.info("Computing text embeddings for labels...")
    batch_size = 100  # Adjust batch size based on memory constraints
    label_embeddings = []
    for i in tqdm(range(0, len(labels), batch_size), desc="Processing Labels", unit="batch"):
        batch_labels = labels[i:i+batch_size]
        inputs = clip_processor(text=batch_labels, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeddings = clip_model.get_text_features(**inputs)
        embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)
        label_embeddings.append(embeddings)
    label_embeddings = torch.cat(label_embeddings, dim=0)
    logging.info("Label embeddings computed successfully.")
    return label_embeddings

def compute_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    image_paths: List[str],
    search_term: str,
    threshold: float
) -> List[Tuple[str, float]]:
    """
    Computes the similarity between each image and the search term.
    """
    logging.info(f"Processing search term: '{search_term}'")
    # Prepare text inputs
    text_inputs = processor(text=[search_term], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)

    matched_images = []
    logging.info("Computing similarities for images...")
    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        try:
            image = Image.open(image_path).convert('RGB')
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embeddings = model.get_image_features(**image_inputs)
            image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
            # Compute cosine similarity
            similarity = (image_embeddings @ text_embeddings.T).item()
            logging.debug(f"Image: '{image_path}' | Similarity: {similarity:.4f}")
            if similarity >= threshold:
                matched_images.append((image_path, similarity))
        except Exception as e:
            logging.error(f"Error processing image '{image_path}': {e}")
    logging.info(f"Found {len(matched_images)} image(s) matching the search criteria.")
    return matched_images

def get_relevant_labels(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    image_path: str,
    label_embeddings: torch.Tensor,
    labels: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Retrieves the top-K relevant labels for a given image based on similarity.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embedding = model.get_image_features(**image_inputs)
        image_embedding /= image_embedding.norm(p=2, dim=-1, keepdim=True)
        # Compute cosine similarity with all labels
        similarities = (image_embedding @ label_embeddings.T).squeeze(0)
        topk = torch.topk(similarities, k=top_k)
        top_labels = [(labels[idx], similarity.item()) for idx, similarity in zip(topk.indices, topk.values)]
        return top_labels
    except Exception as e:
        logging.error(f"Error retrieving labels for image '{image_path}': {e}")
        return []

def display_matched_images(
    matched_images: List[Tuple[str, float]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    label_embeddings: torch.Tensor,
    labels: List[str]
):
    """
    Displays the list of matched images with their similarity scores and relevant labels.
    """
    if not matched_images:
        logging.info("No images matched the search criteria.")
        return
    logging.info("Matched Images:")
    for image_path, similarity in matched_images:
        try:
            stats = os.stat(image_path)
            creation_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            modification_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            relevant_labels = get_relevant_labels(model, processor, device, image_path, label_embeddings, labels)
            print(f"File: {image_path}")
            print(f"  Similarity to '{args.search_term}': {similarity:.4f}")
            print(f"  Created: {creation_time}")
            print(f"  Modified: {modification_time}")
            print("  Relevant Labels:")
            for label, score in relevant_labels:
                print(f"    - {label} ({score:.4f})")
            print("-" * 60)
        except Exception as e:
            logging.error(f"Error retrieving information for '{image_path}': {e}")

def output_filenames(matched_images: List[Tuple[str, float]]):
    """
    Outputs only the file names of matched images.
    """
    if not matched_images:
        logging.info("No images matched the search criteria.")
        return
    for image_path, _ in matched_images:
        print(image_path)

def main():
    """
    The main function orchestrating the search process.
    """
    global args
    args = parse_arguments()
    setup_logging(args.verbose)

    # Verify source directory
    if not os.path.isdir(args.source_directory):
        logging.error(f"Source directory '{args.source_directory}' does not exist.")
        sys.exit(1)

    # Load CLIP model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, clip_processor = load_clip_model(device)

    # Get list of image paths
    image_paths = get_image_paths(args.source_directory)

    # Get common nouns for labeling
    labels = get_common_nouns(top_n=1000)  # Adjust top_n as needed

    # Compute label embeddings
    label_embeddings = compute_label_embeddings(clip_model, clip_processor, device, labels)

    # Compute similarities
    matched_images = compute_similarity(
        model=clip_model,
        processor=clip_processor,
        device=device,
        image_paths=image_paths,
        search_term=args.search_term,
        threshold=args.threshold
    )

    # Sort matched images by similarity in descending order
    matched_images.sort(key=lambda x: x[1], reverse=True)

    # Output results
    if args.names_only:
        output_filenames(matched_images)
    else:
        display_matched_images(
            matched_images=matched_images,
            model=clip_model,
            processor=clip_processor,
            device=device,
            label_embeddings=label_embeddings,
            labels=labels
        )

if __name__ == '__main__':
    main()
