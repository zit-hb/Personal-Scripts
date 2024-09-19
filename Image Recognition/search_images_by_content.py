#!/usr/bin/env python3

# -------------------------------------------------------
# Script: search_images_by_content.py
#
# Description:
# This script searches for images within a specified directory that match given textual queries.
# It utilizes the CLIP (Contrastive Language–Image Pretraining) model to compute the similarity
# between image features and the provided positive and negative text queries.
# By default, it outputs human-readable information about each matched image,
# including file details and similarity scores, sorted by relevance.
# An option is available to output only the file names of matched images for further processing.
# Images must match all positive queries and must not match any negative queries.
#
# Usage:
# ./search_images_by_content.py [source_directory] [--query "positive_query1" ...] [--negative-query "negative_query1" ...] [options]
#
# - [source_directory]: The directory containing the images to be searched.
# - --query (-q): The textual content to search for within the images (e.g., "a playful cat sitting on a mat").
#                 Multiple --query (-q) arguments can be provided. All must match.
# - --negative-query (-Q): The textual content that should NOT be present in the images.
#                          Multiple --negative-query (-Q) arguments can be provided. None should match.
#
# Options:
# -t PT, --positive-threshold PT             Similarity threshold for positive queries (default: 0.2).
# -T NT, --negative-threshold NT             Similarity threshold for negative queries (default: 0.2).
# -n, --names-only                           Output only the file names of matched images.
# --verbose                                  Enable verbose output.
#
# Requirements:
# - Python 3.7 or higher
# - PyTorch (install via: pip install torch torchvision)
# - transformers (install via: pip install transformers)
# - Pillow (install via: pip install pillow)
# - tqdm (install via: pip install tqdm)
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

# Suppress specific warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch._utils')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


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
        '-q', '--query',
        type=str,
        action='append',
        help='The textual content to search for within the images (e.g., "a playful cat sitting on a mat"). '
             'Multiple --query (-q) arguments can be provided. All must match.'
    )
    parser.add_argument(
        '-Q', '--negative-query',
        type=str,
        action='append',
        help='The textual content that should NOT be present in the images. '
             'Multiple --negative-query (-Q) arguments can be provided. None should match.'
    )
    parser.add_argument(
        '-t', '--positive-threshold',
        type=float,
        default=0.2,
        help='Similarity threshold for positive queries (default: 0.2).'
    )
    parser.add_argument(
        '-T', '--negative-threshold',
        type=float,
        default=0.2,
        help='Similarity threshold for negative queries (default: 0.2).'
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

    # Ensure at least one of --query or --negative-query is provided
    if not args.query and not args.negative_query:
        parser.error("At least one of --query (-q) or --negative-query (-Q) must be provided.")

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


def compute_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    image_paths: List[str],
    positive_queries: List[str],
    negative_queries: List[str],
    pos_threshold: float,
    neg_threshold: float
) -> List[Tuple[str, float, List[float], List[float]]]:
    """
    Computes the similarity between each image and the search queries.
    Images must match all positive queries and must not match any negative queries.
    Returns a list of matched images with their average positive similarity scores and individual similarities.
    """
    logging.info(f"Processing positive queries: {positive_queries}")
    if negative_queries:
        logging.info(f"Processing negative queries: {negative_queries}")

    # Prepare text inputs for positive queries
    pos_text_inputs = processor(text=positive_queries, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        pos_text_embeddings = model.get_text_features(**pos_text_inputs)
    pos_text_embeddings /= pos_text_embeddings.norm(p=2, dim=-1, keepdim=True)

    # Prepare text inputs for negative queries, if any
    if negative_queries:
        neg_text_inputs = processor(text=negative_queries, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            neg_text_embeddings = model.get_text_features(**neg_text_inputs)
        neg_text_embeddings /= neg_text_embeddings.norm(p=2, dim=-1, keepdim=True)
    else:
        neg_text_embeddings = None

    matched_images = []
    logging.info("Computing similarities for images...")
    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        try:
            image = Image.open(image_path).convert('RGB')
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**image_inputs)
            image_embedding /= image_embedding.norm(p=2, dim=-1, keepdim=True)

            # Compute cosine similarity with all positive queries
            pos_similarities = (image_embedding @ pos_text_embeddings.T).squeeze(0)
            pos_similarities = pos_similarities.cpu().numpy()

            # Check if all positive similarities meet the threshold
            if not all(sim >= pos_threshold for sim in pos_similarities):
                continue  # Skip this image

            # If negative queries exist, compute similarities and check thresholds
            if neg_text_embeddings is not None:
                neg_similarities = (image_embedding @ neg_text_embeddings.T).squeeze(0)
                neg_similarities = neg_similarities.cpu().numpy()
                if any(sim >= neg_threshold for sim in neg_similarities):
                    continue  # Skip this image
            else:
                neg_similarities = []

            # Calculate average positive similarity for sorting
            avg_pos_similarity = pos_similarities.mean()

            matched_images.append((image_path, avg_pos_similarity, pos_similarities.tolist(), neg_similarities.tolist()))
        except Exception as e:
            logging.error(f"Error processing image '{image_path}': {e}")
    logging.info(f"Found {len(matched_images)} image(s) matching the search criteria.")
    return matched_images


def display_matched_images(
    matched_images: List[Tuple[str, float, List[float], List[float]]],
    positive_queries: List[str],
    negative_queries: List[str]
):
    """
    Displays the list of matched images with their similarity scores and file information.
    """
    if not matched_images:
        logging.info("No images matched the search criteria.")
        return
    logging.info("Matched Images:")
    for image_path, avg_similarity, pos_sims, neg_sims in matched_images:
        try:
            stats = os.stat(image_path)
            creation_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            modification_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"File: {image_path}")
            print(f"  Average Similarity to Positive Queries: {avg_similarity:.4f}")
            for idx, query in enumerate(positive_queries):
                print(f"    Positive Query [{idx+1}]: '{query}' | Similarity: {pos_sims[idx]:.4f}")
            if negative_queries:
                for idx, query in enumerate(negative_queries):
                    print(f"    Negative Query [{idx+1}]: '{query}' | Similarity: {neg_sims[idx]:.4f}")
            print(f"  Created: {creation_time}")
            print(f"  Modified: {modification_time}")
            print("-" * 60)
        except Exception as e:
            logging.error(f"Error retrieving information for '{image_path}': {e}")


def output_filenames(matched_images: List[Tuple[str, float, List[float], List[float]]]):
    """
    Outputs only the file names of matched images.
    """
    if not matched_images:
        logging.info("No images matched the search criteria.")
        return
    for image_path, _, _, _ in matched_images:
        print(image_path)


def main():
    """
    The main function orchestrating the search process.
    """
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

    # Compute similarities
    matched_images = compute_similarity(
        model=clip_model,
        processor=clip_processor,
        device=device,
        image_paths=image_paths,
        positive_queries=args.query if args.query else [],
        negative_queries=args.negative_query if args.negative_query else [],
        pos_threshold=args.positive_threshold,
        neg_threshold=args.negative_threshold
    )

    # Sort matched images by average similarity in descending order
    matched_images.sort(key=lambda x: x[1], reverse=True)

    # Output results
    if args.names_only:
        output_filenames(matched_images)
    else:
        display_matched_images(
            matched_images=matched_images,
            positive_queries=args.query if args.query else [],
            negative_queries=args.negative_query if args.negative_query else []
        )


if __name__ == '__main__':
    main()
