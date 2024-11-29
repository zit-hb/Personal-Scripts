#!/usr/bin/env python3

# -------------------------------------------------------
# Script: sort_images_by_content.py
#
# Description:
# This script organizes images into hierarchical clusters based on visual similarity.
# It uses CLIP for feature extraction and generates labels based on the most similar common English nouns.
# Clustering is performed using Agglomerative Clustering, and dimensionality reduction is done using UMAP.
# Labels are assigned by selecting the most similar noun to the cluster's centroid.
#
# Usage:
# ./sort_images_by_content.py [source_directory] [options]
#
# - [source_directory]: The directory containing the images to be organized.
#
# Options:
# -l LEVEL, --level LEVEL                     Maximum directory depth level for sorting (default: 1).
# --max-clusters MAX_CLUSTERS                 Maximum number of clusters per level (default: 10).
# --min-cluster-size MIN_CLUSTER_SIZE         Minimum cluster size for Agglomerative Clustering (default: 2).
# --verbose                                   Enable verbose output.
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
# - PyTorch (install via: pip install torch torchvision)
# - transformers (install via: pip install transformers)
# - scikit-learn (install via: pip install scikit-learn)
# - umap-learn (install via: pip install umap-learn)
# - nltk (install via: pip install nltk)
# - Pillow (install via: pip install pillow)
# - pandas (install via: pip install pandas)
# - huggingface-hub (install via: pip install huggingface-hub)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import shutil
import sys
from typing import List, Dict, Any, Set

import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import umap.umap_ as umap
import numpy as np
import pandas as pd

from transformers import CLIPProcessor, CLIPModel

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch._utils')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=UserWarning, module='umap')
warnings.filterwarnings("ignore", category=UserWarning, module='numba')
from numba.core.errors import NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)

import nltk
from nltk.corpus import brown, wordnet as wn

# Ensure WordNet data is downloaded
nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Organize images into hierarchical clusters based on visual similarity.'
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
        '--max-clusters',
        type=int,
        default=10,
        help='Maximum number of clusters per level (default: 10).'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum cluster size for Agglomerative Clustering (default: 2).'
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


def load_models():
    """
    Loads the CLIP model for feature extraction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Loading CLIP model on device: {device}")

    # Load CLIP model
    clip_model_name = 'openai/clip-vit-base-patch32'
    logging.info(f"Loading CLIP model '{clip_model_name}' for feature extraction...")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    logging.info("CLIP model loaded successfully.")
    return clip_model, clip_processor, device


def get_common_nouns(top_n: int = 5000) -> List[str]:
    """
    Retrieves a list of the most common English nouns from the Brown Corpus.
    Limits the list to the top_n most frequent nouns to ensure relevance.
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


def compute_label_embeddings(clip_model: CLIPModel, clip_processor: CLIPProcessor, device: torch.device, labels: List[str]) -> np.ndarray:
    """
    Computes CLIP text embeddings for a list of labels.
    """
    logging.info("Computing text embeddings for labels...")
    inputs = clip_processor(text=labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)
    text_embeddings = text_embeddings.cpu().numpy()
    # Normalize embeddings
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    logging.info("Text embeddings computed successfully.")
    return text_embeddings


def extract_features(model: CLIPModel, processor: CLIPProcessor, device: torch.device, image_paths: List[str]) -> (np.ndarray, List[str]):
    """
    Extracts features from all images using CLIP.
    """
    features = []
    valid_image_paths = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = image_features.cpu().numpy().flatten()
            # Normalize features
            image_features /= np.linalg.norm(image_features)
            features.append(image_features)
            valid_image_paths.append(image_path)
            logging.debug(f"Extracted features for '{image_path}'")
        except Exception as e:
            logging.error(f"Error processing image '{image_path}': {e}")

    if not features:
        logging.error("No features extracted. Exiting.")
        sys.exit(1)

    return np.array(features), valid_image_paths


def reduce_dimensions(features: np.ndarray) -> np.ndarray:
    """
    Reduces the dimensionality of the feature vectors using UMAP.
    Ensures that the number of components is strictly less than the number of samples.
    """
    n_samples = len(features)

    if n_samples < 2:
        logging.error("Not enough samples to perform dimensionality reduction.")
        sys.exit(1)

    # Set number of components to be smaller than the number of samples
    n_components = min(30, n_samples // 2)  # Use half of the samples as the maximum number of components

    logging.info(f"Reducing to {n_components} dimensions using UMAP (for {n_samples} samples).")

    reducer = umap.UMAP(n_components=n_components, random_state=42)

    try:
        reduced_features = reducer.fit_transform(features)
        logging.info(f"UMAP reduced features to {reduced_features.shape[1]} dimensions.")
    except Exception as e:
        logging.error(f"UMAP dimensionality reduction failed: {e}")
        sys.exit(1)

    return reduced_features


def cluster_features(features: np.ndarray, min_cluster_size: int, max_clusters: int) -> np.ndarray:
    """
    Clusters the feature vectors using Agglomerative Clustering.
    Limits the number of clusters to 'max_clusters'.
    """
    n_samples = len(features)
    if n_samples < min_cluster_size:
        logging.info("Not enough samples to form clusters. Assigning all images to a single cluster.")
        return np.zeros(n_samples, dtype=int)

    # Determine the number of clusters: min(max_clusters, n_samples // min_cluster_size)
    possible_clusters = n_samples // min_cluster_size
    n_clusters = min(max_clusters, possible_clusters) if possible_clusters > 1 else 1

    logging.info(f"Clustering into {n_clusters} clusters using Agglomerative Clustering.")

    try:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(features)
    except Exception as e:
        logging.error(f"Agglomerative Clustering failed: {e}. Assigning all points to a single cluster.")
        return np.zeros(n_samples, dtype=int)

    num_clusters = len(set(labels))
    logging.info(f"Number of clusters formed: {num_clusters}")

    return labels


def assign_labels_to_clusters(
    cluster_image_paths: List[str],
    labels: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    text_embeddings: np.ndarray,
    cluster_id: int
) -> str:
    """
    Assigns a label to a cluster based on the most similar label from the label list.
    Prepends the cluster ID to ensure directory names are unique.
    """
    logging.debug(f"Assigning label for cluster {cluster_id} with {len(cluster_image_paths)} images.")

    # Compute the centroid of the cluster's feature vectors
    cluster_features = []
    for image_path in cluster_image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_feature = clip_model.get_image_features(**inputs)
            image_feature = image_feature.cpu().numpy().flatten()
            # Normalize
            image_feature /= np.linalg.norm(image_feature)
            cluster_features.append(image_feature)
        except Exception as e:
            logging.error(f"Error processing image '{image_path}' for centroid computation: {e}")

    if not cluster_features:
        label = f"{cluster_id}_unlabeled"
        logging.warning(f"No features available for cluster {cluster_id}. Using label '{label}'.")
        return label

    cluster_centroid = np.mean(cluster_features, axis=0)
    cluster_centroid /= np.linalg.norm(cluster_centroid)

    # Compute cosine similarity between cluster centroid and all label embeddings
    similarities = np.dot(text_embeddings, cluster_centroid)
    best_label_idx = np.argmax(similarities)
    best_label = labels[best_label_idx]

    # Ensure the label is not empty
    if not best_label.strip():
        label = f"{cluster_id}_unlabeled"
    else:
        # Replace spaces with underscores for directory naming
        label = best_label.replace(' ', '_')

    # Prepend cluster ID to ensure uniqueness
    unique_label = f"{cluster_id}_{label}"

    logging.debug(f"Assigned label '{unique_label}' to cluster {cluster_id}.")

    return unique_label


def build_hierarchy(
    features: np.ndarray,
    image_paths: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    text_labels: List[str],
    text_embeddings: np.ndarray,
    cluster_id_start: int,
    device: torch.device,
    method: str,
    level: int,
    max_level: int,
    max_clusters: int,
    min_cluster_size: int,
    used_labels: Set[str]
) -> Dict[str, Any]:
    """
    Recursively builds the hierarchical clustering and labeling.
    """
    n_samples = len(image_paths)
    logging.info(f"Building hierarchy at level {level} with {n_samples} images.")

    if level > max_level or n_samples < min_cluster_size:
        return image_paths  # Return list of image paths directly

    # Cluster the features
    labels = cluster_features(features, min_cluster_size, max_clusters)

    # Organize images into clusters
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, {'features': [], 'image_paths': []})
        clusters[label]['features'].append(features[idx])
        clusters[label]['image_paths'].append(image_paths[idx])

    hierarchy = {}
    current_cluster_id = cluster_id_start

    for label_id, cluster_data in clusters.items():
        cluster_features_array = np.array(cluster_data['features'])
        cluster_image_paths = cluster_data['image_paths']

        logging.info(f"Level {level} Cluster {label_id} contains {len(cluster_image_paths)} images.")

        # Assign a unique label to the cluster
        cluster_label = assign_labels_to_clusters(
            cluster_image_paths,
            text_labels,
            clip_model,
            clip_processor,
            device,
            text_embeddings,
            cluster_id=current_cluster_id
        )

        logging.info(f"Cluster {label_id} at level {level} labeled as '{cluster_label}'.")

        # Recursively build hierarchy
        subtree = build_hierarchy(
            features=cluster_features_array,
            image_paths=cluster_image_paths,
            clip_model=clip_model,
            clip_processor=clip_processor,
            text_labels=text_labels,
            text_embeddings=text_embeddings,
            cluster_id_start=current_cluster_id + 1,
            device=device,
            method=method,
            level=level + 1,
            max_level=max_level,
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            used_labels=used_labels
        )

        hierarchy[cluster_label] = subtree
        current_cluster_id += 1

    return hierarchy


def move_images(hierarchy: Any, base_dir: str, path_parts: List[str]):
    """
    Moves images to their final destinations based on the hierarchy.
    """
    if isinstance(hierarchy, list):
        # Leaf node: move images to the current path
        target_dir = os.path.join(base_dir, *path_parts)
        os.makedirs(target_dir, exist_ok=True)
        for image_path in hierarchy:
            image_name = os.path.basename(image_path)
            target_path = os.path.join(target_dir, image_name)
            try:
                shutil.move(image_path, target_path)
                logging.debug(f"Moved '{image_name}' to '{target_dir}'")
            except Exception as e:
                logging.error(f"Failed to move '{image_name}' to '{target_dir}': {e}")
    elif isinstance(hierarchy, dict):
        # Internal node: recurse into subgroups
        for key, value in hierarchy.items():
            move_images(value, base_dir, path_parts + [key])
    else:
        logging.error(f"Invalid hierarchy node: {hierarchy}")


def organize_images(args):
    """
    Main function to organize images based on visual similarity and CLIP-based labels.
    """
    source_dir = args.source_directory
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Load models
    clip_model, clip_processor, device = load_models()

    # Collect image paths
    image_files = [
        os.path.join(source_dir, f) for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
    ]

    if not image_files:
        logging.error(f"No image files found in directory '{source_dir}'.")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to process.")

    # Extract features using CLIP
    logging.info("Extracting features from images...")
    features, valid_image_paths = extract_features(clip_model, clip_processor, device, image_files)

    if features.size == 0:
        logging.error("No valid images to process.")
        sys.exit(1)

    # Reduce dimensions with UMAP
    logging.info("Reducing feature dimensions...")
    features_reduced = reduce_dimensions(features)

    # Generate label list from Brown Corpus
    logging.info("Generating label list from Brown Corpus...")
    label_list = get_common_nouns(top_n=5000)  # Adjust top_n as needed

    # Compute text embeddings for labels
    label_embeddings = compute_label_embeddings(clip_model, clip_processor, device, label_list)

    # Build hierarchical clustering and labeling
    logging.info("Building hierarchical clustering and labeling...")
    used_labels = set()  # Track labels that have been used
    cluster_id_start = 0  # Initialize cluster ID

    hierarchy = build_hierarchy(
        features=features_reduced,
        image_paths=valid_image_paths,
        clip_model=clip_model,
        clip_processor=clip_processor,
        text_labels=label_list,
        text_embeddings=label_embeddings,
        cluster_id_start=cluster_id_start,
        device=device,
        method='agglomerative',
        level=1,
        max_level=args.level,
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
        used_labels=used_labels
    )

    # Move images to their final destinations
    logging.info("Moving images to final destinations...")
    move_images(hierarchy, source_dir, [])

    logging.info("Image organization completed.")


def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    organize_images(args)


if __name__ == '__main__':
    main()
