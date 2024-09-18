#!/usr/bin/env python3

# -------------------------------------------------------
# Script: sort_images_by_content_improved.py
#
# Description:
# This script organizes images into hierarchical clusters based on visual similarity.
# It uses CLIP for feature extraction and the WD14 tagger for labeling clusters.
# Clustering is performed using HDBSCAN, and dimensionality reduction is done using UMAP.
#
# Usage:
# ./sort_images_by_content_improved.py [source_directory] [options]
#
# - [source_directory]: The directory containing the images to be organized.
#
# Options:
# -l LEVEL, --level LEVEL                     Maximum directory depth level for sorting (default: 1).
# --min-cluster-size MIN_CLUSTER_SIZE         Minimum cluster size for HDBSCAN (default: 2).
# --min-samples MIN_SAMPLES                   Minimum samples for HDBSCAN (default: 1).
# --tfidf-threshold TFIDF_THRESHOLD           Threshold for TF-IDF scores when labeling clusters (default: 0.0).
# --tag-probability-threshold TAG_PROB        Probability threshold for including tags (default: 0.001).
# --verbose                                   Enable verbose output.
#
# Requirements:
# - PyTorch (install via: pip install torch torchvision)
# - transformers (install via: pip install transformers)
# - scikit-learn (install via: pip install scikit-learn)
# - umap-learn (install via: pip install umap-learn)
# - hdbscan (install via: pip install hdbscan)
# - Pillow (install via: pip install pillow)
# - onnxruntime (install via: pip install onnxruntime)
# - pandas (install via: pip install pandas)
# - huggingface-hub (install via: pip install huggingface-hub)
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
import hdbscan
import umap.umap_ as umap
import numpy as np
import pandas as pd
import onnxruntime as ort

from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download
from numba.core.errors import NumbaWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch._utils')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=UserWarning, module='umap')
warnings.filterwarnings("ignore", category=UserWarning, module='numba')
warnings.filterwarnings("ignore", category=NumbaWarning)


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
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum cluster size for HDBSCAN (default: 2).'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=1,
        help='Minimum samples for HDBSCAN (default: 1).'
    )
    parser.add_argument(
        '--tfidf-threshold',
        type=float,
        default=0.0,
        help='Threshold for TF-IDF scores when labeling clusters (default: 0.0).'
    )
    parser.add_argument(
        '--tag-probability-threshold',
        type=float,
        default=0.001,
        help='Probability threshold for including tags (default: 0.001).'
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
    Loads the CLIP model for feature extraction and sets up the ONNX runtime for WD14 tagger.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CLIP model for feature extraction
    clip_model_name = 'openai/clip-vit-base-patch32'
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Setup ONNX runtime session for WD14 tagger
    wd14_model_name = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    wd14_model_path = hf_hub_download(repo_id=wd14_model_name, filename='model.onnx')
    wd14_session = ort.InferenceSession(wd14_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Load tag list with categories
    tags_path = hf_hub_download(repo_id=wd14_model_name, filename='selected_tags.csv')
    tags_df = pd.read_csv(tags_path)
    tag_list = tags_df['name'].tolist()
    tag_categories = tags_df['category'].tolist()

    return clip_model, clip_processor, wd14_session, tag_list, tag_categories, device


def preprocess_image_for_wd14(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the image for WD14 tagger.
    """
    image = image.convert('RGB')
    image = image.resize((448, 448), resample=Image.BICUBIC)
    image_data = np.array(image, dtype=np.float32)
    image_data = image_data / 255.0  # Normalize to [0, 1]
    image_data = (image_data - 0.5) / 0.5  # Scale to [-1, 1]
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image_data


def extract_features(model, processor, device, image_paths: List[str]) -> (np.ndarray, List[str]):
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
            features.append(image_features)
            valid_image_paths.append(image_path)
            logging.debug(f"Extracted features for '{image_path}'")
        except Exception as e:
            logging.error(f"Error processing image '{image_path}': {e}")

    return np.array(features), valid_image_paths


def reduce_dimensions(features: np.ndarray) -> np.ndarray:
    """
    Reduces the dimensionality of the feature vectors using UMAP.
    """
    reducer = umap.UMAP(n_components=50, random_state=42)
    reduced_features = reducer.fit_transform(features)
    logging.info(f"UMAP reduced features to {reduced_features.shape[1]} dimensions.")
    return reduced_features


def cluster_features(features: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """
    Clusters the feature vectors using HDBSCAN.
    """
    n_samples = len(features)
    if n_samples <= min_cluster_size:
        # Not enough points to cluster, assign all to one cluster
        labels = np.zeros(n_samples, dtype=int)
        return labels

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    try:
        labels = clusterer.fit_predict(features)
    except ValueError as e:
        logging.error(f"Clustering failed: {e}. Assigning all points to a single cluster.")
        labels = np.zeros(n_samples, dtype=int)
        return labels

    # Check if all points are assigned to noise
    if np.all(labels == -1):
        # Assign all points to a single cluster
        labels = np.zeros(n_samples, dtype=int)
    else:
        # Assign noise points (-1) to their own unique clusters
        noise_indices = np.where(labels == -1)[0]
        max_label = labels.max() if labels.max() >= 0 else 0
        for idx in noise_indices:
            max_label += 1
            labels[idx] = max_label

    return labels


def get_wd14_labels(session, tag_list: List[str], tag_categories: List[str], image_paths: List[str], tag_prob_threshold: float) -> Dict[str, Dict[str, float]]:
    """
    Assigns labels to images using WD14 tagger, including character tags.
    """
    labels = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image_data = preprocess_image_for_wd14(image)
            inputs = {session.get_inputs()[0].name: image_data}
            outputs = session.run(None, inputs)
            probs = outputs[0][0]
            tags = {}
            for idx, prob in enumerate(probs):
                tag = tag_list[idx]
                category = tag_categories[idx]
                # Include general and character tags
                if category in ['character'] and prob >= tag_prob_threshold:
                    tags[tag] = prob
            labels[image_path] = tags
        except Exception as e:
            logging.error(f"Error assigning tags to image '{image_path}': {e}")
            labels[image_path] = {}
    return labels


def label_clusters(image_labels: Dict[str, Dict[str, float]], cluster_image_paths: List[str], parent_labels: set, cluster_id: int, level: int, global_tag_counts: Dict[str, int], tfidf_threshold: float) -> str:
    """
    Determines the most informative label for a cluster using adjusted TF-IDF, ensuring it's not in parent labels.
    """
    # Calculate term frequencies (TF) for tags in the cluster
    cluster_tag_counts = {}
    for image_path in cluster_image_paths:
        tags = image_labels.get(image_path, {})
        for tag, prob in tags.items():
            cluster_tag_counts[tag] = cluster_tag_counts.get(tag, 0) + prob

    # Calculate adjusted TF-IDF scores for tags in the cluster
    tf_idf_scores = {}
    total_tags_in_cluster = sum(cluster_tag_counts.values())
    total_documents = len(image_labels)
    for tag, count in cluster_tag_counts.items():
        if tag in parent_labels:
            continue
        tf = count / total_tags_in_cluster
        df = global_tag_counts.get(tag, 1)
        idf = np.log((total_documents + 1) / (df + 1)) + 1  # Smoothing
        tf_idf = tf * idf
        tf_idf_scores[tag] = tf_idf

    # Filter tags based on TF-IDF threshold
    filtered_tags = {tag: score for tag, score in tf_idf_scores.items() if score >= tfidf_threshold}

    if filtered_tags:
        cluster_label = max(filtered_tags, key=filtered_tags.get)
    else:
        cluster_label = f"unlabeled_{np.random.randint(1000)}"

    # Ensure uniqueness by appending cluster_id and level
    cluster_label = f"{cluster_label}_{cluster_id}_{level}"
    logging.debug(f"Cluster labeled as '{cluster_label}' with TF-IDF scores: {tf_idf_scores}")
    return cluster_label


def build_hierarchy(
    features: np.ndarray,
    image_paths: List[str],
    clip_model,
    clip_processor,
    wd14_session,
    tag_list,
    tag_categories,
    device,
    method: str,
    level: int,
    max_level: int,
    parent_labels: set,
    global_tag_counts: Dict[str, int],
    image_labels: Dict[str, Dict[str, float]],
    min_cluster_size: int,
    min_samples: int,
    tfidf_threshold: float
):
    """
    Builds the hierarchical clustering and labeling.
    """
    n_samples = len(image_paths)
    logging.info(f"Building hierarchy at level {level} with {n_samples} images.")

    if level > max_level or n_samples <= min_cluster_size:
        return image_paths  # Return list of image paths directly

    # Cluster features
    labels = cluster_features(features, min_cluster_size, min_samples)

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, {'features': [], 'image_paths': []})
        clusters[label]['features'].append(features[idx])
        clusters[label]['image_paths'].append(image_paths[idx])

    hierarchy = {}
    for label_id, cluster_data in clusters.items():
        cluster_features_array = np.array(cluster_data['features'])
        cluster_image_paths = cluster_data['image_paths']

        logging.info(f"Level {level} Cluster {label_id} contains {len(cluster_image_paths)} images.")

        # Label the cluster
        cluster_label = label_clusters(
            image_labels,
            cluster_image_paths,
            parent_labels,
            label_id,
            level,
            global_tag_counts,
            tfidf_threshold
        ).replace(' ', '_')

        # Update parent labels
        new_parent_labels = parent_labels.union({cluster_label})

        logging.info(f"Cluster {label_id} at level {level} labeled as '{cluster_label}'.")

        # Recursively build hierarchy
        subtree = build_hierarchy(
            features=np.array(cluster_features_array),
            image_paths=cluster_image_paths,
            clip_model=clip_model,
            clip_processor=clip_processor,
            wd14_session=wd14_session,
            tag_list=tag_list,
            tag_categories=tag_categories,
            device=device,
            method=method,
            level=level + 1,
            max_level=max_level,
            parent_labels=new_parent_labels,
            global_tag_counts=global_tag_counts,
            image_labels=image_labels,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            tfidf_threshold=tfidf_threshold
        )

        hierarchy[cluster_label] = subtree

    return hierarchy


def move_images(
    hierarchy: Any,
    base_dir: str,
    path_parts: List[str]
):
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
    Main function to organize images based on visual similarity and WD14 labels.
    """
    source_dir = args.source_directory
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Load models
    clip_model, clip_processor, wd14_session, tag_list, tag_categories, device = load_models()

    # Collect image paths
    image_files = [
        os.path.join(source_dir, f) for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
    ]

    if not image_files:
        logging.error(f"No image files found in directory '{source_dir}'.")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to process.")

    logging.info("Extracting features from images...")
    features, valid_image_paths = extract_features(clip_model, clip_processor, device, image_files)

    if features.size == 0:
        logging.error("No valid images to process.")
        sys.exit(1)

    logging.info("Reducing feature dimensions...")
    features_reduced = reduce_dimensions(features)

    logging.info("Assigning tags to images using WD14 tagger...")
    image_labels = get_wd14_labels(
        wd14_session,
        tag_list,
        tag_categories,
        valid_image_paths,
        tag_prob_threshold=args.tag_probability_threshold
    )

    logging.info("Calculating global tag counts...")
    # Calculate how many images each tag appears in
    global_tag_counts = {}
    for tags in image_labels.values():
        unique_tags = set(tags.keys())
        for tag in unique_tags:
            global_tag_counts[tag] = global_tag_counts.get(tag, 0) + 1

    logging.info("Building hierarchical clustering and labeling...")
    hierarchy = build_hierarchy(
        features=features_reduced,
        image_paths=valid_image_paths,
        clip_model=clip_model,
        clip_processor=clip_processor,
        wd14_session=wd14_session,
        tag_list=tag_list,
        tag_categories=tag_categories,
        device=device,
        method='hdbscan',
        level=1,
        max_level=args.level,
        parent_labels=set(),
        global_tag_counts=global_tag_counts,
        image_labels=image_labels,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        tfidf_threshold=args.tfidf_threshold
    )

    logging.info("Moving images to final destinations...")
    move_images(hierarchy, source_dir, [])

    logging.info("Image organization completed.")


def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    organize_images(args)


if __name__ == '__main__':
    main()
