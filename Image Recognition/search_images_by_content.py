#!/usr/bin/env python3

# -------------------------------------------------------
# Script: search_images_by_content.py
#
# Description:
# This script searches for images within a specified directory that match given textual queries.
# It leverages three different techniques to achieve highly accurate and flexible image searches:
#
# 1. BLIP Image Captioning:
#    Automatically generates descriptive captions for each image.
#
# 2. Sentence Transformers (Semantic Similarity):
#    Compares the generated captions to user queries at a semantic level.
#
# 3. CLIP Visual-Text Alignment:
#    Uses CLIP to directly measure how well query text aligns with the visual content of each image.
#
# The final similarity score for each query is computed from a weighted combination of the
# semantic similarity (from captions) and the CLIP similarity.
# Positive queries must exceed a specified similarity threshold, while negative queries must remain below it.
#
# Usage:
# ./search_images_by_content.py [source_directory] [options]
#
# Arguments:
#   - [source_directory]: The directory containing the images to be searched.
#
# Options:
#   -q QUERY, --query QUERY                Positive query string. Multiple queries can be provided.
#   -Q QUERY, --negative-query QUERY       Negative query string. Multiple queries can be provided.
#   -C WEIGHT, --caption-weight WEIGHT     Weight for the caption-based semantic similarity (default: 1.0).
#   -M WEIGHT, --clip-weight WEIGHT        Weight for the CLIP similarity score (default: 1.0).
#   -PT, --positive-threshold PT           Similarity threshold for positive queries (default: 0.3).
#   -NT, --negative-threshold NT           Similarity threshold for negative queries (default: 0.3).
#   -n, --names-only                       Output only the file names of matched images.
#   -r, --recursive                        Recursively search for images in subdirectories.
#   -v, --verbose                          Enable verbose output.
#   -vv, --debug                           Enable debug logging.
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
#   - PyTorch (install via: pip install torch==2.5.1 torchvision==0.20.1)
#   - transformers (install via: pip install transformers==4.47.0)
#   - Pillow (install via: pip install pillow==11.0.0)
#   - sentence-transformers (install via: pip install sentence-transformers==3.3.1)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional
import warnings
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=UserWarning, module='torch._utils')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')


@dataclass
class CaptionResult:
    image_path: str
    caption: str


@dataclass
class SimilarityScores:
    st_pos_sims: List[float]
    st_neg_sims: List[float]
    clip_pos_sims: List[float]
    clip_neg_sims: List[float]
    success: bool


@dataclass
class MatchedImageResult:
    image_path: str
    avg_similarity: float
    pos_sims: List[float]
    neg_sims: List[float]


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Search images by content using BLIP, Sentence Transformers, and CLIP.'
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
        help='Positive query string. Multiple queries can be provided.'
    )
    parser.add_argument(
        '-Q', '--negative-query',
        type=str,
        action='append',
        help='Negative query string. Multiple queries can be provided.'
    )
    parser.add_argument(
        '-C', '--caption-weight',
        type=float,
        default=1.0,
        help='Weight for the caption-based semantic similarity.'
    )
    parser.add_argument(
        '-M', '--clip-weight',
        type=float,
        default=1.0,
        help='Weight for the CLIP similarity score.'
    )
    parser.add_argument(
        '-PT', '--positive-threshold',
        type=float,
        default=0.3,
        help='Similarity threshold for positive queries.'
    )
    parser.add_argument(
        '-NT', '--negative-threshold',
        type=float,
        default=0.3,
        help='Similarity threshold for negative queries.'
    )
    parser.add_argument(
        '-n', '--names-only',
        action='store_true',
        help='Output only the file names of matched images.'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively search for images in subdirectories.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '-vv', '--debug',
        action='store_true',
        help='Enable debug logging.'
    )

    args = parser.parse_args()
    if not args.query and not args.negative_query:
        parser.error("At least one of --query (-q) or --negative-query (-Q) must be provided.")
    return args


def setup_logging(verbose: bool, debug: bool) -> None:
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR

    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def get_image_paths(source_dir: str, recursive: bool) -> List[str]:
    """
    Retrieves a list of image file paths from the source directory.
    If recursive is True, searches subdirectories recursively.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_paths = []

    if recursive:
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(source_dir):
            fp = os.path.join(source_dir, f)
            if os.path.isfile(fp) and f.lower().endswith(supported_extensions):
                image_paths.append(fp)

    if not image_paths:
        logging.error(f"No image files found in directory '{source_dir}'.")
        sys.exit(1)
    logging.info(f"Found {len(image_paths)} image(s) to process.")
    return image_paths


def load_blip_model(device: torch.device) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Loads the BLIP model and processor for image captioning.
    """
    model_name = "Salesforce/blip-image-captioning-large"
    logging.info(f"Loading BLIP model '{model_name}' for captioning...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    logging.info("BLIP model loaded successfully.")
    return processor, model


def generate_captions(
    image_paths: List[str],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device
) -> List[CaptionResult]:
    """
    Generates captions for each image using BLIP.
    Returns a list of CaptionResult.
    """
    captions = []
    logging.info("Generating captions for images...")
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50)
            caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
            captions.append(CaptionResult(image_path=img_path, caption=caption))
            logging.debug(f"Caption for '{img_path}': {caption}")
        except Exception as e:
            logging.error(f"Failed to generate caption for '{img_path}': {e}")
    return captions


def load_sentence_model() -> SentenceTransformer:
    """
    Loads the sentence transformer model for semantic similarity.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logging.info(f"Loading sentence-transformer '{model_name}'...")
    st_model = SentenceTransformer(model_name)
    logging.info("Sentence-transformer model loaded successfully.")
    return st_model


def load_clip_model(device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Loads the CLIP model and processor for image-text matching.
    """
    clip_model_name = "openai/clip-vit-large-patch14"
    logging.info(f"Loading CLIP model '{clip_model_name}'...")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    logging.info("CLIP model loaded successfully.")
    return clip_model, clip_processor


def compute_clip_text_embeddings(
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
    texts: Optional[List[str]]
) -> Optional[torch.Tensor]:
    """
    Computes CLIP text embeddings for a list of text queries.
    """
    if not texts:
        return None
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    return text_emb


def compute_clip_image_embedding(
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
    img_path: str
) -> torch.Tensor:
    """
    Computes the CLIP image embedding for a single image.
    """
    image = Image.open(img_path).convert('RGB')
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
    return img_emb


def combine_scores(st_score: float, clip_score: float, caption_weight: float, clip_weight: float) -> float:
    """
    Combines the semantic similarity (ST) score and CLIP score using provided weights.
    """
    weighted_sum = (st_score * caption_weight) + (clip_score * clip_weight)
    total_weight = caption_weight + clip_weight
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def encode_queries(
    st_model: SentenceTransformer,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: torch.device,
    queries: Optional[List[str]],
    negative_queries: Optional[List[str]]
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor]
]:
    """
    Encodes positive and negative queries into ST and CLIP embeddings.
    """
    query_emb_st = st_model.encode(queries, convert_to_tensor=True) if queries else None
    neg_query_emb_st = st_model.encode(negative_queries, convert_to_tensor=True) if negative_queries else None

    query_emb_clip = compute_clip_text_embeddings(clip_processor, clip_model, device, queries) if queries else None
    neg_query_emb_clip = compute_clip_text_embeddings(clip_processor, clip_model, device, negative_queries) if negative_queries else None

    return query_emb_st, neg_query_emb_st, query_emb_clip, neg_query_emb_clip


def compute_image_similarities(
    img_path: str,
    caption: str,
    query_emb_st: Optional[torch.Tensor],
    neg_query_emb_st: Optional[torch.Tensor],
    query_emb_clip: Optional[torch.Tensor],
    neg_query_emb_clip: Optional[torch.Tensor],
    st_model: SentenceTransformer,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device
) -> SimilarityScores:
    """
    Computes ST and CLIP similarities for positive and negative queries for a single image.
    Returns a SimilarityScores dataclass.
    """
    caption_emb_st = st_model.encode([caption], convert_to_tensor=True)

    try:
        image_emb_clip = compute_clip_image_embedding(clip_processor, clip_model, device, img_path)
    except Exception as e:
        logging.error(f"Failed to compute CLIP embedding for '{img_path}': {e}")
        return SimilarityScores([], [], [], [], False)

    st_pos_sims = util.cos_sim(caption_emb_st, query_emb_st).cpu().numpy()[0].tolist() if query_emb_st is not None else []
    st_neg_sims = util.cos_sim(caption_emb_st, neg_query_emb_st).cpu().numpy()[0].tolist() if neg_query_emb_st is not None else []

    clip_pos_sims = (image_emb_clip @ query_emb_clip.T).cpu().numpy()[0].tolist() if query_emb_clip is not None else []
    clip_neg_sims = (image_emb_clip @ neg_query_emb_clip.T).cpu().numpy()[0].tolist() if neg_query_emb_clip is not None else []

    return SimilarityScores(st_pos_sims, st_neg_sims, clip_pos_sims, clip_neg_sims, True)


def combine_all_scores(
    queries: Optional[List[str]],
    negative_queries: Optional[List[str]],
    st_pos_sims: List[float],
    st_neg_sims: List[float],
    clip_pos_sims: List[float],
    clip_neg_sims: List[float],
    caption_weight: float,
    clip_weight: float
) -> Tuple[List[float], List[float]]:
    """
    Combines ST and CLIP similarities for positive and negative queries.
    """
    pos_sims_combined = []
    if queries:
        for i in range(len(queries)):
            st_s = st_pos_sims[i] if i < len(st_pos_sims) else 0.0
            cl_s = clip_pos_sims[i] if i < len(clip_pos_sims) else 0.0
            combined = combine_scores(st_s, cl_s, caption_weight, clip_weight)
            pos_sims_combined.append(combined)

    neg_sims_combined = []
    if negative_queries:
        for i in range(len(negative_queries)):
            st_s = st_neg_sims[i] if i < len(st_neg_sims) else 0.0
            cl_s = clip_neg_sims[i] if i < len(clip_neg_sims) else 0.0
            combined = combine_scores(st_s, cl_s, caption_weight, clip_weight)
            neg_sims_combined.append(combined)

    return pos_sims_combined, neg_sims_combined


def check_image_match_criteria(
    pos_sims_combined: List[float],
    neg_sims_combined: List[float],
    pos_threshold: float,
    neg_threshold: float
) -> bool:
    """
    Checks if an image meets the criteria for positive and negative thresholds.
    """
    if pos_sims_combined:
        positive_match = all(sim >= pos_threshold for sim in pos_sims_combined)
    else:
        positive_match = True  # If no positive queries, then no positive constraint

    negative_match = False
    if neg_sims_combined:
        negative_match = any(sim >= neg_threshold for sim in neg_sims_combined)

    return positive_match and not negative_match


def compute_similarity(
    captions_list: List[CaptionResult],
    queries: List[str],
    negative_queries: List[str],
    pos_threshold: float,
    neg_threshold: float,
    caption_weight: float,
    clip_weight: float,
    st_model: SentenceTransformer,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device
) -> List[MatchedImageResult]:
    """
    Computes the similarity for each image against the provided positive and negative queries.
    Returns a list of MatchedImageResult.
    """
    query_emb_st, neg_query_emb_st, query_emb_clip, neg_query_emb_clip = encode_queries(
        st_model, clip_processor, clip_model, device, queries, negative_queries
    )

    matched_images = []
    logging.info("Evaluating images against queries...")

    for caption_data in captions_list:
        scores = compute_image_similarities(
            caption_data.image_path, caption_data.caption,
            query_emb_st, neg_query_emb_st,
            query_emb_clip, neg_query_emb_clip,
            st_model, clip_model, clip_processor, device
        )
        if not scores.success:
            continue

        pos_sims_combined, neg_sims_combined = combine_all_scores(
            queries,
            negative_queries,
            scores.st_pos_sims,
            scores.st_neg_sims,
            scores.clip_pos_sims,
            scores.clip_neg_sims,
            caption_weight,
            clip_weight
        )

        if check_image_match_criteria(pos_sims_combined, neg_sims_combined, pos_threshold, neg_threshold):
            avg_sim = sum(pos_sims_combined) / len(pos_sims_combined) if pos_sims_combined else 0.0
            matched_images.append(MatchedImageResult(
                image_path=caption_data.image_path,
                avg_similarity=avg_sim,
                pos_sims=pos_sims_combined,
                neg_sims=neg_sims_combined
            ))
            logging.debug(
                f"Matched '{caption_data.image_path}' with avg similarity {avg_sim:.4f} "
                f"Pos: {pos_sims_combined}, Neg: {neg_sims_combined}"
            )

    matched_images.sort(key=lambda x: x.avg_similarity, reverse=True)
    return matched_images


def display_matched_images(
    matched_images: List[MatchedImageResult],
    positive_queries: List[str],
    negative_queries: List[str]
) -> None:
    """
    Displays the list of matched images with their similarity scores and file info.
    """
    if not matched_images:
        logging.info("No images matched the search criteria.")
        return
    logging.info("Matched Images:")
    for result in matched_images:
        display_single_image_result(
            result.image_path,
            result.avg_similarity,
            result.pos_sims,
            result.neg_sims,
            positive_queries,
            negative_queries
        )


def display_single_image_result(
    image_path: str,
    avg_similarity: float,
    pos_sims: List[float],
    neg_sims: List[float],
    positive_queries: List[str],
    negative_queries: List[str]
) -> None:
    """
    Displays information for a single matched image.
    """
    try:
        stats = os.stat(image_path)
        creation_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        modification_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"File: {image_path}")
        print(f"  Average Combined Similarity (Positive Queries): {avg_similarity:.4f}")

        for idx, (query, sim) in enumerate(zip(positive_queries, pos_sims), start=1):
            print(f"    Positive Query [{idx}]: '{query}' | Combined Similarity: {sim:.4f}")

        if negative_queries:
            for idx, (query, sim) in enumerate(zip(negative_queries, neg_sims), start=1):
                print(f"    Negative Query [{idx}]: '{query}' | Combined Similarity: {sim:.4f}")

        print(f"  Created: {creation_time}")
        print(f"  Modified: {modification_time}")
        print("-" * 60)
    except Exception as e:
        logging.error(f"Error retrieving information for '{image_path}': {e}")


def output_filenames(matched_images: List[MatchedImageResult]) -> None:
    """
    Outputs only the file names of matched images.
    """
    if not matched_images:
        logging.info("No images matched the search criteria.")
        return
    for result in matched_images:
        print(result.image_path)


def main() -> None:
    """
    The main function orchestrating the search process.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    # Verify source directory
    if not os.path.isdir(args.source_directory):
        logging.error(f"Source directory '{args.source_directory}' does not exist.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.debug(f"Using device: {device}")

    # Get images
    image_paths = get_image_paths(args.source_directory, args.recursive)

    # Load models
    blip_processor, blip_model = load_blip_model(device)
    st_model = load_sentence_model()
    clip_model, clip_processor = load_clip_model(device)

    # Generate captions
    captions_list = generate_captions(image_paths, blip_processor, blip_model, device)

    # Compute similarities
    matched_images = compute_similarity(
        captions_list=captions_list,
        queries=args.query if args.query else [],
        negative_queries=args.negative_query if args.negative_query else [],
        pos_threshold=args.positive_threshold,
        neg_threshold=args.negative_threshold,
        caption_weight=args.caption_weight,
        clip_weight=args.clip_weight,
        st_model=st_model,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device
    )

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
