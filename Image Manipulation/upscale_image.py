#!/usr/bin/env python3

# -------------------------------------------------------
# Script: upscale_image.py
#
# Description:
# This script generates a high-resolution image by starting
# with an initial image and iteratively upscaling it.
# If no initial image is provided, it generates one using Stable
# Diffusion XL (SDXL) based on a given prompt.
#
# The process involves upscaling the initial image using Real-ESRGAN,
# splitting it into tiles of random sizes between 768 and 1024 pixels,
# and performing hallucination (inpainting) on each tile.
# This ensures complete coverage of the image without gaps or overlaps.
#
# Usage:
# ./upscale_image.py [options]
#
# Options:
#   -i, --input              Path to the input image to be upscaled.
#                            If not provided, an initial image will be generated using SDXL.
#
#   SDXL Generation Options  (used only if --input is not specified):
#     -p, --prompt           The prompt to use for SDXL image generation (default: "A beautiful landscape").
#     -s, --seed             Random seed for reproducibility (default: None).
#     -n, --num-steps        Number of inference steps for the SDXL model (default: 50).
#     -g, --guidance-scale   Guidance scale for the SDXL model (default: 7.5).
#     -T, --txt2img-checkpoint
#                            SDXL checkpoint to use for text-to-image generation (default: "stabilityai/stable-diffusion-xl-base-1.0").
#     -C, --inpaint-checkpoint
#                            SDXL checkpoint to use for inpainting tasks (default: "stabilityai/stable-diffusion-2-inpainting").
#
#   Real-ESRGAN Upscaling Options:
#     -u, --upscale-exponent Number of upscaling levels. Controls how many times the hallucination and upscaling steps are repeated.
#                            (default: 1).
#     -f, --upscale-factor   Upscale factor for Real-ESRGAN (default: 4). Allowed values are 2 and 4.
#
#   Hallucination Options:
#     -H, --hallucinate          Enable hallucination feature to modify tiles using SDXL inpainting.
#     -t, --strength             Strength of the inpainting effect (between 0.0 and 1.0). Lower values make the inpainting
#                                closer to the original image. (default: 0.3).
#     -e, --min-area             Minimum area (in pixels) for a superpixel to be considered for hallucination (default: 1000).
#     -b, --border-thresh        Threshold (in pixels) to determine if a superpixel is near the image border (default: 50).
#     -S, --hallucination-steps  Number of times to perform object detection and inpainting per tile.
#                                Higher values result in more hallucinations. (default: 1).
#
#   General Options:
#     -d, --output-dir       Directory to save the output images (default: "output").
#     -v, --verbose          Enable verbose logging (DEBUG level).
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
# - Pillow (install via: pip install Pillow)
# - Torch & Torchvision (install via: pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118)
# - numpy (install via: pip install numpy==1.26.4)
# - diffusers (install via: pip install diffusers)
# - requests (install via: pip install requests)
# - tqdm (install via: pip install tqdm)
# - basicsr (install via: pip install basicsr)
# - realesrgan (install via: pip install realesrgan)
# - scikit-image (required for superpixel segmentation) (install via: pip install scikit-image)
# - OpenCV (install via: apt install python3-opencv opencv-data)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import gc
from typing import Optional, List, Tuple
from pathlib import Path

import warnings
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import cv2

from diffusers import StableDiffusionXLPipeline, StableDiffusionInpaintPipeline
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from skimage.segmentation import slic
from skimage.util import img_as_float

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Constants for tile sizes
INITIAL_IMAGE_SIZE = 1024
MIN_TILE_SIZE = 768
MAX_TILE_SIZE = 1024

# Default Real-ESRGAN model URLs
DEFAULT_REALESRGAN_MODEL_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

# Determine the device to use (GPU if available, else CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define cache directory for models
CACHE_DIR = Path.home() / ".cache" / "upscale_image"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REALESRGAN_CACHE_DIR = CACHE_DIR / "realesrgan"
REALESRGAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generates a high-resolution image by upscaling an initial image using Real-ESRGAN, "
            "splitting it into tiles of random sizes, performing hallucination, and optionally iterating this process "
            "to achieve higher resolutions."
        )
    )

    # Input Image Option
    parser.add_argument(
        '-i', '--input', type=str, default=None,
        help='Path to the input image to be upscaled. If not provided, an initial image will be generated using SDXL.'
    )

    # SDXL Generation Options (only if --input is not specified)
    parser.add_argument(
        '-p', '--prompt', type=str, default='A beautiful landscape',
        help='The prompt to use for SDXL image generation.'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=None,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '-n', '--num-steps', type=int, default=50,
        help='Number of inference steps for the SDXL model.'
    )
    parser.add_argument(
        '-g', '--guidance-scale', type=float, default=7.5,
        help='Guidance scale for the SDXL model.'
    )
    parser.add_argument(
        '-T', '--txt2img-checkpoint', type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='SDXL checkpoint to use for text-to-image generation.'
    )
    parser.add_argument(
        '-C', '--inpaint-checkpoint', type=str,
        default='stabilityai/stable-diffusion-2-inpainting',
        help='Stable Diffusion inpainting checkpoint to use for inpainting tasks.'
    )

    # Real-ESRGAN Upscaling Options
    parser.add_argument(
        '-u', '--upscale-exponent', type=int, default=1,
        help='Number of upscaling levels. Controls how many times the hallucination and upscaling steps are repeated.'
    )
    parser.add_argument(
        '-f', '--upscale-factor', type=int, default=4,
        help='Upscale factor for Real-ESRGAN (default: 4). Allowed values are 2 and 4.'
    )

    # Hallucination Option
    parser.add_argument(
        '-H', '--hallucinate', action='store_true',
        help='Enable hallucination feature to modify tiles using SDXL inpainting.'
    )
    parser.add_argument(
        '-t', '--strength', type=float, default=0.3,
        help='Strength of the inpainting effect (between 0.0 and 1.0). Lower values make the inpainting closer to the original image.'
    )
    parser.add_argument(
        '-e', '--min-area', type=int, default=1000,
        help='Minimum area (in pixels) for a superpixel to be considered for hallucination.'
    )
    parser.add_argument(
        '-b', '--border-thresh', type=int, default=50,
        help='Threshold (in pixels) to determine if a superpixel is near the image border.'
    )
    parser.add_argument(
        '-S', '--hallucination-steps', type=int, default=1,
        help='Number of times to perform object detection and inpainting per tile. Higher values result in more hallucinations. (default: 1).'
    )

    # General Options
    parser.add_argument(
        '-d', '--output-dir', type=str, default='output',
        help='Directory to save the output images.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging (DEBUG level).'
    )

    args = parser.parse_args()

    # Validate upscale_exponent to prevent excessively large images
    MAX_UPSCALE_EXPONENT = 5
    if args.upscale_exponent < 1:
        parser.error("Argument --upscale-exponent must be at least 1.")
    if args.upscale_exponent > MAX_UPSCALE_EXPONENT:
        parser.error(f"Argument --upscale-exponent must not exceed {MAX_UPSCALE_EXPONENT} to prevent memory issues.")

    # Validate hallucination_steps
    if args.hallucination_steps < 1:
        parser.error("Argument --hallucination-steps must be at least 1.")

    # If input image is provided, ensure that it exists
    if args.input is not None and not os.path.isfile(args.input):
        parser.error(f"Input image '{args.input}' does not exist.")

    # Validate strength parameter
    if not 0.0 <= args.strength <= 1.0:
        parser.error("Argument --strength must be between 0.0 and 1.0.")

    # Validate min_area and border_thresh
    if args.min_area < 0:
        parser.error("Argument --min-area must be non-negative.")
    if args.border_thresh < 0:
        parser.error("Argument --border-thresh must be non-negative.")

    # Validate upscale_factor
    if args.upscale_factor not in [2, 4]:
        parser.error("Argument --upscale-factor must be either 2 or 4.")

    return args


def setup_logging(verbose: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def download_realesrgan_model(upscale_factor: int) -> Path:
    """
    Downloads the Real-ESRGAN model weights from the default URL if not present.
    Stores the model in the cache directory.
    Returns the path to the downloaded model.
    """
    try:
        url = DEFAULT_REALESRGAN_MODEL_URLS.get(upscale_factor)
        if url is None:
            logging.error(f"No default Real-ESRGAN model URL for upscale factor {upscale_factor}.")
            sys.exit(1)

        model_filename = os.path.basename(url)
        model_path = REALESRGAN_CACHE_DIR / model_filename

        if model_path.is_file():
            logging.info(f"Real-ESRGAN model already exists at '{model_path}'. Skipping download.")
            return model_path

        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f, tqdm(
            desc=f"Downloading Real-ESRGAN model to '{model_path}'",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        logging.info(f"Successfully downloaded Real-ESRGAN model to '{model_path}'.")
        return model_path
    except Exception as e:
        logging.error(f"Failed to download Real-ESRGAN model from '{url}'. Error: {e}")
        sys.exit(1)


def load_sdxl_pipeline(checkpoint: str) -> StableDiffusionXLPipeline:
    """
    Loads the SDXL pipeline for image generation.
    """
    logging.info(f"Loading SDXL pipeline from checkpoint '{checkpoint}'. This may take a while...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        pipe = pipe.to(DEVICE)
        pipe.safety_checker = None  # Disable safety checker for faster inference
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention for SDXL.")
        except Exception as e:
            logging.warning(f"Failed to enable xformers memory efficient attention for SDXL: {e}")
        logging.info("SDXL pipeline loaded successfully.")
        return pipe
    except Exception as e:
        logging.exception(f"Failed to load SDXL pipeline from checkpoint '{checkpoint}'.")
        sys.exit(1)


def load_inpaint_pipeline(checkpoint: str) -> StableDiffusionInpaintPipeline:
    """
    Loads the Stable Diffusion Inpainting pipeline for hallucination.
    """
    logging.info(f"Loading Stable Diffusion Inpainting pipeline from checkpoint '{checkpoint}'. This may take a while...")
    try:
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        inpaint_pipe = inpaint_pipe.to(DEVICE)
        inpaint_pipe.safety_checker = None  # Disable safety checker for faster inference
        inpaint_pipe.enable_attention_slicing()
        try:
            inpaint_pipe.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention for Stable Diffusion Inpainting.")
        except Exception as e:
            logging.warning(f"Failed to enable xformers memory efficient attention for Stable Diffusion Inpainting: {e}")
        logging.info("Stable Diffusion Inpainting pipeline loaded successfully.")
        return inpaint_pipe
    except Exception as e:
        logging.exception(f"Failed to load Stable Diffusion Inpainting pipeline from checkpoint '{checkpoint}'.")
        sys.exit(1)


def generate_initial_image(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generates the initial image using the SDXL text-to-image pipeline.
    """
    try:
        generator = torch.Generator(device=DEVICE)
        if seed is not None:
            generator.manual_seed(seed)
            logging.info(f"Using seed: {seed}")
        else:
            seed = generator.seed()
            logging.info(f"No seed provided. Using random seed: {seed}")

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
            )

        if not result.images:
            logging.error("SDXL pipeline did not return any images.")
            sys.exit(1)

        image = result.images[0]
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logging.info("Initial image generated using SDXL.")
        return image
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA Out of Memory Error: Failed to generate image with SDXL due to insufficient GPU memory.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Failed to generate initial image with SDXL.")
        sys.exit(1)


def load_realesrgan_model(upscale_factor: int) -> RealESRGANer:
    """
    Loads the Real-ESRGAN model based on the upscale factor.
    Automatically downloads the model if it's not present in the cache.
    """
    logging.info(f"Loading Real-ESRGAN model for upscale factor {upscale_factor}.")
    try:
        model_path = download_realesrgan_model(upscale_factor)

        model_path = model_path.resolve()

        if upscale_factor == 2:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2
            )
        elif upscale_factor == 4:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
        else:
            logging.error(f"Unsupported upscale factor {upscale_factor}.")
            sys.exit(1)

        realesrgan_model = RealESRGANer(
            scale=upscale_factor,
            model_path=str(model_path),
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=10,
            half=True if DEVICE.type == "cuda" else False,
            device=DEVICE
        )

        logging.info(f"Real-ESRGAN model loaded successfully on {DEVICE}.")
        return realesrgan_model
    except Exception as e:
        logging.exception(f"Failed to load Real-ESRGAN model from '{model_path}'.")
        sys.exit(1)


def align_to_multiple(size: int, multiple: int = 8) -> int:
    """
    Aligns the given size down to the nearest multiple of the specified value.
    """
    return size - (size % multiple)


def generate_tile_sizes(total_size: int, min_size: int, max_size: int, multiple: int = 8) -> List[int]:
    """
    Generates a list of tile sizes for one dimension (width or height).
    Each size is between min_size and max_size and divisible by 'multiple'.
    Ensures that the sum of tile sizes equals total_size.
    """
    sizes = []
    remaining = total_size

    while remaining > 0:
        if remaining <= max_size:
            size = align_to_multiple(remaining, multiple)
            if size < min_size and sizes:
                # Adjust the last tile to accommodate the remaining size
                sizes[-1] += remaining
                sizes[-1] = align_to_multiple(sizes[-1], multiple)
            else:
                sizes.append(size)
            break
        else:
            size = np.random.randint(min_size, max_size + 1)
            size = align_to_multiple(size, multiple)
            # Ensure that the remaining size after this tile is not less than min_size
            if remaining - size < min_size:
                size = remaining - min_size
                size = align_to_multiple(size, multiple)
            sizes.append(size)
            remaining -= size

    # Final adjustment to ensure the sum matches total_size
    total_generated = sum(sizes)
    if total_generated < total_size:
        adjustment = total_size - total_generated
        if adjustment % multiple == 0:
            sizes[-1] += adjustment
        else:
            # Adjust to the nearest multiple of 'multiple'
            sizes[-1] += adjustment - (adjustment % multiple)
    elif total_generated > total_size:
        adjustment = total_generated - total_size
        if adjustment % multiple == 0:
            sizes[-1] -= adjustment
        else:
            # Adjust to the nearest multiple of 'multiple'
            sizes[-1] -= adjustment + (multiple - (adjustment % multiple))

    # Final check to ensure all sizes are within constraints and divisible by 'multiple'
    for i, size in enumerate(sizes):
        if size < min_size or size > max_size:
            logging.warning(f"Generated tile size {size} is out of bounds ({min_size}-{max_size}). Adjusting.")
            size = min(max(size, min_size), max_size)
        if size % multiple != 0:
            logging.warning(f"Generated tile size {size} is not divisible by {multiple}. Aligning.")
            size = align_to_multiple(size, multiple)
        # Update the size in the list
        sizes[i] = size

    return sizes


def split_image_into_tiles(image: Image.Image, min_tile_size: int = MIN_TILE_SIZE, max_tile_size: int = MAX_TILE_SIZE) -> List[Tuple[Image.Image, int, int]]:
    """
    Splits the image into tiles of random sizes between min_tile_size and max_tile_size pixels.
    Ensures that all parts of the image are covered without gaps or overlaps.
    Returns a list of tuples (tile, x, y), where x and y are the positions of the tile in the image.
    """
    width, height = image.size
    tiles = []

    # Generate tile sizes for both dimensions
    tile_widths = generate_tile_sizes(width, min_tile_size, max_tile_size)
    tile_heights = generate_tile_sizes(height, min_tile_size, max_tile_size)

    # Calculate x and y positions based on tile sizes
    x_positions = [0]
    for w in tile_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    y_positions = [0]
    for h in tile_heights[:-1]:
        y_positions.append(y_positions[-1] + h)

    # Iterate over all tile positions and sizes
    for y, h in zip(y_positions, tile_heights):
        for x, w in zip(x_positions, tile_widths):
            box = (x, y, x + w, y + h)
            tile = image.crop(box)
            tiles.append((tile, x, y))
            logging.debug(f"Created tile at ({x}, {y}) with size ({w}, {h}).")

    logging.info(f"Image split into {len(tiles)} tiles with varying sizes.")
    return tiles


def upscale_with_realesrgan(model: RealESRGANer, image: Image.Image, outscale: float) -> Image.Image:
    """
    Upscales the given image using the Real-ESRGAN model.
    Converts PIL Image to NumPy array before processing and back after.
    """
    try:
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        upscaled_bgr, _ = model.enhance(img_bgr, outscale=outscale)
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        upscaled_image = Image.fromarray(upscaled_rgb)
        return upscaled_image
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA Out of Memory Error: Failed to upscale image with Real-ESRGAN due to insufficient GPU memory.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Failed to upscale image with Real-ESRGAN.")
        sys.exit(1)


def stitch_tiles(tiles_with_positions: List[Tuple[Image.Image, int, int]], image_size: Tuple[int, int]) -> Image.Image:
    """
    Stitches the tiles back into a single image without overlapping.
    tiles_with_positions is a list of tuples (tile, x, y).
    image_size is (width, height) of the final image.
    """
    final_image = Image.new('RGB', image_size)
    for tile, x, y in tiles_with_positions:
        final_image.paste(tile, (x, y))
    logging.debug(f"Tiles stitched into image of size {image_size[0]}x{image_size[1]}.")
    return final_image


def create_superpixel_mask(image_np: np.ndarray, n_segments: int = 50, compactness: float = 10.0,
                           min_area: int = 1000, border_thresh: int = 50,
                           excluded_segments: Optional[List[int]] = None) -> Optional[Image.Image]:
    """
    Creates a mask using SLIC superpixel segmentation.
    Excludes superpixels that are too small, near the image borders, or in the excluded_segments list.
    Selects the largest suitable superpixel for inpainting.
    """
    try:
        if excluded_segments is None:
            excluded_segments = []

        # Convert image to float for SLIC
        image_float = img_as_float(image_np)
        segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)

        # Get image dimensions
        height, width = segments.shape

        # Calculate area of each segment
        unique, counts = np.unique(segments, return_counts=True)

        # Filter segments based on min_area, border_thresh, and excluded_segments
        suitable_segments = []
        for seg_id, area in zip(unique, counts):
            if area < min_area:
                logging.debug(f"Superpixel {seg_id} skipped: area {area} is smaller than min_area {min_area}.")
                continue

            if seg_id in excluded_segments:
                logging.debug(f"Superpixel {seg_id} skipped: already in excluded_segments.")
                continue

            # Find bounding box of the segment
            ys, xs = np.where(segments == seg_id)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            # Check distance from borders
            if y_min < border_thresh or y_max > (height - border_thresh) or \
               x_min < border_thresh or x_max > (width - border_thresh):
                logging.debug(f"Superpixel {seg_id} skipped: too close to image borders.")
                continue

            suitable_segments.append((seg_id, area))

        if not suitable_segments:
            logging.debug("No suitable superpixels found for masking.")
            return None

        # Select the largest suitable superpixel
        suitable_segments.sort(key=lambda x: x[1], reverse=True)
        selected_seg_id = suitable_segments[0][0]
        logging.debug(f"Selected superpixel {selected_seg_id} with area {suitable_segments[0][1]} for masking.")

        # Create mask for the selected superpixel
        mask = np.zeros(segments.shape, dtype=np.uint8)
        mask[segments == selected_seg_id] = 255

        mask_image = Image.fromarray(mask).convert("L")
        logging.debug("Superpixel mask created successfully.")
        return mask_image

    except Exception:
        logging.exception("Failed to create superpixel mask.")
        return None


def perform_hallucination(
    tile: Image.Image,
    inpaint_pipe: StableDiffusionInpaintPipeline,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    strength: float,
    min_area: int,
    border_thresh: int,
    hallucination_steps: int,
    x: int,
    y: int
) -> Image.Image:
    """
    Performs hallucination on a given tile using superpixel segmentation for the specified number of steps.
    Each step creates a new mask on a different region of the tile and applies inpainting.
    """
    try:
        logging.debug(f"Processing tile at position ({x}, {y}). Tile size: {tile.size}")
        current_tile = tile.copy()
        upscaled_tile_np = np.array(current_tile)

        excluded_segments = []

        for step in range(1, hallucination_steps + 1):
            logging.debug(f"Hallucination step {step}/{hallucination_steps} for tile at position ({x}, {y}).")
            mask = create_superpixel_mask(
                image_np=upscaled_tile_np,
                n_segments=50,
                compactness=10.0,
                min_area=min_area,
                border_thresh=border_thresh,
                excluded_segments=excluded_segments
            )

            if mask is not None:
                inpainted_tile = inpaint_tile(
                    tile=current_tile,
                    mask=mask,
                    inpaint_pipe=inpaint_pipe,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                )
                current_tile = inpainted_tile
                upscaled_tile_np = np.array(current_tile)

                # Update excluded_segments to avoid modifying the same area again
                # Extract the segment ID from the mask
                segments = slic(img_as_float(upscaled_tile_np), n_segments=50, compactness=10.0, start_label=1)
                selected_segments = np.unique(segments[np.array(mask) == 255])
                excluded_segments.extend(selected_segments.tolist())
                logging.debug(f"Excluded segments after step {step}: {excluded_segments}")
            else:
                logging.debug(f"No suitable superpixel mask generated for hallucination step {step} on tile at position ({x}, {y}). Skipping this step.")
                break  # No more suitable segments to process

        return current_tile

    except Exception as e:
        logging.exception("Failed during hallucination process.")
        sys.exit(1)


def inpaint_tile(
    tile: Image.Image,
    mask: Image.Image,
    inpaint_pipe: StableDiffusionInpaintPipeline,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    strength: float,
) -> Image.Image:
    """
    Performs inpainting on the given tile using the provided mask.
    """
    try:
        logging.debug(
            f"Starting inpainting. Tile size: {tile.size}, Mask size: {mask.size}"
        )
        # Ensure that tile dimensions are divisible by 8
        tile_width, tile_height = tile.size
        tile_width_aligned = align_to_multiple(tile_width, 8)
        tile_height_aligned = align_to_multiple(tile_height, 8)
        if tile_width != tile_width_aligned or tile_height != tile_height_aligned:
            logging.debug(f"Aligning tile size from ({tile_width}, {tile_height}) to ({tile_width_aligned}, {tile_height_aligned}).")
            tile = tile.crop((0, 0, tile_width_aligned, tile_height_aligned))
            mask = mask.crop((0, 0, tile_width_aligned, tile_height_aligned))

        result = inpaint_pipe(
            prompt=prompt,
            image=tile,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width=tile.width,
            height=tile.height,
        )
        inpainted_tile = result.images[0]
        logging.debug(f"Inpainting completed. Inpainted tile size: {inpainted_tile.size}")
        return inpainted_tile
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA Out of Memory Error: Failed to perform inpainting due to insufficient GPU memory.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Failed to perform inpainting on tile.")
        sys.exit(1)


def upscale_tile(
    model: RealESRGANer,
    tile: Image.Image,
    outscale: float
) -> Image.Image:
    """
    Upscales a single tile using the Real-ESRGAN model.
    """
    upscaled_tile = upscale_with_realesrgan(model, tile, outscale=outscale)
    return upscaled_tile


def create_high_resolution_image(
    initial_image: Image.Image,
    upscale_exponent: int,
    output_dir: str,
    model: RealESRGANer,
    upscale_factor: int,
    hallucinate: bool = False,
    prompt: str = "",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    strength: float = 0.8,
    inpaint_checkpoint: str = '',
    min_area: int = 1000,
    border_thresh: int = 50,
    hallucination_steps: int = 1
) -> None:
    """
    Generates the final high-resolution image by recursively upscaling and hallucinating the initial image.
    """
    logging.info("Starting to generate the high-resolution image.")

    initial_image_path = os.path.join(output_dir, f'initial_image.png')
    initial_image.save(initial_image_path)
    logging.info(f"Initial image saved to '{initial_image_path}'.")

    # Upscale initial image
    logging.info(f"Upscaling initial image with upscale factor {upscale_factor}.")
    current_image = initial_image

    # Initialize hallucination resources if needed
    inpaint_pipe = None
    if hallucinate:
        inpaint_pipe = load_inpaint_pipeline(checkpoint=inpaint_checkpoint)

    for level in range(1, upscale_exponent + 1):
        logging.info(f"Starting level {level}/{upscale_exponent}.")

        # Split image into tiles with random sizes
        tiles_with_positions = split_image_into_tiles(
            current_image,
            min_tile_size=MIN_TILE_SIZE,
            max_tile_size=MAX_TILE_SIZE
        )
        num_tiles = len(tiles_with_positions)
        logging.info(f"Level {level}: Image split into {num_tiles} tiles.")

        processed_tiles = []
        for idx, (tile, x, y) in enumerate(tiles_with_positions):
            logging.debug(f"Processing tile {idx + 1}/{num_tiles} at position ({x}, {y}).")

            if hallucinate and inpaint_pipe:
                tile = perform_hallucination(
                    tile=tile,
                    inpaint_pipe=inpaint_pipe,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    min_area=min_area,
                    border_thresh=border_thresh,
                    hallucination_steps=hallucination_steps,
                    x=x,
                    y=y
                )

            # Upscale the tile
            tile = upscale_with_realesrgan(model, tile, outscale=upscale_factor)

            # Adjust the tile position in the upscaled image
            new_x = x * upscale_factor
            new_y = y * upscale_factor

            processed_tiles.append((tile, new_x, new_y))
            cleanup_memory()

        # Calculate the new image size
        new_width = current_image.width * upscale_factor
        new_height = current_image.height * upscale_factor
        image_size = (new_width, new_height)
        logging.debug(f"Level {level}: New image size will be {image_size}.")

        # Stitch the processed tiles back together
        logging.info(f"Level {level}: Stitching tiles to form a new image of size {new_width}x{new_height}.")
        current_image = stitch_tiles(processed_tiles, image_size=image_size)

        # Save the image at this level
        level_image_path = os.path.join(output_dir, f'level{level}_image.png')
        current_image.save(level_image_path)
        logging.info(f"Level {level}: Image saved to '{level_image_path}'.")

    # Clean up hallucination resources
    if hallucinate:
        del inpaint_pipe
        cleanup_memory()

    # Save final image
    final_image_path = os.path.join(output_dir, 'final_image.png')
    current_image.save(final_image_path)
    logging.info(f"Final image saved as '{final_image_path}'.")

    cleanup_memory()


def cleanup_memory() -> None:
    """
    Frees up memory by deleting unused objects and clearing caches.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.debug("Freed memory after cleanup.")


def create_initial_image_if_needed(args: argparse.Namespace) -> Image.Image:
    """
    Generates an initial image using SDXL if no input image is provided.
    Otherwise, loads the provided input image.
    """
    if args.input is not None:
        try:
            initial_image = Image.open(args.input).convert('RGB')
            logging.info(f"Loaded input image from '{args.input}'.")
            return initial_image
        except Exception as e:
            logging.exception(f"Failed to open input image '{args.input}'.")
            sys.exit(1)
    else:
        # Generate initial image using SDXL
        sdxl_pipe = load_sdxl_pipeline(checkpoint=args.txt2img_checkpoint)
        initial_image = generate_initial_image(
            pipe=sdxl_pipe,
            prompt=args.prompt,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            width=INITIAL_IMAGE_SIZE,
            height=INITIAL_IMAGE_SIZE,
            seed=args.seed,
        )
        return initial_image


def main() -> None:
    """
    Main function to orchestrate the image upscaling process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    logging.info(f"Using device: {DEVICE}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory set to '{args.output_dir}'.")

    # Generate or load the initial image
    initial_image = create_initial_image_if_needed(args)

    # Load Real-ESRGAN model
    realesrgan_model = load_realesrgan_model(upscale_factor=args.upscale_factor)

    # Generate the high-resolution image
    create_high_resolution_image(
        initial_image=initial_image,
        upscale_exponent=args.upscale_exponent,
        output_dir=args.output_dir,
        model=realesrgan_model,
        upscale_factor=args.upscale_factor,
        hallucinate=args.hallucinate,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        strength=args.strength,
        inpaint_checkpoint=args.inpaint_checkpoint,
        min_area=args.min_area,
        border_thresh=args.border_thresh,
        hallucination_steps=args.hallucination_steps
    )

    logging.info("Image upscaling process completed successfully.")
    sys.exit(0)


if __name__ == '__main__':
    main()
