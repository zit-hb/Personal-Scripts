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
# Each upscale step involves splitting the image into fixed-size tiles,
# upscaling each tile to a higher resolution using Real-ESRGAN, and then
# stitching the upscaled tiles back together.
#
# Optionally, a "hallucination" feature can be enabled to modify the upscaled tiles.
# This involves selecting a region within a tile using superpixel segmentation,
# creating a mask, and then using SDXL Inpainting to slightly alter the region and add more details.
# The number of times object detection and inpainting are performed per tile can be controlled
# using the "hallucination-steps" option.
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
#     -u, --upscale-exponent Number of upscaling steps. Each step upscales the image by a factor of 4
#                            (quadrupling width and height). (default: 2).
#     -m, --model            Path to the Real-ESRGAN model weights file. (default: "RealESRGAN_x4plus.pth").
#
#   Hallucination Options:
#     -H, --hallucinate          Enable hallucination feature to modify upscaled tiles using SDXL inpainting.
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
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - Pillow (install via: pip install Pillow)
# - torch (install via: pip install torch)
# - numpy (install via: pip install numpy)
# - diffusers (install via: pip install diffusers)
# - requests (install via: pip install requests)
# - tqdm (install via: pip install tqdm)
# - opencv-python (install via: pip install opencv-python)
# - basicsr (install via: pip install basicsr)
# - realesrgan (install via: pip install realesrgan)
# - scikit-image (required for superpixel segmentation) (install via: pip install scikit-image)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import gc
import math
from typing import Optional, List

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
FIXED_TILE_SIZE = 512          # Size before upscaling
UPSCALED_TILE_SIZE = 2048      # Size after upscaling

# Default Real-ESRGAN model URL
DEFAULT_REALESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Determine the device to use (GPU if available, else CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generates a high-resolution image by upscaling an initial image recursively using Real-ESRGAN. "
            "If no initial image is provided, generates one using Stable Diffusion XL (SDXL) based on a given prompt."
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
        '-u', '--upscale-exponent', type=int, default=2,
        help='Number of upscaling steps. Each step upscales the image by a factor of 4 (quadrupling width and height).'
    )
    parser.add_argument(
        '-m', '--model', type=str, default='RealESRGAN_x4plus.pth',
        help='Path to the Real-ESRGAN model weights file.'
    )

    # Hallucination Option
    parser.add_argument(
        '-H', '--hallucinate', action='store_true',
        help='Enable hallucination feature to modify upscaled tiles using SDXL inpainting.'
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
        help='Number of times to perform object detection and inpainting per tile. (default: 1).'
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

    # Validate Real-ESRGAN model path or download if not present
    if not os.path.isfile(args.model):
        setup_logging(verbose=args.verbose)  # Initialize logging to capture download logs
        logging.info(f"Real-ESRGAN model file '{args.model}' not found. Attempting to download...")
        download_realesrgan_model(args.model)
        if not os.path.isfile(args.model):
            parser.error(f"Failed to download Real-ESRGAN model to '{args.model}'. Please provide a valid Real-ESRGAN model weights file.")

    # Validate strength parameter
    if not 0.0 <= args.strength <= 1.0:
        parser.error("Argument --strength must be between 0.0 and 1.0.")

    # Validate min_area and border_thresh
    if args.min_area < 0:
        parser.error("Argument --min-area must be non-negative.")
    if args.border_thresh < 0:
        parser.error("Argument --border-thresh must be non-negative.")

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


def download_realesrgan_model(model_path: str) -> None:
    """
    Downloads the Real-ESRGAN model weights from the default URL if not present.
    """
    try:
        response = requests.get(DEFAULT_REALESRGAN_MODEL_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        if os.path.dirname(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
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
    except Exception as e:
        logging.error(f"Failed to download Real-ESRGAN model from '{DEFAULT_REALESRGAN_MODEL_URL}'. Error: {e}")
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


def load_realesrgan_model(model_path: str) -> RealESRGANer:
    """
    Loads the Real-ESRGAN model from the specified weights file using the realesrgan package.
    """
    logging.info(f"Loading Real-ESRGAN model from '{model_path}'. This may take a while...")
    try:
        model_path = os.path.abspath(model_path)

        if not os.path.isfile(model_path):
            logging.error(f"Model file '{model_path}' does not exist.")
            sys.exit(1)

        rrdbnet = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )

        model = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=rrdbnet,
            tile=0,
            tile_pad=10,
            pre_pad=10,
            half=True if DEVICE.type == "cuda" else False,
            device=DEVICE
        )

        logging.info(f"Real-ESRGAN model loaded successfully on {DEVICE}.")
        return model
    except Exception as e:
        logging.exception(f"Failed to load Real-ESRGAN model from '{model_path}'.")
        sys.exit(1)


def split_image_into_tiles(image: Image.Image, tile_size: int = FIXED_TILE_SIZE) -> List[Image.Image]:
    """
    Splits the image into tiles of specified size without overlapping.
    """
    width, height = image.size
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tile = image.crop(box)
            if tile.size != (tile_size, tile_size):
                tile = tile.resize((tile_size, tile_size), resample=Image.BICUBIC)
            tiles.append(tile)
    logging.debug(f"Image split into {len(tiles)} tiles ({math.ceil(height / tile_size)} rows x {math.ceil(width / tile_size)} cols).")
    return tiles


def upscale_with_realesrgan(model: RealESRGANer, image: Image.Image) -> Image.Image:
    """
    Upscales the given image using the Real-ESRGAN model.
    Converts PIL Image to NumPy array before processing and back after.
    """
    try:
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        upscaled_bgr, _ = model.enhance(img_bgr, outscale=4)
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        upscaled_image = Image.fromarray(upscaled_rgb)
        return upscaled_image
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA Out of Memory Error: Failed to upscale image with Real-ESRGAN due to insufficient GPU memory.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Failed to upscale image with Real-ESRGAN.")
        sys.exit(1)


def stitch_tiles(tiles: List[Image.Image], rows: int, cols: int, tile_size: int = UPSCALED_TILE_SIZE) -> Image.Image:
    """
    Stitches the tiles back into a single image without overlapping.
    """
    if not tiles:
        logging.error("No tiles to stitch.")
        sys.exit(1)

    final_width = tile_size * cols
    final_height = tile_size * rows
    final_image = Image.new('RGB', (final_width, final_height))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < len(tiles):
                tile = tiles[idx]
                final_image.paste(tile, (col * tile_size, row * tile_size))
                idx += 1
    logging.debug(f"Tiles stitched into image of size {final_width}x{final_height}.")
    return final_image


def align_edges(image: Image.Image, tile_size: int = UPSCALED_TILE_SIZE) -> Image.Image:
    """
    Post-processes the stitched image to ensure edge alignment without blending or blurring.
    Placeholder for edge alignment logic.
    """
    # Implement edge correction algorithms if necessary
    # Currently returns the image as-is
    return image


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
        segment_areas = dict(zip(unique, counts))

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

    except Exception as e:
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
    row_idx: int,
    col_idx: int
) -> Image.Image:
    """
    Performs hallucination on a given tile using superpixel segmentation for the specified number of steps.
    Each step creates a new mask on a different region of the tile and applies inpainting.
    """
    try:
        logging.debug(f"Processing tile at Row: {row_idx}, Column: {col_idx}. Tile size: {tile.size}")
        current_tile = tile.copy()
        upscaled_tile_np = np.array(current_tile)

        excluded_segments = []

        for step in range(1, hallucination_steps + 1):
            logging.debug(f"Hallucination step {step}/{hallucination_steps} for tile at Row: {row_idx}, Column: {col_idx}.")
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
                selected_segments = np.unique(segments[mask == 255])
                excluded_segments.extend(selected_segments.tolist())
                logging.debug(f"Excluded segments after step {step}: {excluded_segments}")
            else:
                logging.debug(f"No suitable superpixel mask generated for hallucination step {step} on tile at Row: {row_idx}, Column: {col_idx}. Skipping this step.")
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
        result = inpaint_pipe(
            prompt=prompt,
            image=tile,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width=FIXED_TILE_SIZE,
            height=FIXED_TILE_SIZE,
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
    tile: Image.Image
) -> Image.Image:
    """
    Upscales a single tile using the Real-ESRGAN model.
    """
    upscaled_tile = upscale_with_realesrgan(model, tile)
    return upscaled_tile


def create_high_resolution_image(
    initial_image: Image.Image,
    upscale_exponent: int,
    output_dir: str,
    model: RealESRGANer,
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
    Generates the final high-resolution image by recursively upscaling the initial image.
    """
    logging.info("Starting to generate the seamless high-resolution image.")

    # Save initial image
    initial_image_path_out = os.path.join(output_dir, 'initial_image.png')
    initial_image.save(initial_image_path_out)
    logging.info(f"Initial image saved to '{initial_image_path_out}'.")

    current_image = initial_image

    # Initialize hallucination resources if needed
    inpaint_pipe = None
    if hallucinate:
        inpaint_pipe = load_inpaint_pipeline(checkpoint=inpaint_checkpoint)

    for step in range(1, upscale_exponent + 1):
        logging.info(f"Starting upscale step {step}/{upscale_exponent}.")

        tiles = split_image_into_tiles(current_image, tile_size=FIXED_TILE_SIZE)
        num_tiles = len(tiles)
        rows = math.ceil(current_image.height / FIXED_TILE_SIZE)
        cols = math.ceil(current_image.width / FIXED_TILE_SIZE)

        upscaled_tiles = []
        for idx, tile in enumerate(tiles):
            row_idx = idx // cols
            col_idx = idx % cols
            logging.debug(f"Upscaling tile {idx + 1}/{num_tiles} at Row: {row_idx}, Column: {col_idx}.")

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
                    row_idx=row_idx,
                    col_idx=col_idx
                )

            upscaled_tile = upscale_tile(model, tile)

            intermediate_tile_path = os.path.join(
                output_dir, f'step{step}_tile_row{row_idx}_col{col_idx}.png'
            )
            upscaled_tile.save(intermediate_tile_path)
            logging.debug(f"Upscaled tile saved to '{intermediate_tile_path}'.")

            upscaled_tiles.append(upscaled_tile)
            cleanup_memory()

        # Stitch the upscaled tiles back together
        new_width = current_image.width * 4
        new_height = current_image.height * 4
        logging.info(f"Stitching tiles to form a new image of size {new_width}x{new_height}.")
        final_image = stitch_tiles(
            upscaled_tiles, rows=rows, cols=cols, tile_size=UPSCALED_TILE_SIZE
        )

        # Post-process to align edges if necessary
        final_image = align_edges(final_image, tile_size=UPSCALED_TILE_SIZE)

        # Save the upscaled image at this step
        upscaled_image_path = os.path.join(output_dir, f'step{step}_upscaled.png')
        final_image.save(upscaled_image_path)
        logging.info(f"Upscaled image at step {step} saved to '{upscaled_image_path}'.")

        current_image = final_image

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
            width=UPSCALED_TILE_SIZE,
            height=UPSCALED_TILE_SIZE,
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
    realesrgan_model = load_realesrgan_model(model_path=args.model)

    # Generate the high-resolution image
    create_high_resolution_image(
        initial_image=initial_image,
        upscale_exponent=args.upscale_exponent,
        output_dir=args.output_dir,
        model=realesrgan_model,
        hallucinate=args.hallucinate,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        strength=args.strength,
        inpaint_checkpoint=args.inpaint_checkpoint,
        min_area=args.min_area,
        border_thresh=args.border_thresh,
        hallucination_steps=args.hallucination_steps  # Pass the new argument here
    )

    logging.info("Image upscaling process completed successfully.")
    sys.exit(0)


if __name__ == '__main__':
    main()
