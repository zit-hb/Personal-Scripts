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
# splitting it into tiles, and optionally performing hallucination
# (inpainting) on each tile.
#
# Usage:
# ./upscale_image.py [options]
#
# Options:
#   -i, --input                  Path to the input image to be upscaled.
#                                If not provided, an initial image will be generated using SDXL.
#   -d, --output-dir             Directory to save the output images (default: "output").
#   -v, --verbose                Enable verbose logging (DEBUG level).
#
#   SDXL Generation Options  (used only if --input is not specified):
#     -p, --prompt               The prompt to use for SDXL image generation (default: "A beautiful landscape").
#     -P, --negative-prompt      A negative prompt to help guide the model on what to avoid.
#     -s, --seed                 Random seed for reproducibility (default: None).
#     -n, --num-steps            Number of inference steps for SDXL text-to-image (default: 50).
#     -N, --inpaint-steps        Number of inference steps for inpainting (default: 75, higher for better quality).
#     -g, --guidance-scale       Guidance scale for SDXL (default: 7.5).
#     -T, --txt2img-checkpoint   SDXL checkpoint to use for text-to-image generation
#                                (default: "stabilityai/stable-diffusion-xl-base-1.0").
#     -C, --inpaint-checkpoint   SDXL checkpoint to use for inpainting tasks
#                                (default: "stabilityai/stable-diffusion-2-inpainting").
#
#   Real-ESRGAN Upscaling Options:
#     -u, --upscale-exponent     Number of upscaling levels. Controls how many times the hallucination and upscaling steps are repeated.
#                                (default: 1).
#     -f, --upscale-factor       Upscale factor for Real-ESRGAN (default: 4). Allowed values are 2 and 4.
#
#   Hallucination Options:
#     -H, --hallucinate          Enable hallucination feature to modify tiles using SDXL inpainting.
#     -t, --strength             Strength of the inpainting effect (0.0 to 1.0) (default: 0.2).
#     -e, --min-area             Minimum area (in pixels) for a superpixel to be considered (default: 1000).
#     -b, --border-thresh        Threshold (in pixels) to avoid hallucination near borders (default: 50).
#     -S, --hallucination-steps  How many times to perform inpainting per tile (default: 1).
#
#   Tile Settings:
#     -Z, --tile-size            Tile size (both width and height) used for splitting image (default: 768).
#     -O, --tile-overlap         Overlap between tiles in pixels to allow blending (default: 128).
#
#   Additional Options:
#     -rpt, --realesrgan-tile-pad    Tile padding for Real-ESRGAN (default: 30).
#     -rpp, --realesrgan-pre-pad     Pre padding for Real-ESRGAN (default: 10).
#     -sc, --slic-compactness        Compactness parameter for SLIC segmentation (default: 10.0).
#     -sba, --slic-baseline-area     Baseline area for adaptive SLIC segmentation (default: 589824).
#     -bw, --blending-window         Blending window type ("hamming", "hann", "uniform") (default: "hamming").
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
#   - OpenCV (install via: apt-get install -y python3-opencv opencv-data)
#   - Pillow (install via: pip install Pillow==11.0.0)
#   - Torch & Torchvision (install via: pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2)
#   - numpy (install via: pip install numpy==1.26.4)
#   - diffusers (install via: pip install diffusers==0.31.0)
#   - requests (install via: pip install requests==2.32.3)
#   - basicsr (install via: pip install basicsr==1.4.2)
#   - realesrgan (install via: pip install realesrgan==0.3.0)
#   - scikit-image (install via: pip install scikit-image==0.24.0)
#   - transformers (install via: pip install transformers==4.47.0)
#   - accelerate (install via: pip install accelerate==1.2.0)
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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import requests
import numpy as np
from PIL import Image
import torch
import cv2

from diffusers import StableDiffusionXLPipeline, StableDiffusionInpaintPipeline
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from skimage.segmentation import slic
from skimage.util import img_as_float

INITIAL_IMAGE_SIZE = 1024

DEFAULT_REALESRGAN_MODEL_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CACHE_DIR = Path.home() / ".cache" / "upscale_image"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REALESRGAN_CACHE_DIR = CACHE_DIR / "realesrgan"
REALESRGAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for configuring the upscaling and hallucination process.
    """
    parser = argparse.ArgumentParser(
        description="Generates a high-resolution image by upscaling and optionally hallucinating tiles with SDXL."
    )
    # Input / Output
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Path to input image. If not provided, an initial image will be generated using SDXL.'
    )
    parser.add_argument(
        '-d', '--output-dir',
        type=str,
        default='output',
        help='Directory to save output images. (default: output)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level).'
    )

    # SDXL Generation Options
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        default='A beautiful landscape',
        help='Prompt for SDXL generation. (default: "A beautiful landscape")'
    )
    parser.add_argument(
        '-P', '--negative-prompt',
        type=str,
        default=None,
        help='Negative prompt for SDXL to help improve quality.'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Seed for reproducibility.'
    )
    parser.add_argument(
        '-n', '--num-steps',
        type=int,
        default=50,
        help='Number of inference steps for SDXL text-to-image. (default: 50)'
    )
    parser.add_argument(
        '-N', '--inpaint-steps',
        type=int,
        default=75,
        help='Number of inference steps for SDXL inpainting. (default: 75)'
    )
    parser.add_argument(
        '-g', '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale for SDXL. (default: 7.5)'
    )
    parser.add_argument(
        '-T', '--txt2img-checkpoint',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='SDXL checkpoint for text-to-image. (default: stabilityai/stable-diffusion-xl-base-1.0)'
    )
    parser.add_argument(
        '-C', '--inpaint-checkpoint',
        type=str,
        default='stabilityai/stable-diffusion-2-inpainting',
        help='Checkpoint for inpainting tasks. (default: stabilityai/stable-diffusion-2-inpainting)'
    )

    # Upscaling Options
    parser.add_argument(
        '-u', '--upscale-exponent',
        type=int,
        default=1,
        help='How many iterative upscaling/hallucination levels. (default: 1)'
    )
    parser.add_argument(
        '-f', '--upscale-factor',
        type=int,
        default=4,
        choices=[2, 4],
        help='Upscale factor for Real-ESRGAN. (default: 4)'
    )

    # Hallucination Options
    parser.add_argument(
        '-H', '--hallucinate',
        action='store_true',
        help='Enable hallucination (inpainting) on tiles.'
    )
    parser.add_argument(
        '-t', '--strength',
        type=float,
        default=0.2,
        help='Inpainting strength (0.0 to 1.0). (default: 0.2)'
    )
    parser.add_argument(
        '-e', '--min-area',
        type=int,
        default=1000,
        help='Minimum superpixel area for inpainting. (default: 1000)'
    )
    parser.add_argument(
        '-b', '--border-thresh',
        type=int,
        default=50,
        help='Avoid hallucination near borders (in pixels). (default: 50)'
    )
    parser.add_argument(
        '-S', '--hallucination-steps',
        type=int,
        default=1,
        help='Number of inpainting iterations per tile. (default: 1)'
    )

    # Tile Settings
    parser.add_argument(
        '-Z', '--tile-size',
        type=int,
        default=768,
        help='Base tile size for splitting image. (default: 768)'
    )
    parser.add_argument(
        '-O', '--tile-overlap',
        type=int,
        default=128,
        help='Overlap between tiles (in pixels). (default: 128)'
    )

    # Additional adjustable parameters
    parser.add_argument(
        '-rpt', '--realesrgan-tile-pad',
        type=int,
        default=30,
        help='Tile padding for Real-ESRGAN. (default: 30)'
    )
    parser.add_argument(
        '-rpp', '--realesrgan-pre-pad',
        type=int,
        default=10,
        help='Pre padding for Real-ESRGAN. (default: 10)'
    )
    parser.add_argument(
        '-sc', '--slic-compactness',
        type=float,
        default=10.0,
        help='Compactness parameter for SLIC segmentation. (default: 10.0)'
    )
    parser.add_argument(
        '-sba', '--slic-baseline-area',
        type=int,
        default=589824,  # 768x768
        help='Baseline area for adaptive SLIC segmentation. (default: 589824)'
    )
    parser.add_argument(
        '-bw', '--blending-window',
        type=str,
        choices=["hamming", "hann", "uniform"],
        default="hamming",
        help='Blending window type used for tile blending. (default: hamming)'
    )

    args = parser.parse_args()

    # Basic sanity checks.
    if args.upscale_exponent < 1 or args.upscale_exponent > 5:
        parser.error("upscale_exponent must be between 1 and 5.")
    if not (0.0 <= args.strength <= 1.0):
        parser.error("strength must be between 0.0 and 1.0.")
    if args.min_area < 0 or args.border_thresh < 0:
        parser.error("min_area and border_thresh must be non-negative.")
    if args.input is not None and not os.path.isfile(args.input):
        parser.error(f"Input image '{args.input}' does not exist.")

    return args


def setup_logging(verbose: bool = False) -> None:
    """
    Sets up logging configuration with a specific verbosity level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def download_realesrgan_model(upscale_factor: int) -> Path:
    """
    Downloads the Real-ESRGAN model if not already present in the cache directory.
    """
    url = DEFAULT_REALESRGAN_MODEL_URLS.get(upscale_factor)
    if url is None:
        logging.error(f"No default Real-ESRGAN model URL for upscale factor {upscale_factor}.")
        sys.exit(1)

    model_filename = os.path.basename(url)
    model_path = REALESRGAN_CACHE_DIR / model_filename

    if model_path.is_file():
        logging.info(f"Real-ESRGAN model already at '{model_path}'.")
        return model_path

    logging.info(f"Downloading Real-ESRGAN model from '{url}'.")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logging.info(f"Downloaded Real-ESRGAN model to '{model_path}'.")
    return model_path


def load_sdxl_pipeline(checkpoint: str) -> StableDiffusionXLPipeline:
    """
    Loads the SDXL text-to-image pipeline from the specified checkpoint.
    """
    logging.info(f"Loading SDXL pipeline from '{checkpoint}'...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        ).to(DEVICE)
        pipe.safety_checker = None
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        return pipe
    except Exception:
        logging.exception(f"Failed to load SDXL pipeline '{checkpoint}'.")
        sys.exit(1)


def load_inpaint_pipeline(checkpoint: str) -> StableDiffusionInpaintPipeline:
    """
    Loads the SDXL inpainting pipeline from the specified checkpoint.
    """
    logging.info(f"Loading Inpainting pipeline from '{checkpoint}'...")
    try:
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        ).to(DEVICE)
        inpaint_pipe.safety_checker = None
        inpaint_pipe.enable_attention_slicing()
        try:
            inpaint_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        return inpaint_pipe
    except Exception:
        logging.exception("Failed to load inpainting pipeline.")
        sys.exit(1)


def generate_initial_image(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    num_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: Optional[int] = None
) -> Image.Image:
    """
    Generates the initial image using the SDXL text-to-image pipeline.
    """
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
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        )

    if not result.images:
        logging.error("No images returned by SDXL txt2img.")
        sys.exit(1)

    image = result.images[0]
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def load_realesrgan_model(upscale_factor: int, tile_pad: int, pre_pad: int) -> RealESRGANer:
    """
    Loads the Real-ESRGAN model for the specified upscale factor.
    """
    model_path = download_realesrgan_model(upscale_factor)

    if upscale_factor == 2:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
        )
    else:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )

    realesrgan_model = RealESRGANer(
        scale=upscale_factor,
        model_path=str(model_path),
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=True if DEVICE.type == "cuda" else False,
        device=DEVICE
    )
    return realesrgan_model


def cleanup_memory():
    """
    Cleans up memory by collecting garbage and clearing CUDA cache if available.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def align_to_multiple(size: int, multiple: int = 8) -> int:
    """
    Aligns a given size down to the nearest multiple of a specified value.
    """
    return size - (size % multiple)


def generate_tiles(image: Image.Image, tile_size: int, tile_overlap: int) -> List[Tuple[Image.Image, int, int]]:
    """
    Splits the image into overlapping tiles to facilitate blending and reduce seams.
    Edge areas are padded by replicating edge pixels.
    """
    width, height = image.size
    tile_size = align_to_multiple(tile_size, 8)
    tile_overlap = align_to_multiple(tile_overlap, 8)

    tiles = []
    step = tile_size - tile_overlap
    x_positions = list(range(0, width, step))
    y_positions = list(range(0, height, step))

    if x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)
    if y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)

    for y in y_positions:
        for x in x_positions:
            box = (x, y, x + tile_size, y + tile_size)
            tile = image.crop(box)
            # If tile is smaller than expected, replicate edge pixels.
            if tile.size != (tile_size, tile_size):
                tile_np = np.array(tile)
                pad_w = tile_size - tile_np.shape[1]
                pad_h = tile_size - tile_np.shape[0]
                if pad_w > 0 or pad_h > 0:
                    tile_np = np.pad(tile_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                tile = Image.fromarray(tile_np)
            tiles.append((tile, x, y))

    return tiles


def blend_tiles(tiles_with_positions: List[Tuple[Image.Image, int, int]], image_size: Tuple[int, int],
                window_type: str = "hamming") -> Image.Image:
    """
    Blends the processed tiles back into a single image.
    Uses a smooth weighting window (hamming, hann, or uniform) to feather overlapping areas.
    """
    width, height = image_size
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_acc = np.zeros((height, width), dtype=np.float32)

    for tile, x, y in tiles_with_positions:
        tile_np = np.array(tile, dtype=np.float32)
        h, w = tile_np.shape[:2]
        if window_type == "hamming":
            tile_weight = np.outer(np.hamming(h), np.hamming(w))
        elif window_type == "hann":
            tile_weight = np.outer(np.hanning(h), np.hanning(w))
        else:
            tile_weight = np.ones((h, w), dtype=np.float32)
        end_y = min(y + h, height)
        end_x = min(x + w, width)
        tile_crop_h = end_y - y
        tile_crop_w = end_x - x
        if tile_crop_h <= 0 or tile_crop_w <= 0:
            continue

        accumulator[y:end_y, x:end_x] += tile_np[:tile_crop_h, :tile_crop_w] * tile_weight[:tile_crop_h, :tile_crop_w, None]
        weight_acc[y:end_y, x:end_x] += tile_weight[:tile_crop_h, :tile_crop_w]

    weight_acc = np.maximum(weight_acc, 1e-5)
    blended = (accumulator / weight_acc[..., None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended, 'RGB')


def upscale_with_realesrgan(model: RealESRGANer, image: Image.Image, outscale: float) -> Image.Image:
    """
    Upscales a given image using the Real-ESRGAN model.
    """
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    upscaled_bgr, _ = model.enhance(img_bgr, outscale=outscale)
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(upscaled_rgb)


def create_superpixel_mask(image_np: np.ndarray, n_segments: int, compactness: float,
                           min_area: int, border_thresh: int,
                           excluded_segments: Optional[List[int]] = None) -> Optional[Image.Image]:
    """
    Creates a mask by segmenting the image into superpixels and selecting one suitable area for inpainting.
    """
    if excluded_segments is None:
        excluded_segments = []

    image_float = img_as_float(image_np)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)
    height, width = segments.shape
    unique, counts = np.unique(segments, return_counts=True)

    suitable_segments = []
    for seg_id, area in zip(unique, counts):
        if area < min_area:
            continue
        if seg_id in excluded_segments:
            continue
        ys, xs = np.where(segments == seg_id)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        if y_min < border_thresh or y_max > (height - border_thresh) or \
           x_min < border_thresh or x_max > (width - border_thresh):
            continue
        suitable_segments.append((seg_id, area))

    if not suitable_segments:
        return None

    suitable_segments.sort(key=lambda x: x[1], reverse=True)
    selected_seg_id = suitable_segments[0][0]

    mask = np.zeros(segments.shape, dtype=np.uint8)
    mask[segments == selected_seg_id] = 255
    mask_image = Image.fromarray(mask).convert("L")
    return mask_image


def inpaint_tile(
    tile: Image.Image,
    mask: Image.Image,
    inpaint_pipe: StableDiffusionInpaintPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    strength: float,
) -> Image.Image:
    """
    Inpaints a given tile using the provided mask and SDXL inpainting pipeline.
    """
    tw, th = tile.size
    tw_aligned = align_to_multiple(tw, 8)
    th_aligned = align_to_multiple(th, 8)
    if (tw_aligned != tw) or (th_aligned != th):
        tile = tile.crop((0, 0, tw_aligned, th_aligned))
        mask = mask.crop((0, 0, tw_aligned, th_aligned))

    result = inpaint_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=tile,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        width=tile.width,
        height=tile.height,
    )
    inpainted_tile = result.images[0].convert("RGB")
    return inpainted_tile


def perform_hallucination(
    tile: Image.Image,
    inpaint_pipe: StableDiffusionInpaintPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    strength: float,
    min_area: int,
    border_thresh: int,
    hallucination_steps: int,
    slic_baseline_area: int,
    slic_compactness: float
) -> Image.Image:
    """
    Performs hallucination (inpainting) on a tile by finding superpixel masks and refining the tile.
    Adaptive SLIC parameters are used based on tile area.
    Note: Hallucination is now done on the lower resolution tile (before upscaling) to reduce VRAM usage.
    """
    current_tile = tile.copy()
    current_tile_np = np.array(current_tile)
    excluded_segments = []

    for step in range(1, hallucination_steps + 1):
        tile_area = current_tile_np.shape[0] * current_tile_np.shape[1]
        n_segments = max(50, int(50 * (tile_area / slic_baseline_area)))
        mask = create_superpixel_mask(
            image_np=current_tile_np,
            n_segments=n_segments,
            compactness=slic_compactness,
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
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
            )
            current_tile = inpainted_tile
            current_tile_np = np.array(current_tile)
            segments = slic(img_as_float(current_tile_np), n_segments=n_segments, compactness=slic_compactness, start_label=1)
            selected_segments = np.unique(segments[np.array(mask) == 255])
            excluded_segments.extend(selected_segments.tolist())
        else:
            break

    return current_tile


def create_high_resolution_image(
    initial_image: Image.Image,
    upscale_exponent: int,
    output_dir: str,
    model: RealESRGANer,
    upscale_factor: int,
    hallucinate: bool,
    prompt: str,
    negative_prompt: Optional[str],
    guidance_scale: float,
    num_inference_steps: int,
    inpaint_steps: int,
    strength: float,
    inpaint_checkpoint: str,
    min_area: int,
    border_thresh: int,
    hallucination_steps: int,
    tile_size: int,
    tile_overlap: int,
    blending_window: str,
    slic_baseline_area: int,
    slic_compactness: float
) -> None:
    """
    Creates the final high-resolution image by repeatedly upscaling and optionally hallucinating the image.
    Saves intermediate results for each level.
    """
    initial_image_path = os.path.join(output_dir, 'initial_image.png')
    initial_image.save(initial_image_path)

    current_image = initial_image
    inpaint_pipe = None
    if hallucinate:
        inpaint_pipe = load_inpaint_pipeline(checkpoint=inpaint_checkpoint)

    for level in range(1, upscale_exponent + 1):
        logging.info(f"Starting upscale level {level}/{upscale_exponent}.")
        tiles_with_positions = generate_tiles(current_image, tile_size, tile_overlap)
        processed_tiles = []

        for idx, (tile, x, y) in enumerate(tiles_with_positions):
            logging.debug(f"Processing tile {idx+1}/{len(tiles_with_positions)} at ({x}, {y}).")
            # If hallucination is enabled, perform it on the low-resolution tile first to save VRAM.
            if hallucinate and inpaint_pipe:
                tile = perform_hallucination(
                    tile=tile,
                    inpaint_pipe=inpaint_pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=inpaint_steps,
                    strength=strength,
                    min_area=min_area,
                    border_thresh=border_thresh,
                    hallucination_steps=hallucination_steps,
                    slic_baseline_area=slic_baseline_area,
                    slic_compactness=slic_compactness,
                )
            # Then upscale the (optionally hallucinated) tile.
            tile = upscale_with_realesrgan(model, tile, outscale=upscale_factor)
            new_x = x * upscale_factor
            new_y = y * upscale_factor
            processed_tiles.append((tile, new_x, new_y))
            cleanup_memory()

        new_width = current_image.width * upscale_factor
        new_height = current_image.height * upscale_factor
        current_image = blend_tiles(processed_tiles, (new_width, new_height), window_type=blending_window)

        level_image_path = os.path.join(output_dir, f'level{level}_image.png')
        current_image.save(level_image_path)

    final_image_path = os.path.join(output_dir, 'final_image.png')
    current_image.save(final_image_path)

    if inpaint_pipe:
        del inpaint_pipe
    cleanup_memory()


def create_initial_image_if_needed(args: argparse.Namespace) -> Image.Image:
    """
    Loads the input image if provided, otherwise generates an initial image using SDXL.
    """
    if args.input is not None:
        try:
            initial_image = Image.open(args.input).convert('RGB')
            logging.info(f"Loaded input image from '{args.input}'.")
            return initial_image
        except Exception:
            logging.exception(f"Failed to open input image '{args.input}'.")
            sys.exit(1)
    else:
        sdxl_pipe = load_sdxl_pipeline(checkpoint=args.txt2img_checkpoint)
        initial_image = generate_initial_image(
            pipe=sdxl_pipe,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            width=INITIAL_IMAGE_SIZE,
            height=INITIAL_IMAGE_SIZE,
            seed=args.seed,
        )
        return initial_image


def main() -> None:
    """
    The main function to orchestrate the image upscaling and hallucination process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    logging.info(f"Using device: {DEVICE}")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: '{args.output_dir}'")

    initial_image = create_initial_image_if_needed(args)
    realesrgan_model = load_realesrgan_model(
        args.upscale_factor,
        tile_pad=args.realesrgan_tile_pad,
        pre_pad=args.realesrgan_pre_pad
    )

    create_high_resolution_image(
        initial_image=initial_image,
        upscale_exponent=args.upscale_exponent,
        output_dir=args.output_dir,
        model=realesrgan_model,
        upscale_factor=args.upscale_factor,
        hallucinate=args.hallucinate,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        inpaint_steps=args.inpaint_steps,
        strength=args.strength,
        inpaint_checkpoint=args.inpaint_checkpoint,
        min_area=args.min_area,
        border_thresh=args.border_thresh,
        hallucination_steps=args.hallucination_steps,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        blending_window=args.blending_window,
        slic_baseline_area=args.slic_baseline_area,
        slic_compactness=args.slic_compactness,
    )

    logging.info("Image upscaling and processing completed successfully.")
    sys.exit(0)


if __name__ == '__main__':
    main()
