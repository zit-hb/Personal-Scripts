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
# This involves selecting an object within a tile using Detectron2, creating a mask,
# and then using SDXL Inpainting to slightly alter the object and add more details.
#
# Usage:
# ./upscale_image.py [options]
#
# Options:
#   -i, --input            Path to the input image to be upscaled.
#                          If not provided, an initial image will be generated using SDXL.
#
#   SDXL Generation Options (used only if --input is not specified):
#     -p, --prompt          The prompt to use for SDXL image generation (default: "A beautiful landscape").
#     -s, --seed            Random seed for reproducibility (default: None).
#     -n, --num-steps       Number of inference steps for the SDXL model (default: 50).
#     -g, --guidance-scale  Guidance scale for the SDXL model (default: 7.5).
#     -c, --checkpoint      SDXL checkpoint to use (default: "stabilityai/stable-diffusion-xl-base-1.0").
#
#   Real-ESRGAN Upscaling Options:
#     -u, --upscale-exponent Number of upscaling steps. Each step upscales the image by a factor of 4
#                            (quadrupling width and height). (default: 2).
#     -m, --model           Path to the Real-ESRGAN model weights file. (default: "RealESRGAN_x4plus.pth").
#
#   Hallucination Options:
#     -H, --hallucinate     Enable hallucination feature to modify upscaled tiles using SDXL inpainting.
#
#   General Options:
#     -d, --output-dir      Directory to save the output images (default: "output").
#     -v, --verbose         Enable verbose logging (DEBUG level).
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
# - detectron2 (optional, required if using --hallucinate) (install via: pip install detectron2)
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

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

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
        help='The prompt to use for SDXL image generation (default: "A beautiful landscape").'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=None,
        help='Random seed for reproducibility (default: None).'
    )
    parser.add_argument(
        '-n', '--num-steps', type=int, default=50,
        help='Number of inference steps for the SDXL model (default: 50).'
    )
    parser.add_argument(
        '-g', '--guidance-scale', type=float, default=7.5,
        help='Guidance scale for the SDXL model (default: 7.5).'
    )
    parser.add_argument(
        '-c', '--checkpoint', type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='SDXL checkpoint to use (default: "stabilityai/stable-diffusion-xl-base-1.0").'
    )

    # Real-ESRGAN Upscaling Options
    parser.add_argument(
        '-u', '--upscale-exponent', type=int, default=2,
        help='Number of upscaling steps. Each step upscales the image by a factor of 4 '
             '(quadrupling width and height). (default: 2).'
    )
    parser.add_argument(
        '-m', '--model', type=str, default='RealESRGAN_x4plus.pth',
        help='Path to the Real-ESRGAN model weights file. (default: "RealESRGAN_x4plus.pth").'
    )

    # Hallucination Option
    parser.add_argument(
        '-H', '--hallucinate', action='store_true',
        help='Enable hallucination feature to modify upscaled tiles using SDXL inpainting.'
    )

    # General Options
    parser.add_argument(
        '-d', '--output-dir', type=str, default='output',
        help='Directory to save the output images (default: "output").'
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

    # If input image is not provided, ensure that SDXL parameters are valid
    if args.input is not None and not os.path.isfile(args.input):
        parser.error(f"Input image '{args.input}' does not exist.")

    # Validate Real-ESRGAN model path or download if not present
    if not os.path.isfile(args.model):
        setup_logging(verbose=args.verbose)  # Initialize logging to capture download logs
        logging.info(f"Real-ESRGAN model file '{args.model}' not found. Attempting to download...")
        download_realesrgan_model(args.model)
        if not os.path.isfile(args.model):
            parser.error(f"Failed to download Real-ESRGAN model to '{args.model}'. Please provide a valid Real-ESRGAN model weights file.")

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


def load_inpaint_pipeline(checkpoint: str) -> StableDiffusionXLInpaintPipeline:
    """
    Loads the SDXL Inpainting pipeline for hallucination.
    """
    logging.info(f"Loading SDXL Inpainting pipeline from checkpoint '{checkpoint}'. This may take a while...")
    try:
        inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        inpaint_pipe = inpaint_pipe.to(DEVICE)
        inpaint_pipe.safety_checker = None  # Disable safety checker for faster inference
        inpaint_pipe.enable_attention_slicing()
        try:
            inpaint_pipe.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention for SDXL inpainting.")
        except Exception as e:
            logging.warning(f"Failed to enable xformers memory efficient attention for SDXL inpainting: {e}")
        logging.info("SDXL Inpainting pipeline loaded successfully.")
        return inpaint_pipe
    except Exception as e:
        logging.exception(f"Failed to load SDXL Inpainting pipeline from checkpoint '{checkpoint}'.")
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


def initialize_detectron2(device: torch.device) -> 'DefaultPredictor':
    """
    Initializes the Detectron2 predictor for object detection.
    """
    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
    except ImportError:
        logging.error("Detectron2 is not installed. Install it using 'pip install detectron2'. Required for hallucination feature.")
        sys.exit(1)

    logging.info("Initializing Detectron2 predictor for hallucination.")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = str(device)
    predictor = DefaultPredictor(cfg)
    logging.info("Detectron2 predictor initialized successfully.")
    return predictor


def perform_hallucination(
    tile: Image.Image,
    predictor: 'DefaultPredictor',
    inpaint_pipe: StableDiffusionXLInpaintPipeline,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    output_dir: str,
    row_idx: int,
    col_idx: int
) -> Image.Image:
    """
    Performs hallucination on a given tile using Detectron2 and SDXL Inpainting.
    """
    try:
        upscaled_tile_np = np.array(tile)
        outputs = predictor(upscaled_tile_np)
        instances = outputs["instances"]

        selected_instance = select_instance(instances, tile.size)
        if selected_instance is not None:
            mask = create_mask(selected_instance, output_dir, row_idx, col_idx)
            inpainted_tile = inpaint_tile(
                tile=tile,
                mask=mask,
                inpaint_pipe=inpaint_pipe,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_dir=output_dir,
                row_idx=row_idx,
                col_idx=col_idx
            )
            return inpainted_tile
        else:
            logging.debug(f"No suitable object found for hallucination in tile at Row: {row_idx}, Column: {col_idx}.")
            return tile
    except Exception as e:
        logging.exception("Failed during hallucination process.")
        sys.exit(1)


def select_instance(instances, image_size: tuple) -> Optional:
    """
    Selects a suitable instance from Detectron2 predictions based on size and position.
    """
    image_width, image_height = image_size
    min_area = 0.01 * image_width * image_height  # 1% of image area
    max_area = 0.2 * image_width * image_height  # 20% of image area
    border_threshold = 0.05 * min(image_width, image_height)  # 5% of image size

    for i in range(len(instances)):
        bbox = instances.pred_boxes[i].tensor.cpu().numpy()[0]  # x1, y1, x2, y2
        area = instances.pred_boxes.area()[i].item()

        if area < min_area or area > max_area:
            continue

        x1, y1, x2, y2 = bbox
        if x1 < border_threshold or y1 < border_threshold or \
           x2 > image_width - border_threshold or y2 > image_height - border_threshold:
            continue

        return instances[i]
    return None


def create_mask(instance, output_dir: str, row_idx: int, col_idx: int) -> Image.Image:
    """
    Creates and saves a mask image for the selected instance.
    """
    mask = instance.pred_masks.cpu().numpy()[0].astype(np.uint8) * 255
    mask_image = Image.fromarray(mask).convert("L")
    mask_path = os.path.join(output_dir, f'step_mask_row{row_idx}_col{col_idx}.png')
    mask_image.save(mask_path)
    logging.debug(f"Mask saved to '{mask_path}'.")
    return mask_image


def inpaint_tile(
    tile: Image.Image,
    mask: Image.Image,
    inpaint_pipe: StableDiffusionXLInpaintPipeline,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    output_dir: str,
    row_idx: int,
    col_idx: int
) -> Image.Image:
    """
    Performs inpainting on the given tile using the provided mask.
    """
    try:
        result = inpaint_pipe(
            prompt=prompt,
            image=tile,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        inpainted_tile = result.images[0]
        mod_tile_path = os.path.join(output_dir, f'step_mod_tile_row{row_idx}_col{col_idx}.png')
        inpainted_tile.save(mod_tile_path)
        logging.debug(f"Modified tile saved to '{mod_tile_path}'.")
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
    num_inference_steps: int = 50
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

    predictor = None
    inpaint_pipe = None
    if hallucinate:
        predictor = initialize_detectron2(DEVICE)
        inpaint_pipe = load_inpaint_pipeline(checkpoint=args.checkpoint)

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

            upscaled_tile = upscale_tile(model, tile)

            intermediate_tile_path = os.path.join(output_dir, f'step{step}_tile_row{row_idx}_col{col_idx}.png')
            upscaled_tile.save(intermediate_tile_path)
            logging.debug(f"Upscaled tile saved to '{intermediate_tile_path}'.")

            if hallucinate and predictor and inpaint_pipe:
                upscaled_tile = perform_hallucination(
                    tile=upscaled_tile,
                    predictor=predictor,
                    inpaint_pipe=inpaint_pipe,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_dir=output_dir,
                    row_idx=row_idx,
                    col_idx=col_idx
                )

            upscaled_tiles.append(upscaled_tile)
            cleanup_memory()

        # Stitch the upscaled tiles back together
        new_width = current_image.width * 4
        new_height = current_image.height * 4
        logging.info(f"Stitching tiles to form a new image of size {new_width}x{new_height}.")
        final_image = stitch_tiles(upscaled_tiles, rows=rows, cols=cols, tile_size=UPSCALED_TILE_SIZE)

        # Post-process to align edges if necessary
        final_image = align_edges(final_image, tile_size=UPSCALED_TILE_SIZE)

        # Save the upscaled image at this step
        upscaled_image_path = os.path.join(output_dir, f'step{step}_upscaled.png')
        final_image.save(upscaled_image_path)
        logging.info(f"Upscaled image at step {step} saved to '{upscaled_image_path}'.")

        current_image = final_image

    if hallucinate:
        inpaint_pipe = None
        predictor = None
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
        sdxl_pipe = load_sdxl_pipeline(checkpoint=args.checkpoint)
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
        num_inference_steps=args.num_steps
    )

    logging.info("Image upscaling process completed successfully.")
    sys.exit(0)


if __name__ == '__main__':
    main()
