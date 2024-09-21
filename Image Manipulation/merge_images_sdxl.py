#!/usr/bin/env python3

# -------------------------------------------------------
# Script: merge_images_sdxl.py
#
# Description:
# This script intelligently merges multiple images into a single,
# coherent image using the Stable Diffusion XL (SDXL) model's
# image-to-image (img2img) capabilities. It ensures that the
# defining aspects of each input image are preserved in the
# final merged result.
#
# Usage:
# ./merge_images_sdxl.py [input_path] [options]
#
# - [input_path]: The path to the input image file or directory.
#
# Options:
# -r, --recursive            Process directories recursively.
# --blend-mode BLEND_MODE    Method to blend input images.
#                            Choices: "average", "weighted".
#                            (default: "average")
# --weights WEIGHTS          Comma-separated weights for weighted blending
#                            (required if blend-mode is "weighted").
# --prompt PROMPT            Text prompt to guide the image generation.
#                            (default: "A seamless blend of the input images.")
# --num-steps STEPS          Number of inference steps for the model.
#                            (default: 50)
# --guidance-scale SCALE     Guidance scale for the model.
#                            Controls how strongly the model follows the prompt.
#                            (default: 7.5)
# --resolution-mode RES_MODE
#                            Mode to determine output image resolution.
#                            Choices: "smallest", "biggest", "middle", "custom".
#                            (default: "middle")
# --width WIDTH              Custom width for the output image (required if resolution-mode is "custom").
# --height HEIGHT            Custom height for the output image (required if resolution-mode is "custom").
# -o OUTPUT_FILE, --output OUTPUT_FILE
#                            Output file name for the merged image (default: "merged_image_sdxl.png").
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - Pillow (install via: pip install Pillow)
# - PyTorch (install via: pip install torch torchvision torchaudio)
# - Hugging Face Diffusers and Transformers (install via: pip install diffusers transformers)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Intelligently merge multiple images into a single, coherent image using Stable Diffusion XL (SDXL).'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input image file or directory.'
    )
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Process directories recursively.'
    )
    parser.add_argument(
        '--blend-mode',
        type=str,
        default='average',
        choices=['average', 'weighted'],
        help='Method to blend input images (default: average).'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Comma-separated weights for weighted blending (required if blend-mode is "weighted").'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='A seamless blend of the input images.',
        help='Text prompt to guide the image generation (default: "A seamless blend of the input images.").'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='Number of inference steps for the model (default: 50).'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale for the model (default: 7.5). Controls how strongly the model follows the prompt.'
    )
    parser.add_argument(
        '--resolution-mode',
        type=str,
        default='middle',
        choices=['smallest', 'biggest', 'middle', 'custom'],
        help='Mode to determine output image resolution (default: middle).'
    )
    parser.add_argument(
        '--width',
        type=int,
        help='Custom width for the output image (required if resolution-mode is "custom").'
    )
    parser.add_argument(
        '--height',
        type=int,
        help='Custom height for the output image (required if resolution-mode is "custom").'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='merged_image_sdxl.png',
        help='Output file name for the merged image (default: merged_image_sdxl.png).'
    )
    args = parser.parse_args()

    # Validate custom resolution arguments
    if args.resolution_mode == 'custom':
        if args.width is None or args.height is None:
            parser.error('--width and --height must be specified when resolution-mode is "custom".')

    # Validate blend-mode and weights
    # This validation will be performed after collecting images in the main function
    return args

def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def collect_images(input_path, recursive):
    """
    Collects all image files from the input path.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_files = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith(supported_extensions):
            image_files.append(input_path)
        else:
            logging.error(f"File '{input_path}' is not a supported image format.")
    elif os.path.isdir(input_path):
        if recursive:
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_path):
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(input_path, file))
    else:
        logging.error(f"Input path '{input_path}' is neither a file nor a directory.")
        sys.exit(1)

    if not image_files:
        logging.error(f"No image files found in the specified path '{input_path}'.")
        sys.exit(1)

    logging.info(f"Found {len(image_files)} image(s) to merge.")
    return image_files

def determine_output_size(image_files, resolution_mode, custom_width=None, custom_height=None):
    """
    Determines the output image size based on the resolution mode.
    """
    sizes = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except Exception as e:
            logging.error(f"Failed to open image '{img_path}': {e}")
            sys.exit(1)

    if resolution_mode == 'smallest':
        width = min(size[0] for size in sizes)
        height = min(size[1] for size in sizes)
    elif resolution_mode == 'biggest':
        width = max(size[0] for size in sizes)
        height = max(size[1] for size in sizes)
    elif resolution_mode == 'middle':
        sorted_widths = sorted(size[0] for size in sizes)
        sorted_heights = sorted(size[1] for size in sizes)
        width = sorted_widths[len(sorted_widths) // 2]
        height = sorted_heights[len(sorted_heights) // 2]
    elif resolution_mode == 'custom':
        width = custom_width
        height = custom_height
    else:
        logging.error(f"Unknown resolution mode '{resolution_mode}'.")
        sys.exit(1)

    logging.info(f"Output image size set to: {width}x{height}")
    return width, height

def blend_images(image_files, blend_mode, weights=None):
    """
    Blends multiple images into a single image using the specified blend mode.
    """
    logging.info(f"Blending images using '{blend_mode}' mode.")

    # Load images
    images = []
    for idx, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            logging.info(f"Loaded image {idx + 1}/{len(image_files)}: '{img_path}' with size {img.size}")
        except Exception as e:
            logging.error(f"Failed to open image '{img_path}': {e}")
            sys.exit(1)

    # Determine target size based on resolution mode
    target_size = images[0].size
    resized_images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]

    if blend_mode == 'average':
        blended_array = np.mean([np.array(img) for img in resized_images], axis=0).astype(np.uint8)
    elif blend_mode == 'weighted':
        blended_array = np.zeros_like(np.array(resized_images[0]), dtype=np.float32)
        for img, weight in zip(resized_images, weights):
            blended_array += np.array(img) * weight
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
    else:
        logging.error(f"Unknown blend mode '{blend_mode}'.")
        sys.exit(1)

    blended_image = Image.fromarray(blended_array)
    logging.info("Images blended successfully.")
    return blended_image

def load_sdxl_model():
    """
    Loads the Stable Diffusion XL (SDXL) img2img pipeline from Hugging Face.
    """
    logging.info("Loading Stable Diffusion XL (SDXL) model. This may take a while...")
    try:
        # Recommended SDXL checkpoint for image-to-image tasks
        # Example: 'stabilityai/stable-diffusion-xl-base-1.0'
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None  # Disable safety checker for flexibility
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")
        logging.info("Stable Diffusion XL model loaded successfully.")
        return pipe
    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion XL model: {e}")
        sys.exit(1)

def generate_merged_image(pipe, blended_image, prompt, num_steps, guidance_scale, width, height):
    """
    Generates a merged image using the Stable Diffusion XL img2img pipeline.
    """
    logging.info("Generating merged image using Stable Diffusion XL...")
    try:
        # Preprocess the blended image
        init_image = blended_image.resize((width, height))
        init_image = init_image.convert("RGB")

        # Generate the image
        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            result = pipe(
                prompt=prompt,
                init_image=init_image,
                strength=0.6,  # Controls how much to transform the init image (0.0-1.0)
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            )
        merged_image = result.images[0]
        logging.info("Merged image generated successfully.")
        return merged_image
    except Exception as e:
        logging.error(f"Failed to generate merged image: {e}")
        sys.exit(1)

def save_image(image, output_path):
    """
    Saves the PIL image to the specified path.
    """
    try:
        image.save(output_path)
        logging.info(f"Merged image saved as '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save merged image '{output_path}': {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    setup_logging()

    input_path = args.input_path
    recursive = args.recursive
    blend_mode = args.blend_mode
    weights = args.weights
    prompt = args.prompt
    num_steps = args.num_steps
    guidance_scale = args.guidance_scale
    resolution_mode = args.resolution_mode
    custom_width = args.width
    custom_height = args.height
    output_file = args.output

    # Collect image files
    image_files = collect_images(input_path, recursive)

    # Validate weights after collecting images
    if blend_mode == 'weighted':
        if weights is None:
            logging.error('--weights must be specified when blend-mode is "weighted".')
            sys.exit(1)
        try:
            weights = [float(w) for w in weights.split(',')]
            if len(weights) != len(image_files):
                logging.error(f'Number of weights ({len(weights)}) does not match number of input images ({len(image_files)}).')
                sys.exit(1)
            if not np.isclose(sum(weights), 1.0):
                logging.warning('Weights do not sum to 1. Normalizing weights.')
                weights = [w / sum(weights) for w in weights]
            logging.info(f'Using weights: {weights}')
        except ValueError:
            logging.error('Weights must be a comma-separated list of numbers.')
            sys.exit(1)
    else:
        weights = None

    # Determine output image size
    output_width, output_height = determine_output_size(
        image_files,
        resolution_mode,
        custom_width=custom_width,
        custom_height=custom_height
    )

    # Blend images
    blended_image = blend_images(
        image_files,
        blend_mode,
        weights=weights
    )

    # Load SDXL model
    pipe = load_sdxl_model()

    # Generate merged image using img2img
    merged_image = generate_merged_image(
        pipe,
        blended_image,
        prompt,
        num_steps,
        guidance_scale,
        output_width,
        output_height
    )

    # Save the merged image
    save_image(merged_image, output_file)

if __name__ == '__main__':
    main()
