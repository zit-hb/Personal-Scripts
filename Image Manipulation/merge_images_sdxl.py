#!/usr/bin/env python3

# -------------------------------------------------------
# Script: merge_images_sdxl.py
#
# Description:
# This script intelligently merges multiple images into a single,
# coherent image using the Stable Diffusion XL (SDXL) model's
# image-to-image (img2img) capabilities. It preserves the
# defining aspects of each input image while fixing artifacts
# resulting from the merging process.
#
# Usage:
# ./merge_images_sdxl.py [input_path] [options]
#
#   [input_path]: The path to the input image file or directory.
#
# Options:
# -r, --recursive            Process directories recursively.
# -b, --blend-mode BLEND_MODE
#                            Method to blend input images. Choices: "average", "weighted". (default: "average")
# -w, --weights WEIGHTS      Comma-separated weights for weighted blending (required if blend-mode is "weighted").
# -n, --num-steps STEPS      Number of inference steps for the model. (default: 50)
# -g, --guidance-scale SCALE Guidance scale for the model. (default: 4)
# -m, --resolution-mode RES_MODE
#                            Mode to determine output image resolution. Choices: "smallest", "biggest", "middle", "custom". (default: "middle")
# -x, --width WIDTH          Custom width for the output image (required if resolution-mode is "custom").
# -y, --height HEIGHT        Custom height for the output image (required if resolution-mode is "custom").
# -s, --scheduler SCHEDULER  Scheduler (sampler) to use. Choices: "ddim", "plms", "k_lms", "euler", "euler_a", "heun", "dpm_solver". (default: "ddim")
# -c, --checkpoint CHECKPOINT
#                            SDXL checkpoint to use. Can be a Hugging Face model ID or a local path.
# -f, --refiner REFINER      SDXL refiner checkpoint to use. Can be a Hugging Face model ID or a local path. (default: "stabilityai/stable-diffusion-xl-refiner-1.0")
# -l, --lora LORA_PATH       LoRA model path. Can be specified multiple times for multiple LoRAs.
# -o, --output OUTPUT_FILE   Output file name for the merged image (default: "merged_image_sdxl.png").
# -v, --verbose              Enable verbose logging (DEBUG level).
# -d, --default-prompt       Use a default prompt instead of generating one via BLIP.
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
# - Pillow (install via: pip install Pillow==11.0.0)
# - PyTorch (install via: pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1)
# - Hugging Face Diffusers and Transformers (install via: pip install diffusers==0.31.0 transformers==4.46.3)
# - Scikit-Image (install via: pip install scikit-image==0.24.0)
# - Xformers (optional, for memory efficient attention) (install via: pip install xformers==0.0.28)
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
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler
)
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
from typing import List, Optional, Tuple
import gc
from skimage.metrics import structural_similarity as ssim

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.utils")

# Determine the device to use (GPU if available, else CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_arguments() -> argparse.Namespace:
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
        '-b',
        '--blend-mode',
        type=str,
        default='average',
        choices=['average', 'weighted'],
        help='Method to blend input images.'
    )
    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        help='Comma-separated weights for weighted blending (required if blend-mode is "weighted").'
    )
    parser.add_argument(
        '-n',
        '--num-steps',
        type=int,
        default=50,
        help='Number of inference steps for the model.'
    )
    parser.add_argument(
        '-g',
        '--guidance-scale',
        type=float,
        default=4,
        help='Guidance scale for the model. Controls how strongly the model follows the prompt.'
    )
    parser.add_argument(
        '-m',
        '--resolution-mode',
        type=str,
        default='middle',
        choices=['smallest', 'biggest', 'middle', 'custom'],
        help='Mode to determine output image resolution (default: middle).'
    )
    parser.add_argument(
        '-x',
        '--width',
        type=int,
        help='Custom width for the output image (required if resolution-mode is "custom").'
    )
    parser.add_argument(
        '-y',
        '--height',
        type=int,
        help='Custom height for the output image (required if resolution-mode is "custom").'
    )
    parser.add_argument(
        '-s',
        '--scheduler',
        type=str,
        default='ddim',
        choices=['ddim', 'plms', 'k_lms', 'euler', 'euler_a', 'heun', 'dpm_solver'],
        help='Scheduler (sampler) to use for image generation.'
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        default='stabilityai/stable-diffusion-xl-1.0',
        help='SDXL checkpoint to use. Can be a Hugging Face model ID or a local path.'
    )
    parser.add_argument(
        '-f',
        '--refiner',
        type=str,
        default='stabilityai/stable-diffusion-xl-refiner-1.0',
        help='SDXL refiner checkpoint to use. Can be a Hugging Face model ID or a local path.'
    )
    parser.add_argument(
        '-l',
        '--lora',
        type=str,
        action='append',
        help='LoRA model path. Can be specified multiple times for multiple LoRAs.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='merged_image_sdxl.png',
        help='Output file name for the merged image.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level).'
    )
    parser.add_argument(
        '-d',
        '--default-prompt',
        action='store_true',
        help='Use a default prompt instead of generating one via BLIP.'
    )
    args = parser.parse_args()

    # Validate custom resolution arguments
    if args.resolution_mode == 'custom':
        if args.width is None or args.height is None:
            parser.error('--width and --height must be specified when resolution-mode is "custom".')

    return args


def setup_logging(verbose: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def collect_images(input_path: str, recursive: bool) -> List[str]:
    """
    Collects all image files from the input path.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_files: List[str] = []

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


def determine_output_size(
    image_files: List[str],
    resolution_mode: str,
    custom_width: Optional[int] = None,
    custom_height: Optional[int] = None
) -> Tuple[int, int]:
    """
    Determines the output image size based on the resolution mode.
    """
    sizes: List[Tuple[int, int]] = []
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
        width = custom_width  # type: ignore
        height = custom_height  # type: ignore
    else:
        logging.error(f"Unknown resolution mode '{resolution_mode}'.")
        sys.exit(1)

    logging.info(f"Output image size set to: {width}x{height}")
    return width, height


def adjust_image_size(image: Image.Image, multiple: int = 64) -> Image.Image:
    """
    Adjusts the image size to the nearest multiple of the specified value.
    If the current size is not a multiple, it resizes the image accordingly.
    """
    width, height = image.size
    new_width = ((width + multiple - 1) // multiple) * multiple
    new_height = ((height + multiple - 1) // multiple) * multiple

    logging.debug(f"Original image size: ({width}, {height})")
    logging.debug(f"Calculated new size: ({new_width}, {new_height})")

    if (width, height) != (new_width, new_height):
        logging.debug(f"Resizing image from ({width}, {height}) to ({new_width}, {new_height}) to meet size requirements.")
        try:
            # Handle Pillow version compatibility
            if hasattr(Image, 'Resampling'):
                resampling_filter = Image.Resampling.LANCZOS
            else:
                resampling_filter = Image.LANCZOS
            image = image.resize((new_width, new_height), resampling_filter)
            logging.debug(f"Image resized to ({new_width}, {new_height}).")
        except Exception as e:
            logging.error(f"Failed to resize image: {e}")
            sys.exit(1)
    else:
        logging.debug(f"Image size ({width}, {height}) already meets the multiple of {multiple} requirement.")

    return image


def blend_images(
    image_files: List[str],
    blend_mode: str,
    weights: Optional[List[float]] = None
) -> Image.Image:
    """
    Blends multiple images into a single image using the specified blend mode.
    """
    logging.info(f"Blending images using '{blend_mode}' mode.")

    # Load images
    images: List[Image.Image] = []
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
    logging.debug(f"Target size for blending: {target_size}")
    resized_images: List[Image.Image] = [
        img.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        for img in images
    ]

    # Ensure all images have valid pixel values
    for idx, img in enumerate(resized_images):
        img_array = np.array(img)
        if not np.isfinite(img_array).all():
            logging.error(f"Image '{image_files[idx]}' contains invalid pixel values (NaN or Inf).")
            sys.exit(1)

    if blend_mode == 'average':
        blended_array = np.mean([np.array(img) for img in resized_images], axis=0)
    elif blend_mode == 'weighted':
        blended_array = np.zeros_like(np.array(resized_images[0]), dtype=np.float32)
        for img, weight in zip(resized_images, weights):  # type: ignore
            blended_array += np.array(img) * weight
    else:
        logging.error(f"Unknown blend mode '{blend_mode}'.")
        sys.exit(1)

    # Clip values to [0, 255] and convert to uint8
    blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)

    # Convert blended_array to PIL Image and ensure RGB mode
    blended_image = Image.fromarray(blended_array).convert("RGB")
    logging.info("Images blended successfully.")

    return blended_image


def compute_ssim(blended_image: Image.Image, input_images: List[Image.Image]) -> float:
    """
    Computes the average Structural Similarity Index (SSIM) between the blended image
    and each input image. Returns the average SSIM.
    """
    logging.info("Computing Structural Similarity Index (SSIM) between blended image and input images...")
    blended_array = np.array(blended_image)

    ssim_scores = []
    for idx, input_img in enumerate(input_images):
        input_array = np.array(input_img.resize(blended_image.size))
        # Compute SSIM for each channel and average
        try:
            ssim_score = ssim(
                blended_array,
                input_array,
                channel_axis=-1,
                data_range=blended_array.max() - blended_array.min()
            )
            ssim_scores.append(ssim_score)
            logging.debug(f"SSIM between blended image and input image {idx + 1}: {ssim_score:.4f}")
        except ValueError as ve:
            logging.error(f"Failed to compute SSIM between blended image and input image {idx + 1}: {ve}")
            sys.exit(1)

    average_ssim = np.mean(ssim_scores)
    logging.info(f"Average SSIM: {average_ssim:.4f}")
    return average_ssim


def determine_strength(average_ssim: float) -> float:
    """
    Determines the strength parameter based on the average SSIM score.
    """
    strength = 0.0

    # Define strength based on average SSIM
    # These thresholds can be adjusted based on experimentation
    if average_ssim >= 0.9:
        strength = 0.3
        logging.info("Very high similarity detected. Using very low strength to preserve input images.")
    elif average_ssim >= 0.8:
        strength = 0.6
        logging.info("High similarity detected. Using low strength to preserve input images.")
    elif average_ssim >= 0.7:
        strength = 0.9
        logging.info("Moderate similarity detected. Using moderate strength to preserve input images.")
    else:
        strength = 0.95
        logging.warning("Low similarity detected. Using higher strength to enhance and reduce artifacts.")
    return strength


def load_sdxl_model(
    scheduler_name: str,
    checkpoint: str,
    refiner_checkpoint: Optional[str],
    lora_paths: Optional[List[str]] = None
) -> Tuple[StableDiffusionXLImg2ImgPipeline, Optional[StableDiffusionXLImg2ImgPipeline]]:
    """
    Loads the Stable Diffusion XL (SDXL) img2img pipeline with the specified scheduler and LoRAs.
    """
    logging.info("Loading Stable Diffusion XL (SDXL) img2img model. This may take a while...")
    try:
        # Determine if checkpoint is a local path or a Hugging Face model ID
        if os.path.exists(checkpoint):
            model_id = checkpoint
            logging.info(f"Loading SDXL model from local path: {model_id}")
        else:
            model_id = checkpoint
            logging.info(f"Loading SDXL model from Hugging Face Model Hub: {model_id}")

        # Load the img2img pipeline with appropriate torch_dtype
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        pipe = pipe.to(DEVICE)  # Move pipeline to the determined device

        # Disable the safety checker
        pipe.safety_checker = lambda images, **kwargs: (images, False)

        # Enable attention slicing to reduce VRAM usage
        pipe.enable_attention_slicing()

        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention.")
        except Exception as e:
            logging.warning(f"Failed to enable xformers memory efficient attention: {e}")

        # Enable sequential CPU offloading for better memory management
        try:
            pipe.enable_sequential_cpu_offload()
            logging.info("Enabled sequential model CPU offloading.")
        except Exception as e:
            logging.warning(f"Failed to enable sequential CPU offloading: {e}")

        # Replace the scheduler based on user input
        scheduler_mapping = {
            'ddim': DDIMScheduler.from_config(pipe.scheduler.config),
            'plms': PNDMScheduler.from_config(pipe.scheduler.config),
            'k_lms': LMSDiscreteScheduler.from_config(pipe.scheduler.config),
            'euler': EulerDiscreteScheduler.from_config(pipe.scheduler.config),
            'euler_a': EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
            'heun': HeunDiscreteScheduler.from_config(pipe.scheduler.config),
            'dpm_solver': DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
        }

        if scheduler_name not in scheduler_mapping:
            logging.error(f"Unsupported scheduler '{scheduler_name}'.")
            sys.exit(1)

        pipe.scheduler = scheduler_mapping[scheduler_name]
        logging.info(f"Scheduler set to '{scheduler_name}'.")

        # Load and apply LoRAs if specified
        if lora_paths:
            for lora_path in lora_paths:
                if os.path.exists(lora_path):
                    try:
                        pipe.load_lora_weights(lora_path)
                        logging.info(f"Loaded LoRA weights from '{lora_path}'.")
                    except Exception as e:
                        logging.error(f"Failed to load LoRA from '{lora_path}': {e}")
                        sys.exit(1)
                else:
                    logging.error(f"LoRA path '{lora_path}' does not exist.")
                    sys.exit(1)

        # Load refiner model if specified
        refiner_pipe = None
        if refiner_checkpoint:
            logging.info("Loading SDXL refiner model. This may take additional time...")
            try:
                if os.path.exists(refiner_checkpoint):
                    refiner_model_id = refiner_checkpoint
                    logging.info(f"Loading SDXL refiner from local path: {refiner_model_id}")
                else:
                    refiner_model_id = refiner_checkpoint
                    logging.info(f"Loading SDXL refiner from Hugging Face Model Hub: {refiner_model_id}")

                refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_model_id,
                    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                )
                refiner_pipe = refiner_pipe.to(DEVICE)

                # Disable the safety checker
                refiner_pipe.safety_checker = lambda images, **kwargs: (images, False)

                # Enable attention slicing
                refiner_pipe.enable_attention_slicing()

                try:
                    refiner_pipe.enable_xformers_memory_efficient_attention()
                    logging.info("Enabled xformers memory efficient attention for refiner.")
                except Exception as e:
                    logging.warning(f"Failed to enable xformers memory efficient attention for refiner: {e}")

                # Enable sequential CPU offloading
                try:
                    refiner_pipe.enable_sequential_cpu_offload()
                    logging.info("Enabled sequential model CPU offloading for refiner.")
                except Exception as e:
                    logging.warning(f"Failed to enable sequential CPU offloading for refiner: {e}")

                # Set the scheduler for the refiner
                refiner_pipe.scheduler = scheduler_mapping[scheduler_name]
                logging.info(f"Refiner scheduler set to '{scheduler_name}'.")

            except Exception as e:
                logging.exception("Failed to load SDXL refiner model.")
                sys.exit(1)

        logging.info(f"Stable Diffusion XL img2img model loaded successfully on {DEVICE}.")
        if refiner_pipe:
            logging.info(f"Refiner model loaded successfully on {DEVICE}.")

        return pipe, refiner_pipe
    except Exception as e:
        logging.exception("Failed to load Stable Diffusion XL img2img model.")
        sys.exit(1)


def load_captioning_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Loads the BLIP image captioning model for automated prompt generation.
    """
    logging.info("Loading BLIP image captioning model for prompt generation...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to('cpu')  # Keep BLIP on CPU to save GPU memory
        logging.info("BLIP model loaded successfully on CPU.")
        return processor, model
    except Exception as e:
        logging.exception("Failed to load BLIP model.")
        sys.exit(1)


def generate_prompt(
    blended_image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    use_default: bool = False
) -> str:
    """
    Generates a descriptive prompt based on the blended image using an image captioning model.
    If use_default is True, returns a predefined prompt.
    """
    if use_default:
        logging.info("Using default prompt instead of generating one via BLIP.")
        return "A seamlessly merged image combining multiple elements."

    logging.info("Generating detailed prompt from the blended image using BLIP...")
    try:
        inputs = processor(images=blended_image, return_tensors="pt").to('cpu')  # Ensure inputs are on CPU
        logging.debug("Moved BLIP inputs to CPU.")
        with torch.no_grad():
            # Generate multiple captions with different sampling parameters
            out1 = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2)
            out2 = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)

        caption1 = processor.decode(out1[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        caption2 = processor.decode(out2[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Combine the two captions for a more detailed prompt
        combined_prompt = f"{caption1}, {caption2}"

        # Clean up any redundant whitespace
        combined_prompt = ' '.join(combined_prompt.split())

        if not combined_prompt.strip():
            logging.warning("BLIP failed to generate a detailed caption. Using a default prompt.")
            combined_prompt = "A seamlessly merged image combining multiple elements."

        logging.info(f"Generated Detailed Prompt: {combined_prompt}")
        return combined_prompt
    except Exception as e:
        logging.exception("Failed to generate detailed caption for blended image.")
        sys.exit(1)


def generate_merged_image(
    base_pipe: StableDiffusionXLImg2ImgPipeline,
    refiner_pipe: Optional[StableDiffusionXLImg2ImgPipeline],
    blended_image: Image.Image,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    strength: float
) -> Image.Image:
    """
    Generates a merged image using the Stable Diffusion XL img2img pipeline, optionally refining it.
    """
    logging.info("Generating merged image using Stable Diffusion XL img2img pipeline...")
    try:
        # Preprocess the blended image
        init_image = blended_image.resize((width, height))
        init_image = init_image.convert("RGB")

        # Debugging: Check type and mode
        logging.debug(f"init_image type: {type(init_image)}, mode: {init_image.mode}, size: {init_image.size}")

        # Use torch.no_grad() to prevent gradient computation and save memory
        with torch.no_grad():
            # Generate the image using the base pipeline
            result = base_pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps
            )

        if not result.images:
            logging.error("No images were returned by the base pipeline.")
            sys.exit(1)

        merged_image = result.images[0]

        # Validate the merged image for NaN or Inf values
        merged_array = np.array(merged_image)
        if not np.isfinite(merged_array).all():
            logging.error("Merged image contains invalid pixel values (NaN or Inf).")
            sys.exit(1)

        # Additional Logging: Check image statistics
        logging.debug(f"Merged image array stats: min={merged_array.min()}, max={merged_array.max()}, mean={merged_array.mean()}, std={merged_array.std()}")

        logging.info("Merged image generated successfully with base model.")

        # If refiner pipeline is provided, refine the image
        if refiner_pipe:
            logging.info("Refining the merged image using SDXL refiner...")
            with torch.no_grad():
                # Generate the refined image
                refined_result = refiner_pipe(
                    prompt=prompt,
                    image=merged_image,
                    strength=0,  # Strength should be 0 for the refiner
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps
                )

            if not refined_result.images:
                logging.error("No images were returned by the refiner pipeline.")
                sys.exit(1)

            merged_image = refined_result.images[0]

            # Validate the refined image for NaN or Inf values
            refined_array = np.array(merged_image)
            if not np.isfinite(refined_array).all():
                logging.error("Refined image contains invalid pixel values (NaN or Inf).")
                sys.exit(1)

            logging.debug(f"Refined image array stats: min={refined_array.min()}, max={refined_array.max()}, mean={refined_array.mean()}, std={refined_array.std()}")

            logging.info("Merged image refined successfully.")

        return merged_image
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA Out of Memory Error: Failed to generate merged image due to insufficient GPU memory.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Failed to generate merged image.")
        sys.exit(1)


def save_image(image: Image.Image, output_path: str) -> None:
    """
    Saves the PIL image to the specified path.
    """
    try:
        image.save(output_path)
        logging.info(f"Merged image saved as '{output_path}'.")
    except Exception as e:
        logging.exception(f"Failed to save merged image '{output_path}'.")
        sys.exit(1)


def prepare_input_images(image_files: List[str], target_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Loads and resizes input images for SSIM computation.
    """
    input_images: List[Image.Image] = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB').resize(target_size)
            input_images.append(img)
            logging.debug(f"Loaded image for SSIM computation: '{img_path}'")
        except Exception as e:
            logging.error(f"Failed to open image '{img_path}' for SSIM computation: {e}")
            sys.exit(1)
    return input_images


def cleanup_memory():
    """
    Frees up memory by deleting unused objects and clearing caches.
    """
    gc.collect()
    torch.cuda.empty_cache()
    logging.debug("Freed memory after cleanup.")


def main() -> None:
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    logging.info(f"Using device: {DEVICE}")

    # Collect image files
    image_files: List[str] = collect_images(args.input_path, args.recursive)

    # Validate and parse weights if using weighted blending
    if args.blend_mode == 'weighted':
        if args.weights is None:
            logging.error('--weights must be specified when blend-mode is "weighted".')
            sys.exit(1)
        try:
            weights_list: List[float] = [float(w) for w in args.weights.split(',')]
            if len(weights_list) != len(image_files):
                logging.error(f'Number of weights ({len(weights_list)}) does not match number of input images ({len(image_files)}).')
                sys.exit(1)
            total_weight = sum(weights_list)
            if not np.isclose(total_weight, 1.0):
                logging.warning(f'Weights sum to {total_weight}, which is not close to 1.0. Normalizing weights.')
                weights_list = [w / total_weight for w in weights_list]
            logging.info(f'Using weights: {weights_list}')
            weights = weights_list
        except ValueError:
            logging.error('Weights must be a comma-separated list of numbers.')
            sys.exit(1)
    else:
        weights = None

    # Determine output image size
    output_width, output_height = determine_output_size(
        image_files,
        args.resolution_mode,
        custom_width=args.width,
        custom_height=args.height
    )

    # Blend images
    blended_image: Image.Image = blend_images(
        image_files,
        args.blend_mode,
        weights=weights
    )

    # Adjust image size to be multiples of 64
    blended_image = adjust_image_size(blended_image, multiple=64)

    # Confirm the image has been resized
    logging.debug(f"Blended image size after adjustment: {blended_image.size}")
    if blended_image.size[0] % 64 != 0 or blended_image.size[1] % 64 != 0:
        logging.error(f"Blended image size {blended_image.size} is not a multiple of 64.")
        sys.exit(1)
    else:
        logging.info(f"Blended image size {blended_image.size} meets the multiple of 64 requirement.")

    # Prepare input images for SSIM computation
    input_images: List[Image.Image] = prepare_input_images(image_files, blended_image.size)

    # Compute SSIM similarity
    average_ssim: float = compute_ssim(blended_image, input_images)

    # Free memory after SSIM computation
    del input_images
    cleanup_memory()

    # Determine strength based on SSIM
    strength: float = determine_strength(average_ssim)

    # Load BLIP model and generate prompt from blended image
    processor, caption_model = load_captioning_model()
    prompt: str = generate_prompt(blended_image, processor, caption_model, args.default_prompt)

    # Free BLIP model memory
    del processor
    del caption_model
    cleanup_memory()

    # Ensure the prompt is non-empty
    if not prompt.strip():
        logging.warning("Generated prompt is empty. Using a default prompt.")
        prompt = "A seamlessly merged image combining multiple elements."

    # Load SDXL model with the selected scheduler, refiner, and LoRAs
    base_pipe, refiner_pipe = load_sdxl_model(args.scheduler, args.checkpoint, args.refiner, args.lora)

    # Generate merged image using img2img, optionally refining it
    merged_image: Image.Image = generate_merged_image(
        base_pipe,
        refiner_pipe,
        blended_image,
        prompt=prompt,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        width=output_width,
        height=output_height,
        strength=strength
    )

    # Save the merged image
    save_image(merged_image, args.output)


if __name__ == '__main__':
    main()
