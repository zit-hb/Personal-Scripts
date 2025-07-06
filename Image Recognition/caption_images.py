#!/usr/bin/env python3

# -------------------------------------------------------
# Script: caption_images.py
#
# Description:
# This script takes one or more input image files (or a directory of images),
# processes each image (including decompressing .gz and converting to JPEG),
# sends it to the GPT API to generate a list of 20 short, fitting descriptions
# (comma-separated), and saves the captions to a .txt file with the same base
# name as the image. By default, the .txt files are created alongside the input
# images, but you can specify a different output directory.
#
# Usage:
#   ./caption_images.py [options] <input_path>
#
# Options:
#   -k, --api-key API_KEY         Your OpenAI API key (or set via OPENAI_API_KEY).
#   -m, --model MODEL             OpenAI model to use (default: "gpt-4o").
#   -o, --output-dir DIR          Directory to write .txt caption files (default: same as each image).
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - openai (install via: pip install openai==1.63.2)
#   - Pillow (install via: pip install Pillow==11.1.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import base64
import gzip
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List

from openai import OpenAI
from PIL import Image


# Prompt instructing GPT to produce 20 comma-separated descriptions
CAPTION_PROMPT = (
    "Generate a list of 20 short, fitting descriptions for the content of the attached image. "
    "Return only the descriptions as a comma-separated list. Don't add a dot at the end."
)

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class CaptionConfig:
    """Configuration for image captioning."""

    model: str


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns the namespace.
    """
    parser = argparse.ArgumentParser(
        description="Caption images by generating 20 keyword-style descriptions using GPT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="Your OpenAI API key (or set via OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        help='OpenAI model to use (default: "gpt-4o").',
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Directory to write .txt caption files (default: same as each image).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level).",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configures logging level and format based on flags.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def decompress_gz(file_path: str) -> str:
    """
    Decompresses a .gz file to a temporary file with the underlying extension.
    Returns the path to the temporary decompressed file.
    """
    base_name = os.path.basename(file_path[:-3])
    _, ext = os.path.splitext(base_name)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    with gzip.open(file_path, "rb") as gz_file:
        data = gz_file.read()
    with open(temp_file.name, "wb") as out_file:
        out_file.write(data)
    return temp_file.name


def process_image_to_jpg(file_path: str) -> str:
    """
    Converts any image to RGB JPEG, stripping metadata.
    Returns the path to the temporary JPEG file.
    """
    with Image.open(file_path) as img:
        img_converted = img.convert("RGB")
        new_img = Image.new("RGB", img_converted.size)
        new_img.putdata(list(img_converted.getdata()))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        new_img.save(temp_file.name, format="JPEG")
        return temp_file.name


def extract_content(file_path: str) -> str:
    """
    Handles .gz decompression recursively, then converts to JPEG.
    Returns the path to the processed image file.
    """
    if file_path.endswith(".gz"):
        temp = decompress_gz(file_path)
        try:
            return extract_content(temp)
        finally:
            try:
                os.remove(temp)
            except Exception as e:
                logging.warning(f"Could not remove temporary file '{temp}': {e}")
    return process_image_to_jpg(file_path)


def caption_image(image_path: str, config: CaptionConfig, client: OpenAI) -> str:
    """
    Sends the image to the GPT API with a captioning prompt
    and returns the comma-separated list of descriptions.
    """
    system_prompt = (
        "You are a helpful assistant that generates keyword-style image descriptions."
    )
    user_content = [
        {"type": "text", "text": CAPTION_PROMPT},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error during GPT API call for image captioning: {e}")
        return ""


def write_captions(captions: str, image_file: str, output_dir: str = None) -> None:
    """
    Writes the captions string to a .txt file matching the image base name.
    Creates output directory if needed.
    """
    base, _ = os.path.splitext(os.path.basename(image_file))
    txt_name = f"{base}.txt"
    target_dir = output_dir or os.path.dirname(image_file)
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, txt_name)
    with open(out_path, "w") as f:
        f.write(captions)
    logging.info(f"Wrote captions to {out_path}")


def process_file(
    file_path: str, config: CaptionConfig, client: OpenAI, output_dir: str = None
) -> None:
    """
    Orchestrates processing of a single image:
    decompress/convert, caption via GPT, write output, and cleanup.
    """
    logging.info(f"Processing {file_path}...")
    tmp_img = extract_content(file_path)
    captions = caption_image(tmp_img, config, client)
    if captions:
        write_captions(captions, file_path, output_dir)
    else:
        logging.error(f"Failed to generate captions for {file_path}")
    try:
        os.remove(tmp_img)
    except Exception:
        pass


def process_directory(
    dir_path: str, config: CaptionConfig, client: OpenAI, output_dir: str = None
) -> None:
    """
    Iterates over supported image files in a directory and processes each.
    """
    for entry in os.listdir(dir_path):
        fp = os.path.join(dir_path, entry)
        if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in IMAGE_EXTENSIONS:
            process_file(fp, config, client, output_dir)


def main() -> None:
    """
    Parses arguments, sets up logging and OpenAI client,
    then processes either a single image or all in a directory.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error(
            "API key not provided via command-line or OPENAI_API_KEY environment variable."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    config = CaptionConfig(model=args.model)

    if os.path.isdir(args.input_path):
        process_directory(args.input_path, config, client, args.output_dir)
    elif os.path.isfile(args.input_path):
        process_file(args.input_path, config, client, args.output_dir)
    else:
        logging.error(f"Input path '{args.input_path}' is not a file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
