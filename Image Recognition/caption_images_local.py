#!/usr/bin/env python3

# -------------------------------------------------------
# Script: caption_images_local.py
#
# Description:
# This script takes one or more input image files (or a directory of images),
# reads each file’s content as base64, sends it to a local Ollama model to generate a list of 20 short, fitting descriptions
# (comma-separated) between <answer> tags, and saves the captions to a .txt file with the same base
# name as the image. By default, the .txt files are created alongside the input
# images, but you can specify a different output directory.
#
# Usage:
#   ./caption_images_local.py [options] <input_path>
#
# Options:
#   -H, --host HOST_URL          Base URL of the Ollama server (default: http://localhost:11434).
#   -m, --model MODEL            Ollama model to use (default: "remyxai/spaceom").
#   -o, --output-dir DIR         Directory to write .txt caption files (default: same as each image).
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - requests (install via: pip install requests==2.31.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import base64
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import requests

# Prompt instructing the model to produce 20 comma-separated descriptions
CAPTION_PROMPT = (
    "Generate a list of 20 unique, concise descriptions of visible elements in the attached image. "
    "Each description should be short (1-3 words), non-repetitive, and focused strictly on what is directly observable "
    "(e.g., colors, objects, actions, and environment). Avoid any inference beyond the image content. "
    "Return only the descriptions as a comma-separated list, without any trailing punctuation."
)

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class CaptionConfig:
    """Configuration for Ollama-based image captioning."""

    model: str
    host: str


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns the populated namespace.
    """
    parser = argparse.ArgumentParser(
        description="Caption images using a local Ollama model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Base URL of the Ollama server (default: http://localhost:11434).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="remyxai/spaceom",
        help='Ollama model to use (default: "remyxai/spaceom").',
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
    Configures logging level and format based on verbose/debug flags.
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


def caption_image(image_path: str, config: CaptionConfig) -> str:
    """
    Reads the image file as base64, sends it to Ollama with the caption prompt,
    extracts the text between <answer>...</answer>, and returns it.
    """
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    url = f"{config.host}/api/generate"
    payload = {
        "model": config.model,
        "prompt": CAPTION_PROMPT,
        "images": [b64_str],
        "stream": False,
    }
    try:
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        raw = data.get("response", "")
        match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()
    except Exception as e:
        logging.error(f"Error during Ollama API call for image captioning: {e}")
        return ""


def write_captions(
    captions: str, image_file: str, output_dir: Optional[str] = None
) -> None:
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
    file_path: str,
    config: CaptionConfig,
    output_dir: Optional[str] = None,
) -> None:
    """
    Orchestrates sending a single image for captioning and writing output.
    """
    logging.info(f"Processing {file_path}...")
    captions = caption_image(file_path, config)
    if captions:
        write_captions(captions, file_path, output_dir)
    else:
        logging.error(f"Failed to generate captions for {file_path}")


def process_directory(
    dir_path: str,
    config: CaptionConfig,
    output_dir: Optional[str] = None,
) -> None:
    """
    Iterates over supported image files in a directory and processes each.
    """
    for entry in os.listdir(dir_path):
        fp = os.path.join(dir_path, entry)
        if os.path.isfile(fp) and os.path.splitext(fp)[1].lower() in IMAGE_EXTENSIONS:
            process_file(fp, config, output_dir)


def main() -> None:
    """
    Parses arguments, sets up logging and configuration,
    then processes either a single image or all in a directory.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    config = CaptionConfig(model=args.model, host=args.host)

    if os.path.isdir(args.input_path):
        process_directory(args.input_path, config, args.output_dir)
    elif os.path.isfile(args.input_path):
        process_file(args.input_path, config, args.output_dir)
    else:
        logging.error(f"Input path '{args.input_path}' is not a file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
