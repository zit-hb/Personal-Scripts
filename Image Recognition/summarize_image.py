#!/usr/bin/env python3

# -------------------------------------------------------
# Script: summarize_image.py
#
# Description:
# This script takes an input image file, normalizes the image,
# and sends it to the GPT API to generate a summary.
#
# Usage:
#   ./summarize_image.py [options] <input_file>
#
# Options:
#   -k, --api-key API_KEY         Your OpenAI API key (or set via OPENAI_API_KEY).
#   -m, --model MODEL             OpenAI model to use (default: "gpt-4o").
#   -d, --detail-level LEVEL      Summary detail level: brief, medium, or detailed (default: medium).
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

from openai import OpenAI
from PIL import Image


DETAIL_LEVEL_PROMPTS = {
    "brief": "Provide a brief summary of the image content.",
    "medium": "Provide a concise yet informative summary of the image content, capturing its key elements.",
    "detailed": "Provide a detailed and analytical summary of the image, describing its key elements, context, and any notable details.",
}


@dataclass
class SummaryConfig:
    """Configuration for image summarization."""

    detail_level: str
    model: str


def get_image_prompt(detail_level: str) -> str:
    """
    Returns the image prompt based on the desired detail level.
    """
    return DETAIL_LEVEL_PROMPTS.get(detail_level, DETAIL_LEVEL_PROMPTS["medium"])


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns the namespace.
    """
    parser = argparse.ArgumentParser(
        description="Summarize image content using GPT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input image file to summarize."
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
        "-d",
        "--detail-level",
        type=str,
        choices=["brief", "medium", "detailed"],
        default="medium",
        help="Summary detail level (default: medium).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration.
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
    Decompresses a .gz file to a temporary file with the underlying file extension.
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
    Processes an image file by converting it to JPG, ensuring RGB mode and removing metadata.
    """
    with Image.open(file_path) as img:
        img_converted = img.convert("RGB")
        new_img = Image.new("RGB", img_converted.size)
        new_img.putdata(list(img_converted.getdata()))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file_path = temp_file.name
        temp_file.close()
        new_img.save(temp_file_path, format="JPEG")
        return temp_file_path


def extract_content(file_path: str) -> str:
    """
    Extracts content from the given image file.
    If the file ends with '.gz', it is decompressed and processed.
    """
    if file_path.endswith(".gz"):
        temp_file = decompress_gz(file_path)
        try:
            return extract_content(temp_file)
        finally:
            try:
                os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Could not remove temporary file '{temp_file}': {e}")

    return process_image_to_jpg(file_path)


def summarize_image(
    image_path: str, detail_level: str, config: SummaryConfig, client: OpenAI
) -> str:
    """
    Sends an image to the GPT API for summarization using image recognition.
    """
    prompt_prefix = get_image_prompt(detail_level)
    system_prompt = (
        "You are a helpful assistant that summarizes the content of an image."
    )
    user_prompt = (
        f"{prompt_prefix}\n\nPlease analyze the attached image and provide a summary."
    )
    try:
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")

        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logging.error(f"Error during GPT API call for image summarization: {e}")
        return "Error summarizing image."


def process_image_summary(
    image_path: str, config: SummaryConfig, client: OpenAI
) -> None:
    """
    Processes image content by sending it to the GPT API for summarization.
    """
    logging.info("Processing image for summarization...")
    summary = summarize_image(image_path, config.detail_level, config, client)
    print(summary)
    try:
        os.remove(image_path)
    except Exception as e:
        logging.warning(f"Could not remove temporary image file: {e}")


def main() -> None:
    """
    Orchestrates the parsing of arguments, content extraction, and image summarization.
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

    config = SummaryConfig(
        detail_level=args.detail_level,
        model=args.model,
    )

    image_file = extract_content(args.input_file)
    process_image_summary(image_file, config, client)


if __name__ == "__main__":
    main()
