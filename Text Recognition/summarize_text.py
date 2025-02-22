#!/usr/bin/env python3

# -------------------------------------------------------
# Script: summarize_text.py
#
# Description:
# This script takes an input file containing text, extracts its
# content using the appropriate method, and sends it to the GPT
# API to generate a summary.
#
# Usage:
#   ./summarize_text.py [options] <input_file>
#
# Options:
#   -k, --api-key API_KEY         Your OpenAI API key (or set via OPENAI_API_KEY).
#   -m, --model MODEL             OpenAI model to use (default: "gpt-4o").
#   -c, --chunk-size TOKENS       Maximum tokens per request chunk (default: 500).
#   -R, --max-requests N          Maximum number of GPT requests (default: 0 for unlimited).
#   -M, --max-input-size TOKENS   Maximum total tokens to process (default: 10000).
#   -d, --detail-level LEVEL      Summary detail level: brief, medium, or detailed (default: medium).
#   -f, --force-format FORMAT     Force input file format (e.g., txt, pdf, docx).
#   -s, --min-string-length N     Minimum length for a string segment (default: 4).
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - openai (install via: pip install openai==1.63.2)
#   - PyPDF2 (install via: pip install PyPDF2==3.0.1)
#   - python-docx (install via: pip install python-docx==1.1.2)
#   - Pillow (install via: pip install Pillow==11.1.0)
#   - pytesseract (install via: pip install pytesseract==0.3.13)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import gzip
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, List

import openai


DETAIL_LEVEL_PROMPTS = {
    "brief": "Provide a brief summary of the text I will provide you in my next messages.",
    "medium": "Provide a summary of the text I will provide you in my next messages. The length and detail should be well-balanced.",
    "detailed": "Provide a detailed and analytical summary of the text I will provide you in my next messages.",
}


@dataclass
class SummaryConfig:
    """Configuration for summarization."""

    chunk_size: int
    max_requests: Optional[int]
    max_input_size: int
    detail_level: str
    model: str


def get_text_prompt(detail_level: str) -> str:
    """
    Returns the text prompt based on the desired detail level.
    """
    return DETAIL_LEVEL_PROMPTS.get(detail_level, DETAIL_LEVEL_PROMPTS["medium"])


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns the namespace.
    """
    parser = argparse.ArgumentParser(
        description="Summarize text content using GPT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file (text, pdf, docx, or image) to summarize.",
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
        "-c",
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum tokens per request chunk (default: 500).",
    )
    parser.add_argument(
        "-R",
        "--max-requests",
        type=int,
        default=0,
        help="Maximum number of GPT requests (default: 0 for unlimited).",
    )
    parser.add_argument(
        "-M",
        "--max-input-size",
        type=int,
        default=10000,
        help="Maximum total tokens to process (default: 10000).",
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
        "-f",
        "--force-format",
        type=str,
        help="Force input file format (e.g., txt, pdf, docx, image).",
    )
    parser.add_argument(
        "-s",
        "--min-string-length",
        type=int,
        default=4,
        help="Minimum length for a string segment (default: 4).",
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


def determine_file_format(file_path: str, forced_format: Optional[str]) -> str:
    """
    Determines the file format based on forced format or file extension.
    """
    if forced_format:
        return forced_format.lower()
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".txt":
        return "txt"
    elif ext == ".pdf":
        return "pdf"
    elif ext == ".docx":
        return "docx"
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
        return "image"
    else:
        return "unknown"


def extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a plain text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDF2.
    """
    from PyPDF2 import PdfReader

    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a Word (.docx) document.
    """
    import docx

    document = docx.Document(file_path)
    paragraphs = [para.text for para in document.paragraphs]
    return "\n".join(paragraphs)


def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from an image file using OCR (pytesseract).
    """
    from PIL import Image
    import pytesseract

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error processing image for OCR: {e}")
        sys.exit(1)


def extract_text_from_unknown(file_path: str, min_string_length: int) -> str:
    """
    Extracts printable text segments from an unknown file type, similar to the 'strings' command.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    pattern = re.compile(rb"[ -~]{" + str(min_string_length).encode("ascii") + rb",}")
    matches = pattern.findall(data)
    return "\n".join(match.decode("utf-8", errors="replace") for match in matches)


def extract_content(
    file_path: str, forced_format: Optional[str], min_string_length: int
) -> str:
    """
    Extracts text content from the given file based on its type.
    If the file ends with '.gz', it is decompressed and processed.
    """
    if file_path.endswith(".gz"):
        temp_file = decompress_gz(file_path)
        try:
            return extract_content(temp_file, forced_format, min_string_length)
        finally:
            try:
                os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Could not remove temporary file '{temp_file}': {e}")

    file_format = determine_file_format(file_path, forced_format)
    logging.debug(f"Determined file format: {file_format}")
    if file_format == "txt":
        return extract_text_from_txt(file_path)
    elif file_format == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_format == "docx":
        return extract_text_from_docx(file_path)
    elif file_format == "image":
        return extract_text_from_image(file_path)
    elif file_format == "unknown":
        return extract_text_from_unknown(file_path, min_string_length)
    else:
        return extract_text_from_txt(file_path)


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text using whitespace.
    """
    return text.split()


def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Splits text into chunks of approximately the specified token size.
    """
    tokens = tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def summarize_text_chunk(
    chunk: str,
    config: SummaryConfig,
    api_key: str,
    history: List[dict],
) -> str:
    """
    Sends a text chunk to the GPT API for summarization,
    including the conversation history (previous chunks and summaries).
    """
    history.append({"role": "user", "content": chunk})
    try:
        response = openai.ChatCompletion.create(
            model=config.model,
            messages=history,
            api_key=api_key,
        )
        summary = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": summary})
        return summary
    except Exception as e:
        logging.error(f"Error during GPT API call for text chunk: {e}")
        summary = "Error summarizing chunk."
        history.append({"role": "assistant", "content": summary})
        return summary


def process_text_summary(content: str, config: SummaryConfig, api_key: str) -> None:
    """
    Processes text content: tokenizes, splits into chunks, and summarizes each chunk.
    The previous conversation history (chunks and summaries) is included for every new summary.
    """
    tokens = tokenize(content)
    logging.debug(f"Total tokens in input: {len(tokens)}")
    if len(tokens) > config.max_input_size:
        logging.info(
            f"Input exceeds maximum size of {config.max_input_size} tokens. Truncating."
        )
        tokens = tokens[: config.max_input_size]
        content = " ".join(tokens)
    chunks = split_text_into_chunks(content, config.chunk_size)
    logging.info(f"Split content into {len(chunks)} chunk(s).")
    if config.max_requests is not None:
        chunks = chunks[: config.max_requests]

    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes text.",
        },
        {
            "role": "user",
            "content": get_text_prompt(config.detail_level)
            + "\nThe complete text is split into chunks for summarization. "
            + "When you create a summary for the latest chunk, you should consider the context of the previous summaries.",
        },
        {
            "role": "assistant",
            "content": "Understood. I will provide a summary for each chunk in the context of the previous chunks.",
        },
    ]

    for i, chunk in enumerate(chunks, start=1):
        logging.info(f"Summarizing chunk {i}/{len(chunks)}...")
        summary = summarize_text_chunk(chunk, config, api_key, conversation_history)
        print(summary)


def main() -> None:
    """
    Orchestrates the parsing of arguments, content extraction, and summarization.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error(
            "API key not provided via command-line or OPENAI_API_KEY environment variable."
        )
        sys.exit(1)

    config = SummaryConfig(
        chunk_size=args.chunk_size,
        max_requests=args.max_requests if args.max_requests > 0 else None,
        max_input_size=args.max_input_size,
        detail_level=args.detail_level,
        model=args.model,
    )

    content = extract_content(
        args.input_file, args.force_format, args.min_string_length
    )
    process_text_summary(content, config, api_key)


if __name__ == "__main__":
    main()
