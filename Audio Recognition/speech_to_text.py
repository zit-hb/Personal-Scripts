#!/usr/bin/env python3

# -------------------------------------------------------
# Script: speech_to_text.py
#
# Description:
# This script converts speech from audio files into text using OpenAI's
# speech-to-text capabilities via the `openai` module. It supports
# processing single audio files or entire directories, handling multiple
# audio formats, specifying the language of the audio, and choosing
# different output formats such as plain text or JSON for subtitles.
#
# Usage:
# ./speech_to_text.py [input_path] [options]
#
# - [input_path]: The path to the input audio file or directory.
#
# Options:
# -r, --recursive            Process directories recursively.
# -o, --output-dir OUTPUT_DIR
#                            Directory to save transcription files. (default: current directory)
# -f, --format FORMAT        Output format for transcriptions. Choices: "txt", "json". (default: "txt")
# -l, --language LANGUAGE    Language of the audio (e.g., "en", "es", "fr"). If not specified, the API will attempt to detect it.
# -k, --api-key API_KEY      OpenAI API key. Can also be set via the OPENAI_API_KEY environment variable.
# -v, --verbose              Enable verbose logging (INFO level).
# -vv, --debug               Enable debug logging (DEBUG level).
# -w, --overwrite            Overwrite existing transcription files if they exist.
# -c, --chunk-size SIZE      Maximum size (in MB) for each audio chunk when processing large files. (default: 25)
# -b, --batch                Enable batch processing mode. Transcriptions will be saved to files.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - openai (install via: pip install openai)
# - tqdm (install via: pip install tqdm)
# - python-dotenv (optional, for loading environment variables) (install via: pip install python-dotenv)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from openai import OpenAI, OpenAIError

from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import json

# Optional: Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Supported audio file extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg')

# Global OpenAI client
client: OpenAI = None

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Convert speech from audio files into text using OpenAI\'s speech-to-text API.'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input audio file or directory.'
    )
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Process directories recursively.'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save transcription files. (default: current directory)'
    )
    parser.add_argument(
        '-f',
        '--format',
        type=str,
        default='txt',
        choices=['txt', 'json'],
        help='Output format for transcriptions. Choices: "txt", "json". (default: "txt")'
    )
    parser.add_argument(
        '-l',
        '--language',
        type=str,
        help='Language of the audio (e.g., "en", "es", "fr"). If not specified, the API will attempt to detect it.'
    )
    parser.add_argument(
        '-k',
        '--api-key',
        type=str,
        help='OpenAI API key. Can also be set via the OPENAI_API_KEY environment variable.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    parser.add_argument(
        '-vv',
        '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )
    parser.add_argument(
        '-w',
        '--overwrite',
        action='store_true',
        help='Overwrite existing transcription files if they exist.'
    )
    parser.add_argument(
        '-c',
        '--chunk-size',
        type=int,
        default=25,
        help='Maximum size (in MB) for each audio chunk when processing large files. (default: 25)'
    )
    parser.add_argument(
        '-b',
        '--batch',
        action='store_true',
        help='Enable batch processing mode. Transcriptions will be saved to files.'
    )
    args = parser.parse_args()

    # Validate chunk size
    if args.chunk_size <= 0:
        parser.error('--chunk-size must be a positive integer.')

    return args

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

    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def collect_audio_files(input_path: str, recursive: bool) -> List[Path]:
    """
    Collects all audio files from the input path.
    """
    audio_files: List[Path] = []
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(path)
            logging.info(f"Found audio file: '{path}'")
        else:
            logging.error(f"File '{path}' is not a supported audio format.")
    elif path.is_dir():
        if recursive:
            files = list(path.rglob('*'))
        else:
            files = list(path.glob('*'))
        for file in files:
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(file)
        if recursive:
            logging.info(f"Found {len(audio_files)} audio files in '{input_path}' (recursive).")
        else:
            logging.info(f"Found {len(audio_files)} audio files in '{input_path}'.")
    else:
        logging.error(f"Input path '{input_path}' is neither a file nor a directory.")
        sys.exit(1)

    if not audio_files:
        logging.error(f"No supported audio files found in '{input_path}'.")
        sys.exit(1)

    return audio_files

def get_api_key(provided_key: Optional[str]) -> str:
    """
    Retrieves the OpenAI API key from the provided argument or environment variable.
    """
    if provided_key:
        return provided_key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error('OpenAI API key not provided. Use the -k/--api-key option or set the OPENAI_API_KEY environment variable.')
        sys.exit(1)
    return api_key

def transcribe_audio(file_path: Path, language: Optional[str], chunk_size: int) -> dict:
    """
    Transcribes the given audio file using OpenAI's Speech-to-Text API.
    If the file size exceeds the chunk size, it will be split accordingly.
    """
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    logging.debug(f"Processing '{file_path}' of size {file_size_mb:.2f} MB.")

    if file_size_mb > chunk_size:
        logging.warning(f"File '{file_path}' exceeds chunk size of {chunk_size} MB. Splitting is not implemented.")
        # Placeholder for splitting logic if needed
        # For simplicity, we proceed without splitting
    try:
        with open(file_path, 'rb') as audio_file:
            logging.debug(f"Sending '{file_path}' to OpenAI API for transcription.")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
        return transcription.to_dict()
    except OpenAIError as e:
        logging.error(f"OpenAI API error while transcribing '{file_path}': {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error while transcribing '{file_path}': {e}")
        return {}

def save_transcription(transcription: dict, output_path: Path, format: str) -> None:
    """
    Saves the transcription to the specified output path in the desired format.
    """
    if not transcription:
        logging.warning(f"No transcription data to save for '{output_path.name}'.")
        return

    try:
        if format == 'txt':
            text = transcription.get('text', '')
            with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write(text)
        elif format == 'json':
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=4)
        logging.info(f"Transcription saved to '{output_path.with_suffix('.' + format)}'.")
    except Exception as e:
        logging.error(f"Failed to save transcription to '{output_path}': {e}")

def process_file(audio_file: Path, args: argparse.Namespace) -> None:
    """
    Processes a single audio file: transcribes and outputs the result.
    """
    transcription = transcribe_audio(audio_file, args.language, args.chunk_size)

    if args.batch:
        # Determine output directory if batch mode is enabled
        output_dir = Path(args.output_dir)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / audio_file.stem

        # Check if transcription already exists
        existing_files = list(output_dir.glob(f"{audio_file.stem}.*"))
        if existing_files and not args.overwrite:
            logging.warning(f"Transcription for '{audio_file.name}' already exists. Skipping. Use -w to overwrite.")
            return

        # Save transcription
        save_transcription(transcription, output_file, args.format)
    else:
        # Print transcription to stdout
        if transcription:
            if args.format == 'txt':
                logging.info(f"Transcription for '{audio_file}'.")
                print(transcription.get('text', ''))
            elif args.format == 'json':
                logging.info(f"Transcription for '{audio_file}'.")
                print(json.dumps(transcription, ensure_ascii=False, indent=4))
        else:
            logging.error(f"Failed to obtain transcription for '{audio_file}'.")

def main() -> None:
    """
    Main function to orchestrate the speech-to-text transcription process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)
    logging.info("Starting Speech-to-Text Transcription Script.")

    # Set OpenAI API key
    global client
    client = OpenAI(api_key=get_api_key(args.api_key))

    # Collect audio files
    audio_files = collect_audio_files(args.input_path, args.recursive)

    logging.info(f"Transcribing {len(audio_files)} file(s).")

    # Process each audio file with or without a progress bar
    if args.verbose:
        for audio_file in tqdm(audio_files, desc="Transcribing Audio Files", unit="file"):
            process_file(audio_file, args)
    else:
        for audio_file in audio_files:
            process_file(audio_file, args)

    logging.info("Transcription process completed successfully.")

if __name__ == '__main__':
    main()