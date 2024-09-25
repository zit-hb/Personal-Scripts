#!/usr/bin/env python3

# -------------------------------------------------------
# Script: refactor_code_style.py
#
# Description:
# This script refactors the style of given code files using a language model (LLM).
# It sends the code to the LLM via the `openai` module, transforms the style of the code according to specified coding styles,
# and writes the output to files. If no output directory is specified, the input files are overwritten.
# It supports processing single files or directories, with options for recursive traversal,
# inclusion and exclusion patterns, and automatic language detection.
#
# Usage:
# ./refactor_code_style.py [input_path] [options]
#
# - [input_path]: The path to the input code file or directory.
#
# Options:
# -o, --output-dir OUTPUT_DIR   Directory to save refactored code files. (default: overwrite input files)
# -s, --style STYLE             Coding style to apply. Choices: "Default", "Google", "Airbnb", "PEP8", "Standard". (default: "Default")
# -r, --recursive               Process directories recursively.
# --include PATTERNS            Comma-separated list of glob patterns to include (e.g., "*.py,*.js,*.java"). (default: all files)
# --exclude PATTERNS            Comma-separated list of glob patterns to exclude (e.g., "*.min.js, test_*"). (default: none)
# -m, --model MODEL             OpenAI model to use for code refactoring. (default: "gpt-4o")
# -L, --level LEVEL             Level of changes to apply. Choices: "minimal", "small_fixes", "bug_fixes", "rewrite". (default: "minimal")
# -v, --verbose                 Enable verbose logging (INFO level).
# -vv, --debug                  Enable debug logging (DEBUG level).
# -k, --api-key API_KEY         OpenAI API key. Can also be set via the OPENAI_API_KEY environment variable.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - openai (install via: pip install openai)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import re
from pathlib import Path
import fnmatch
from typing import List, Optional

# Optional: Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

client: Optional[OpenAI] = None  # Global OpenAI client


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Refactor the style of code files using an LLM.'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input code file or directory.'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        help='Directory to save refactored code files. (default: overwrite input files)'
    )
    parser.add_argument(
        '-s',
        '--style',
        type=str,
        default='Default',
        choices=['Default', 'Google', 'Airbnb', 'PEP8', 'Standard'],
        help='Coding style to apply. Choices: "Default", "Google", "Airbnb", "PEP8", "Standard". (default: "Default")'
    )
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Process directories recursively.'
    )
    parser.add_argument(
        '--include',
        type=str,
        help='Comma-separated list of glob patterns to include (e.g., "*.py,*.js,*.java"). (default: all files)'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        help='Comma-separated list of glob patterns to exclude (e.g., "*.min.js, test_*"). (default: none)'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use for code refactoring. (default: "gpt-4o")'
    )
    parser.add_argument(
        '-L',
        '--level',
        type=str,
        default='minimal',
        choices=['minimal', 'small_fixes', 'bug_fixes', 'rewrite'],
        help='Level of changes to apply. Choices: "minimal", "small_fixes", "bug_fixes", "rewrite". (default: "minimal")'
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
        '-k',
        '--api-key',
        type=str,
        help='OpenAI API key. Can also be set via the OPENAI_API_KEY environment variable.'
    )
    args = parser.parse_args()

    # Parse include and exclude patterns
    if args.include:
        args.include = [pattern.strip() for pattern in args.include.split(',')]
    else:
        args.include = []

    if args.exclude:
        args.exclude = [pattern.strip() for pattern in args.exclude.split(',')]
    else:
        args.exclude = []

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


def read_file(file_path: Path) -> Optional[str]:
    """
    Reads the content of the file at file_path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        logging.error(f"Failed to read file '{file_path}': {e}")
        return None


def write_file(file_path: Path, content: str) -> None:
    """
    Writes content to the file at file_path.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        logging.error(f"Failed to write to file '{file_path}': {e}")
        sys.exit(1)


def detect_language(file_path: Path) -> str:
    """
    Detects the programming language of the code file based on its extension.
    """
    extension_to_language = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.php': 'PHP',
        '.swift': 'Swift',
        '.ts': 'TypeScript',
        '.html': 'HTML',
        '.css': 'CSS',
        '.rs': 'Rust',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        # Add more mappings as needed
    }
    ext = file_path.suffix.lower()
    language = extension_to_language.get(ext, 'Plain Text')
    logging.debug(f"Detected language '{language}' for file '{file_path}'.")
    return language


def is_code_file(file_path: Path) -> bool:
    """
    Determines if a file is a code file based on its content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1024)
            if '\0' in content:
                logging.debug(f"File '{file_path}' is binary.")
                return False  # Binary file
            else:
                return True
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        return False


def should_include_file(file_path: Path, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """
    Determines if a file should be included based on include and exclude patterns.
    """
    filename = file_path.name

    if include_patterns:
        if not any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns):
            logging.debug(f"File '{filename}' skipped due to include filter.")
            return False

    if exclude_patterns:
        if any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns):
            logging.debug(f"File '{filename}' excluded by pattern.")
            return False

    return True


def collect_code_files(input_path: str, recursive: bool, include_patterns: List[str], exclude_patterns: List[str]) -> List[Path]:
    """
    Collects all code files from the input path.
    """
    code_files: List[Path] = []

    path = Path(input_path)

    if path.is_file():
        if is_code_file(path):
            if should_include_file(path, include_patterns, exclude_patterns):
                code_files.append(path)
    elif path.is_dir():
        if recursive:
            files = list(path.rglob('*'))
        else:
            files = list(path.glob('*'))
        for file in files:
            if file.is_file() and is_code_file(file):
                if should_include_file(file, include_patterns, exclude_patterns):
                    code_files.append(file)
    else:
        logging.error(f"Input path '{input_path}' is neither a file nor a directory.")
        sys.exit(1)

    if not code_files:
        logging.error(f"No code files found in '{input_path}'.")
        sys.exit(1)

    logging.info(f"Collected {len(code_files)} code file(s) for processing.")
    return code_files


def prepare_prompt(code: str, language: str, style: str, level: str) -> str:
    """
    Prepares the prompt to send to the LLM.
    """
    prompt = f"Refactor the following {language} code according to the specified coding style and requirements.\n"

    if style != 'Default':
        prompt += f"Use the {style} coding style guidelines for {language}.\n"
    else:
        prompt += f"Use the default coding style for {language}.\n"

    if level == 'minimal':
        prompt += "Make minimal changes to adjust the style without altering functionality.\n"
    elif level == 'small_fixes':
        prompt += "Adjust the style and fix any minor issues you find without changing the functionality.\n"
    elif level == 'bug_fixes':
        prompt += "Adjust the style and fix any bugs you find if you are sure they are bugs and how to fix them.\n"
    elif level == 'rewrite':
        prompt += "Feel free to completely rewrite the code to improve it while preserving its functionality.\n"

    prompt += "Only output the refactored code and nothing else.\n\n"
    prompt += "Original Code:\n"
    prompt += "```"
    prompt += code
    prompt += "```"
    return prompt


def call_openai_api(prompt: str, model: str) -> Optional[str]:
    """
    Sends the prompt to the OpenAI API and returns the response.
    """
    global client
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return None


def filter_code_output(response: str, language: str) -> str:
    """
    Filters out any non-code text from the response.
    """
    # Remove any content before and after code blocks
    code_blocks = re.findall(rf'```(?:{language.lower()})?\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return '\n'.join(code_blocks)
    else:
        # If no code blocks, assume the entire response is code
        return response.strip()


def process_file(file_path: Path, args: argparse.Namespace, output_dir: Optional[Path]) -> None:
    """
    Processes a single code file: refactors and outputs the result.
    """
    input_code = read_file(file_path)
    if input_code is None:
        logging.error(f"Skipping file '{file_path}'.")
        return

    language = detect_language(file_path)

    # Prepare prompt
    prompt = prepare_prompt(input_code, language, args.style, args.level)
    logging.debug(f"Prepared prompt for file '{file_path}'.")

    # Call OpenAI API
    response = call_openai_api(prompt, args.model)
    if response is None:
        logging.error(f"Skipping file '{file_path}' due to API error.")
        return

    # Filter code output
    refactored_code = filter_code_output(response, language)
    if not refactored_code:
        logging.error(f"Failed to extract code for file '{file_path}'.")
        return

    # Determine output file path
    if output_dir:
        relative_path = file_path.relative_to(Path(args.input_path))
        output_file_path = output_dir / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file_path = file_path

    # Write refactored code to output file
    write_file(output_file_path, refactored_code)
    logging.info(f"Refactored code written to '{output_file_path}'.")


def main() -> None:
    """
    Main function to orchestrate the code refactoring process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)
    logging.info("Starting Code Style Refactoring Script.")

    # Get OpenAI API key
    api_key = get_api_key(args.api_key)

    # Initialize OpenAI client
    global client
    client = OpenAI(api_key=api_key)

    # Collect code files
    code_files = collect_code_files(
        args.input_path,
        args.recursive,
        args.include,
        args.exclude
    )

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Process each code file
    for file_path in code_files:
        process_file(file_path, args, output_dir)

    logging.info("Code refactoring process completed.")


if __name__ == '__main__':
    main()
