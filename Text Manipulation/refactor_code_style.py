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
# Additionally, it can include the last refactored code in the chat history to maintain consistency across multiple files.
#
# Usage:
# ./refactor_code_style.py [input_path] [options]
#
# Arguments:
#   - [input_path]: The path to the input code file or directory.
#
# Options:
#   -o, --output-dir DIR          Directory to save refactored code files. (default: overwrite input files)
#   -s, --style STYLE             Coding style to apply. Choices: "Default", "Google", "Airbnb", "PEP8", "Standard". (default: "Default")
#   -r, --recursive               Process directories recursively.
#   -i, --include PATTERNS        Comma-separated list of glob patterns to include (e.g., "*.py,*.js,*.java"). (default: known code extensions)
#   -e, --exclude PATTERNS        Comma-separated list of glob patterns to exclude (e.g., "*.min.js, test_*"). (default: none)
#   -m, --model MODEL             OpenAI model to use for code refactoring. (default: "gpt-4o")
#   -L, --level LEVEL             Level of changes to apply. Choices: "minimal", "small_fixes", "bug_fixes", "rewrite". (default: "minimal")
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#   -k, --api-key API_KEY         OpenAI API key. Can also be set via the OPENAI_API_KEY environment variable.
#   -l, --include-last-cleaned    Include the last refactored code in the chat history for consistency across multiple files.
#
# Template: ubuntu22.04
#
# Requirements:
#   - openai (install via: pip install openai==1.55.3)
#   - python-dotenv (install via: pip install python-dotenv==1.0.1)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from pathlib import Path
import fnmatch
from typing import List, Optional, Dict
import json

# Optional: Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

client: Optional[OpenAI] = None  # Global OpenAI client

# Mapping of file extensions to programming languages
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
        '-i',
        '--include',
        type=str,
        help='Comma-separated list of glob patterns to include (e.g., "*.py,*.js,*.java"). (default: known code extensions)'
    )
    parser.add_argument(
        '-e',
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
    parser.add_argument(
        '-l',
        '--include-last-cleaned',
        action='store_true',
        help='Include the last refactored code in the chat history for consistency across multiple files.'
    )
    args = parser.parse_args()

    # Parse include and exclude patterns
    if args.include:
        args.include = [pattern.strip() for pattern in args.include.split(',')]
    else:
        # Default to known code file extensions
        args.include = ['*' + ext for ext in extension_to_language.keys()]

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


def prepare_messages(
    code: str,
    language: str,
    style: str,
    level: str,
    include_additional_info: bool,
    example_refactored_code: Optional[str] = None
) -> List[Dict]:
    """
    Prepares the messages to send to the LLM.
    Structures the conversation with distinct messages for task instructions,
    example refactored code (if available), and the actual code to refactor.
    """
    messages = []

    # System Message: General Instructions
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that refactors code according to specified coding styles and levels."
            " Use the provided function to return the refactored code."
            + (" Also provide additional information if requested." if include_additional_info else "")
        )
    }
    messages.append(system_message)

    # User Message: Task Description
    task_message_content = (
        f"Please refactor the following {language} code according to the {style} coding style."
    )

    # Add level-specific instructions
    if level == 'minimal':
        task_message_content += " Make minimal changes to adjust the style without altering functionality."
    elif level == 'small_fixes':
        task_message_content += " Adjust the style and fix any minor issues you find without changing the functionality."
    elif level == 'bug_fixes':
        task_message_content += " Adjust the style and fix any bugs you find if you are sure they are bugs and know how to fix them."
    elif level == 'rewrite':
        task_message_content += " Feel free to completely rewrite the code to improve it while preserving its functionality."

    task_message_content += " Please acknowledge these instructions."

    task_message = {
        "role": "user",
        "content": task_message_content
    }
    messages.append(task_message)

    # Assistant Message: Acknowledgment of Task
    assistant_ack_task = {
        "role": "assistant",
        "content": "Understood. I'm ready to receive the code you want to refactor."
    }
    messages.append(assistant_ack_task)

    # (Optional) User Message: Example Refactored Code
    if example_refactored_code:
        example_message_content = (
            "Here is an example of previously refactored code to help maintain consistency:\n"
            "```\n"
            f"{example_refactored_code}\n"
            "```"
        )
        example_message = {
            "role": "user",
            "content": example_message_content
        }
        messages.append(example_message)

        # Assistant Message: Acknowledgment of Example
        assistant_ack_example = {
            "role": "assistant",
            "content": "Got it. I'll use this example to maintain consistency in refactoring."
        }
        messages.append(assistant_ack_example)

    # User Message: Actual Code to Refactor
    code_message_content = f"Here is the code that needs to be refactored:\n```\n{code}\n```"
    code_message = {
        "role": "user",
        "content": code_message_content
    }
    messages.append(code_message)

    return messages


def call_openai_api(messages: List[Dict], model: str, functions: Optional[List[Dict]] = None) -> Optional[Dict]:
    """
    Sends the messages to the OpenAI API and returns the response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions
        )
        return response
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return None


def get_refactored_code(messages: List[Dict], model: str, functions: Optional[List[Dict]]) -> Optional[Dict]:
    """
    Calls the OpenAI API to get the refactored code, with retry logic.
    Returns the function arguments as a dict if successful, else None.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        # Call OpenAI API
        response = call_openai_api(messages, model, functions)
        if response is None:
            logging.error(f"OpenAI API call failed on attempt {attempt + 1}.")
            if attempt < max_attempts - 1:
                logging.info(f"Retrying... ({attempt + 1}/{max_attempts})")
                continue
            else:
                return None

        assistant_message = response.choices[0].message

        if hasattr(assistant_message, 'function_call') and assistant_message.function_call is not None:
            function_call = assistant_message.function_call
            arguments = function_call.arguments

            # Parse arguments (which is a JSON string)
            try:
                function_args = json.loads(arguments)
                return function_args
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse function arguments on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    logging.info(f"Retrying... ({attempt + 1}/{max_attempts})")
                    continue
                else:
                    return None
        else:
            # If no function call, try to get the content directly
            if assistant_message.content:
                refactored_code = assistant_message.content.strip()
                if refactored_code:
                    function_args = {'refactored_code': refactored_code}
                    return function_args
                else:
                    logging.error("No content in assistant's response.")
                    if attempt < max_attempts - 1:
                        logging.info(f"Retrying... ({attempt + 1}/{max_attempts})")
                        continue
                    else:
                        return None
            else:
                logging.error("No function call or content in assistant's response.")
                if attempt < max_attempts - 1:
                    logging.info(f"Retrying... ({attempt + 1}/{max_attempts})")
                    continue
                else:
                    return None
    return None


def determine_output_file_path(file_path: Path, output_dir: Optional[Path], input_path: str) -> Path:
    """
    Determines the output file path, creating necessary directories.
    """
    if output_dir:
        try:
            relative_path = file_path.relative_to(Path(input_path))
        except ValueError:
            # If file_path is not relative to input_path
            relative_path = file_path.name
        output_file_path = output_dir / relative_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file_path = file_path
    return output_file_path


def process_file(
    file_path: Path,
    args: argparse.Namespace,
    output_dir: Optional[Path],
    example_refactored_code: Optional[str] = None
) -> Optional[str]:
    """
    Processes a single code file: refactors and outputs the result.
    Returns the refactored code if successful, else None.
    """
    input_code = read_file(file_path)
    if input_code is None:
        logging.error(f"Skipping file '{file_path}'.")
        return None

    language = detect_language(file_path)

    # Determine if additional info is needed based on logging level
    include_additional_info = logging.getLogger().isEnabledFor(logging.INFO)

    # Prepare messages, including example refactored code if provided
    messages = prepare_messages(
        code=input_code,
        language=language,
        style=args.style,
        level=args.level,
        include_additional_info=include_additional_info,
        example_refactored_code=example_refactored_code
    )

    # Define functions
    function_properties = {
        "refactored_code": {"type": "string", "description": "The refactored code."}
    }

    if include_additional_info:
        function_properties.update({
            "description": {"type": "string", "description": "A very short description of the code."},
            "quality": {"type": "string", "description": "A very short assessment of code quality."},
            "style": {"type": "string", "description": "The coding style used."},
            "language": {"type": "string", "description": "Programming language of the code."},
            "libraries": {"type": "array", "items": {"type": "string"}, "description": "A short list of libraries used."},
            "frameworks": {"type": "array", "items": {"type": "string"}, "description": "A short list of frameworks used."}
        })

    functions = [
        {
            "name": "refactor_code",
            "description": "Refactors the code according to the specified coding style and level.",
            "parameters": {
                "type": "object",
                "properties": function_properties,
                "required": ["refactored_code"],
                "additionalProperties": False
            }
        }
    ]

    # Get refactored code and additional info
    function_args = get_refactored_code(messages, args.model, functions)
    if function_args is None:
        logging.error(f"Skipping file '{file_path}' due to errors.")
        return None

    refactored_code = function_args.get('refactored_code')
    if not refactored_code:
        logging.error("Refactored code not found in function arguments.")
        return None

    # Log additional information if available
    if include_additional_info:
        description = function_args.get('description', '')
        quality = function_args.get('quality', '')
        used_style = function_args.get('style', '')
        used_language = function_args.get('language', '')
        libraries = function_args.get('libraries', [])
        frameworks = function_args.get('frameworks', [])

        logging.info(f"File: {file_path}")
        if description:
            logging.info(f"Description: {description}")
        if quality:
            logging.info(f"Quality: {quality}")
        if used_style:
            logging.info(f"Style: {used_style}")
        if used_language:
            logging.info(f"Language: {used_language}")
        if libraries:
            logging.info(f"Libraries: {', '.join(libraries)}")
        if frameworks:
            logging.info(f"Frameworks: {', '.join(frameworks)}")

    # Determine output file path
    output_file_path = determine_output_file_path(file_path, output_dir, args.input_path)

    # Write refactored code to output file
    write_file(output_file_path, refactored_code)
    logging.info(f"Refactored code written to '{output_file_path}'.")

    return refactored_code  # Return the refactored code to update example_refactored_code


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

    # Initialize variable to store the example refactored code
    example_refactored_code: Optional[str] = None

    # Process each code file
    for file_path in code_files:
        if args.include_last_cleaned:
            current_refactored_code = process_file(
                file_path,
                args,
                output_dir,
                example_refactored_code
            )
        else:
            current_refactored_code = process_file(
                file_path,
                args,
                output_dir
            )

        # Update the example_refactored_code if processing was successful
        if args.include_last_cleaned and current_refactored_code:
            example_refactored_code = current_refactored_code

    logging.info("Code refactoring process completed.")


if __name__ == '__main__':
    main()
