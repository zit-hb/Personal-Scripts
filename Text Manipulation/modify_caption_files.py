#!/usr/bin/env python3

# -------------------------------------------------------
# Script: modify_caption_files.py
#
# Description:
# This script provides multiple functionalities to modify text files.
# It uses sub-commands to organize different tasks such as modifying
# text files corresponding to images, removing tags, and ensuring unique tags.
#
# Usage:
# ./modify_caption_files.py [command] [options]
#
# Commands:
#   - mod      Modify first lines of text files corresponding to images.
#   - remove   Remove specific tags from text files.
#   - unique   Ensure all tags in text files are unique.
#
# Template: ubuntu22.04
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List, Optional, Set


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments using sub-commands.
    """
    parser = argparse.ArgumentParser(
        description='Modify text files using various commands.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Sub-command: mod
    parser_mod = subparsers.add_parser(
        'mod',
        help='Modify first lines of text files corresponding to images.'
    )
    parser_mod.add_argument(
        'images_directory',
        type=str,
        help='The path to the directory containing images.'
    )
    parser_mod.add_argument(
        'texts_directory',
        type=str,
        help='The path to the directory containing text files.'
    )
    group_mod = parser_mod.add_argument_group('Modification options')
    group_mod.add_argument(
        '-p', '--prepend',
        type=str,
        help='String to prepend to the first line of the text files.'
    )
    group_mod.add_argument(
        '-a', '--append',
        type=str,
        help='String to append to the first line of the text files.'
    )

    # Sub-command: remove
    parser_remove = subparsers.add_parser(
        'remove',
        help='Remove specific tags from text files.'
    )
    parser_remove.add_argument(
        'texts_directory',
        type=str,
        help='The path to the directory containing text files.'
    )
    parser_remove.add_argument(
        '-t', '--tag',
        type=str,
        action='append',
        required=True,
        help='Tag to remove from the text files. Can be specified multiple times.'
    )

    # Sub-command: unique
    parser_unique = subparsers.add_parser(
        'unique',
        help='Ensure all tags in text files are unique.'
    )
    parser_unique.add_argument(
        'texts_directory',
        type=str,
        help='The path to the directory containing text files.'
    )

    args = parser.parse_args()

    return args


def setup_logging() -> None:
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def modify_text_file(text_file_path: str, prepend_str: Optional[str] = None, append_str: Optional[str] = None) -> None:
    """
    Modifies the first line of the text file by prepending and/or appending strings.
    """
    try:
        with open(text_file_path, 'r+') as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip('\n')
                if prepend_str:
                    first_line = prepend_str + first_line
                if append_str:
                    first_line = first_line + append_str
                lines[0] = first_line + '\n'
                f.seek(0)
                f.writelines(lines)
                f.truncate()
            else:
                logging.warning(f"Text file '{text_file_path}' is empty.")
    except Exception as e:
        logging.error(f"Failed to modify text file '{text_file_path}': {e}")


def remove_tags_from_file(text_file_path: str, remove_tags: List[str]) -> None:
    """
    Removes specific tags from the first line of the text file.
    """
    try:
        with open(text_file_path, 'r+') as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip('\n')
                tags = [tag.strip() for tag in first_line.split(',')]
                original_tags = tags.copy()
                for tag in remove_tags:
                    if tag in tags:
                        tags.remove(tag)
                if len(tags) != len(original_tags):
                    first_line = ', '.join(tags)
                    lines[0] = first_line + '\n'
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                    logging.info(f"Removed tags {remove_tags} from '{text_file_path}'.")
                else:
                    logging.info(f"No specified tags found in '{text_file_path}'.")
            else:
                logging.warning(f"Text file '{text_file_path}' is empty.")
    except Exception as e:
        logging.error(f"Failed to remove tags from text file '{text_file_path}': {e}")


def ensure_unique_tags_in_file(text_file_path: str) -> None:
    """
    Ensures that all tags in the first line of the text file are unique.
    """
    try:
        with open(text_file_path, 'r+') as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip('\n')
                tags = [tag.strip() for tag in first_line.split(',')]
                unique_tags: List[str] = []
                seen_tags: Set[str] = set()
                for tag in tags:
                    if tag not in seen_tags:
                        unique_tags.append(tag)
                        seen_tags.add(tag)
                if len(unique_tags) != len(tags):
                    first_line = ', '.join(unique_tags)
                    lines[0] = first_line + '\n'
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                    logging.info(f"Removed duplicate tags in '{text_file_path}'.")
                else:
                    logging.info(f"No duplicate tags found in '{text_file_path}'.")
            else:
                logging.warning(f"Text file '{text_file_path}' is empty.")
    except Exception as e:
        logging.error(f"Failed to ensure unique tags in text file '{text_file_path}': {e}")


def handle_mod_command(args: argparse.Namespace) -> None:
    """
    Handles the 'mod' command.
    """
    images_directory: str = args.images_directory
    texts_directory: str = args.texts_directory
    prepend_str: Optional[str] = args.prepend
    append_str: Optional[str] = args.append

    if not os.path.isdir(images_directory):
        logging.error(f"Directory '{images_directory}' does not exist or is not a directory.")
        sys.exit(1)

    if not os.path.isdir(texts_directory):
        logging.error(f"Directory '{texts_directory}' does not exist or is not a directory.")
        sys.exit(1)

    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    files_in_images_dir = os.listdir(images_directory)

    for filename in files_in_images_dir:
        if filename.lower().endswith(image_extensions):
            base_name = os.path.splitext(filename)[0]
            text_file_name = base_name + '.txt'
            text_file_path = os.path.join(texts_directory, text_file_name)
            if os.path.isfile(text_file_path):
                logging.info(f"Modifying text file '{text_file_path}'.")
                modify_text_file(text_file_path, prepend_str, append_str)
            else:
                logging.info(f"No corresponding text file for image '{filename}'.")
        else:
            logging.debug(f"File '{filename}' is not an image. Skipping.")


def handle_remove_command(args: argparse.Namespace) -> None:
    """
    Handles the 'remove' command.
    """
    texts_directory: str = args.texts_directory
    remove_tags: List[str] = args.tag  # This is a list of tags

    if not os.path.isdir(texts_directory):
        logging.error(f"Directory '{texts_directory}' does not exist or is not a directory.")
        sys.exit(1)

    text_files = [f for f in os.listdir(texts_directory) if f.lower().endswith('.txt')]

    for text_file in text_files:
        text_file_path = os.path.join(texts_directory, text_file)
        logging.info(f"Processing text file '{text_file_path}'.")
        remove_tags_from_file(text_file_path, remove_tags)


def handle_unique_command(args: argparse.Namespace) -> None:
    """
    Handles the 'unique' command.
    """
    texts_directory: str = args.texts_directory

    if not os.path.isdir(texts_directory):
        logging.error(f"Directory '{texts_directory}' does not exist or is not a directory.")
        sys.exit(1)

    text_files = [f for f in os.listdir(texts_directory) if f.lower().endswith('.txt')]

    for text_file in text_files:
        text_file_path = os.path.join(texts_directory, text_file)
        logging.info(f"Ensuring unique tags in '{text_file_path}'.")
        ensure_unique_tags_in_file(text_file_path)


def main() -> None:
    """
    Main function that orchestrates the text file modification process.
    """
    args = parse_arguments()
    setup_logging()

    if args.command == 'mod':
        handle_mod_command(args)
    elif args.command == 'remove':
        handle_remove_command(args)
    elif args.command == 'unique':
        handle_unique_command(args)
    else:
        logging.error("Unknown command.")
        sys.exit(1)


if __name__ == '__main__':
    main()
