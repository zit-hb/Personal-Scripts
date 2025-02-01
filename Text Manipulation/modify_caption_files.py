#!/usr/bin/env python3

# -------------------------------------------------------
# Script: modify_caption_files.py
#
# Description:
# This script provides multiple functionalities to modify text files.
# It uses sub-commands to organize different tasks such as modifying
# text files corresponding to images, removing tags, ensuring unique tags,
# and adding labels derived from filenames.
#
# Usage:
# ./modify_caption_files.py [command] [options]
#
# Commands:
#   - mod        Modify first lines of text files corresponding to images.
#   - remove     Remove specific tags from text files.
#   - unique     Ensure all tags in text files are unique.
#   - name       Extract name labels from image filenames and add them to text files.
#
# Template: ubuntu22.04
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import re
import sys
from typing import List, Optional, Set


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments using sub-commands.
    """
    parser = argparse.ArgumentParser(
        description="Modify text files using various commands."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Sub-command: mod
    parser_mod = subparsers.add_parser(
        "mod",
        help="Modify first lines of text files corresponding to images.",
    )
    parser_mod.add_argument(
        "images_directory",
        type=str,
        help="The path to the directory containing images.",
    )
    parser_mod.add_argument(
        "texts_directory",
        type=str,
        help="The path to the directory containing text files.",
    )
    group_mod = parser_mod.add_argument_group("Modification options")
    group_mod.add_argument(
        "-p",
        "--prepend",
        type=str,
        help="String to prepend to the first line of the text files.",
    )
    group_mod.add_argument(
        "-a",
        "--append",
        type=str,
        help="String to append to the first line of the text files.",
    )

    # Sub-command: remove
    parser_remove = subparsers.add_parser(
        "remove",
        help="Remove specific tags from text files.",
    )
    parser_remove.add_argument(
        "texts_directory",
        type=str,
        help="The path to the directory containing text files.",
    )
    parser_remove.add_argument(
        "-t",
        "--tag",
        type=str,
        action="append",
        required=True,
        help="Tag to remove from the text files. Can be specified multiple times.",
    )

    # Sub-command: unique
    parser_unique = subparsers.add_parser(
        "unique",
        help="Ensure all tags in text files are unique.",
    )
    parser_unique.add_argument(
        "texts_directory",
        type=str,
        help="The path to the directory containing text files.",
    )

    # Sub-command: name
    parser_name = subparsers.add_parser(
        "name",
        help="Extract name labels from image filenames and add them to text files.",
    )
    parser_name.add_argument(
        "images_directory",
        type=str,
        help="Directory containing images. By default, caption files are assumed to be in the same directory.",
    )
    parser_name.add_argument(
        "-t",
        "--texts_directory",
        type=str,
        default=None,
        help="Directory containing text files if different from images directory.",
    )
    parser_name.add_argument(
        "-c",
        "--create-missing",
        action="store_true",
        help="Create the text file if it does not exist.",
    )
    group_name = parser_name.add_argument_group("Name label options")
    group_name.add_argument(
        "-p",
        "--prepend",
        action="store_true",
        help="Prepend the extracted filename label to the first line of the text file.",
    )
    group_name.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append the extracted filename label to the first line of the text file (default).",
    )
    group_name.add_argument(
        "-r",
        "--regex",
        type=str,
        default=None,
        help="Regex pattern to extract only a specific part of the filename (without extension).",
    )
    group_name.add_argument(
        "-R",
        "--replace",
        type=str,
        default=None,
        help="Regex pattern to replace with spaces in the extracted filename label.",
    )

    args = parser.parse_args()
    return args


def setup_logging() -> None:
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def modify_text_file(
    text_file_path: str,
    prepend_str: Optional[str] = None,
    append_str: Optional[str] = None,
) -> None:
    """
    Modifies the first line of the text file by prepending and/or appending strings.
    """
    try:
        with open(text_file_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip("\n")
                if prepend_str:
                    first_line = prepend_str + first_line
                if append_str:
                    first_line = first_line + append_str
                lines[0] = first_line + "\n"
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
        with open(text_file_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip("\n").strip()
                tags = [tag.strip() for tag in first_line.split(",")]
                original_tags = tags.copy()
                for tag in remove_tags:
                    if tag in tags:
                        tags.remove(tag)
                if len(tags) != len(original_tags):
                    new_first_line = ", ".join(tags)
                    lines[0] = new_first_line + "\n"
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
        with open(text_file_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip("\n").strip()
                tags = [tag.strip() for tag in first_line.split(",")]
                unique_tags: List[str] = []
                seen_tags: Set[str] = set()
                for tag in tags:
                    if tag not in seen_tags:
                        unique_tags.append(tag)
                        seen_tags.add(tag)
                if len(unique_tags) != len(tags):
                    new_first_line = ", ".join(unique_tags)
                    lines[0] = new_first_line + "\n"
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                    logging.info(f"Removed duplicate tags in '{text_file_path}'.")
                else:
                    logging.info(f"No duplicate tags found in '{text_file_path}'.")
            else:
                logging.warning(f"Text file '{text_file_path}' is empty.")
    except Exception as e:
        logging.error(
            f"Failed to ensure unique tags in text file '{text_file_path}': {e}"
        )


def extract_label_from_filename(
    filename: str,
    regex_pattern: Optional[str] = None,
    replace_pattern: Optional[str] = None,
) -> str:
    """
    Extracts a label from the base name of the filename, optionally using a regex to pick a part,
    and optionally replacing characters matched by a regex with spaces.
    """
    base_name, _ = os.path.splitext(filename)
    # If a regex pattern is given, try to match and extract the first group or the entire match.
    if regex_pattern:
        match = re.search(regex_pattern, base_name)
        if match:
            # If the pattern has a capture group, we use that, else we use the whole match.
            if match.groups():
                # use the first group
                extracted = match.group(1)
            else:
                extracted = match.group(0)
        else:
            # If no match is found, fallback to the entire base name
            extracted = base_name
    else:
        extracted = base_name

    # If a replace pattern is given, replace those occurrences with space.
    if replace_pattern:
        extracted = re.sub(replace_pattern, " ", extracted)

    # Strip the extracted label
    return extracted.strip()


def handle_mod_command(args: argparse.Namespace) -> None:
    """
    Handles the 'mod' command.
    """
    images_directory: str = args.images_directory
    texts_directory: str = args.texts_directory
    prepend_str: Optional[str] = args.prepend
    append_str: Optional[str] = args.append

    if not os.path.isdir(images_directory):
        logging.error(
            f"Directory '{images_directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    if not os.path.isdir(texts_directory):
        logging.error(
            f"Directory '{texts_directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    image_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    files_in_images_dir = os.listdir(images_directory)

    for filename in files_in_images_dir:
        if filename.lower().endswith(image_extensions):
            base_name = os.path.splitext(filename)[0]
            text_file_name = base_name + ".txt"
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
        logging.error(
            f"Directory '{texts_directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    text_files = [f for f in os.listdir(texts_directory) if f.lower().endswith(".txt")]

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
        logging.error(
            f"Directory '{texts_directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    text_files = [f for f in os.listdir(texts_directory) if f.lower().endswith(".txt")]

    for text_file in text_files:
        text_file_path = os.path.join(texts_directory, text_file)
        logging.info(f"Ensuring unique tags in '{text_file_path}'.")
        ensure_unique_tags_in_file(text_file_path)


def handle_name_command(args: argparse.Namespace) -> None:
    """
    Handles the 'name' command.
    """
    images_directory: str = args.images_directory
    texts_directory: Optional[str] = args.texts_directory
    create_missing: bool = args.create_missing
    use_prepend: bool = args.prepend
    use_append: bool = args.append
    regex_pattern: Optional[str] = args.regex
    replace_pattern: Optional[str] = args.replace

    if not os.path.isdir(images_directory):
        logging.error(
            f"Directory '{images_directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    # If no texts_directory is specified, we assume it's the same as images_directory
    if not texts_directory:
        texts_directory = images_directory

    if not os.path.isdir(texts_directory):
        logging.error(
            f"Directory '{texts_directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    image_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    files_in_images_dir = os.listdir(images_directory)

    # Default to append if neither prepend nor append is specified
    if not use_prepend and not use_append:
        use_append = True

    for filename in files_in_images_dir:
        if filename.lower().endswith(image_extensions):
            # Extract the label from the filename
            label = extract_label_from_filename(
                filename, regex_pattern, replace_pattern
            )

            # If label is empty, skip
            if not label:
                logging.info(f"Extracted label is empty for '{filename}'. Skipping.")
                continue

            base_name = os.path.splitext(filename)[0]
            text_file_name = base_name + ".txt"
            text_file_path = os.path.join(texts_directory, text_file_name)

            if not os.path.isfile(text_file_path):
                if create_missing:
                    logging.info(f"Creating missing text file '{text_file_path}'.")
                    with open(text_file_path, "w", encoding="utf-8") as f:
                        # Write label if we want to treat it as "append" or "prepend" on empty file
                        # but let's keep it blank for consistency, the loop will handle it.
                        f.write("")
                else:
                    logging.info(
                        f"No caption file for '{filename}' and creation not requested. Skipping."
                    )
                    continue

            # Now read the file, parse first line, add label
            try:
                with open(text_file_path, "r+", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        first_line = lines[0].rstrip("\n").strip()
                        tags = (
                            [t.strip() for t in first_line.split(",")]
                            if first_line
                            else []
                        )
                    else:
                        tags = []

                    # Prepend or append label
                    if use_prepend:
                        tags.insert(0, label)
                    elif use_append:
                        tags.append(label)

                    # Join them back
                    new_first_line = ", ".join([t for t in tags if t])
                    lines[0:1] = [new_first_line + "\n"]  # replace the first line

                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()

                logging.info(f"Added label '{label}' to '{text_file_path}'.")
            except Exception as e:
                logging.error(f"Failed to process text file '{text_file_path}': {e}")
        else:
            logging.debug(f"File '{filename}' is not an image. Skipping.")


def main() -> None:
    """
    Main function that orchestrates the text file modification process.
    """
    args = parse_arguments()
    setup_logging()

    if args.command == "mod":
        handle_mod_command(args)
    elif args.command == "remove":
        handle_remove_command(args)
    elif args.command == "unique":
        handle_unique_command(args)
    elif args.command == "name":
        handle_name_command(args)
    else:
        logging.error("Unknown command.")
        sys.exit(1)


if __name__ == "__main__":
    main()
