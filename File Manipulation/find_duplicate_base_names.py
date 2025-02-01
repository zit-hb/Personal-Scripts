#!/usr/bin/env python3

# -------------------------------------------------------
# Script: find_duplicate_base_names.py
#
# Description:
# This script scans a specified directory (and optionally subdirectories)
# for files that share the same base name but have different extensions.
#
# Usage:
# ./find_duplicate_base_names.py [directory] [options]
#
# Arguments:
#   - [directory]: The path to the directory to scan for files.
#
# Options:
#   -r, --recursive               Scan directories recursively.
#   -e, --exclude EXTS            File extension to exclude (e.g., .txt). Can be specified multiple times.
#   -i, --ignore-case             Perform case-insensitive comparison of base names.
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#
# Template: ubuntu22.04
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Find files with the same base name but different extensions."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory to scan for files.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan directories recursively.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        help="File extension to exclude (e.g., .txt). Can be specified multiple times.",
    )
    parser.add_argument(
        "-i",
        "--ignore-case",
        action="store_true",
        help="Perform case-insensitive comparison of base names.",
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
    args = parser.parse_args()

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
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_files(
    directory: str,
    recursive: bool,
    exclude_exts: Optional[List[str]],
    ignore_case: bool,
) -> List[Path]:
    """
    Collects all files from the directory, excluding files with certain extensions.
    """
    files: List[Path] = []
    path = Path(directory)

    if not path.is_dir():
        logging.error(f"Directory '{directory}' does not exist or is not a directory.")
        sys.exit(1)

    if exclude_exts:
        # Normalize extensions for comparison
        if ignore_case:
            exclude_exts_set = set(ext.lower() for ext in exclude_exts)
        else:
            exclude_exts_set = set(exclude_exts)
    else:
        exclude_exts_set = set()

    if recursive:
        all_files = path.rglob("*")
    else:
        all_files = path.glob("*")

    for file in all_files:
        if file.is_file():
            # Normalize file extension for comparison
            file_ext = file.suffix
            if ignore_case:
                file_ext = file_ext.lower()
            if file_ext in exclude_exts_set:
                logging.debug(
                    f"Excluding file '{file}' due to extension '{file.suffix}'"
                )
                continue
            files.append(file)

    logging.info(f"Collected {len(files)} file(s) from '{directory}'")
    return files


def find_duplicate_base_names(
    files: List[Path], ignore_case: bool
) -> Dict[str, List[Path]]:
    """
    Finds files with the same base name but different extensions.
    Returns a dictionary mapping from base name to list of files with that base name.
    """
    base_name_to_files: Dict[str, List[Path]] = {}
    for file in files:
        base_name = file.stem  # Filename without extension
        if ignore_case:
            base_name_key = base_name.lower()
        else:
            base_name_key = base_name
        base_name_to_files.setdefault(base_name_key, []).append(file)

    duplicates = {}
    for base_name_key, paths in base_name_to_files.items():
        extensions = {p.suffix.lower() if ignore_case else p.suffix for p in paths}
        if len(extensions) > 1:
            duplicates[base_name_key] = paths

    logging.info(
        f"Found {len(duplicates)} sets of files with duplicate base names and different extensions."
    )
    return duplicates


def main() -> None:
    """
    Main function to orchestrate the process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)
    logging.info("Starting Duplicate Base Name Finder Script.")

    # Collect files
    files = collect_files(
        args.directory, args.recursive, args.exclude, args.ignore_case
    )

    if not files:
        logging.warning("No files found to process.")
        sys.exit(0)

    # Find duplicates
    duplicates = find_duplicate_base_names(files, args.ignore_case)

    if not duplicates:
        logging.info("No files with duplicate base names found.")
        sys.exit(0)

    # Display duplicates
    for base_name_key, files in duplicates.items():
        file_list = ", ".join(str(file) for file in files)
        print(f"Files with base name '{base_name_key}': {file_list}")

    logging.info("Process completed successfully.")


if __name__ == "__main__":
    main()
