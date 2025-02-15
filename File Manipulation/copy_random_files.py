#!/usr/bin/env python3

# -------------------------------------------------------
# Script: copy_random_files.py
#
# Description:
# This script copies a random selection of files from a source directory
# to a destination directory.
#
# Usage:
# ./copy_random_files.py [source directory] [destination directory] [options]
#
# Arguments:
#   - [source directory]: The path to the directory containing the files to copy.
#   - [destination directory]: The path to the directory where the files will be copied.
#
# Options:
#   -n, --number NUMBER     The number or percentage of files to copy.
#                           Examples:
#                             --number 50   -> copy 50 files
#                             --number 20%  -> copy 20% of the files
#                           Defaults to "20%" if not provided.
#   -r, --recursive         Scan directories recursively.
#   -e, --exclude EXCLUDES  Comma-separated list of directories to exclude from copying.
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#   -d, --dry-run           Perform a dry run without copying any files.
#
# Template: ubuntu24.04
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import random
import shutil


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Copy a random selection of files from the source directory to "
            "the destination directory."
        )
    )
    parser.add_argument(
        "source_directory",
        type=str,
        help="The path to the directory containing the files to copy.",
    )
    parser.add_argument(
        "destination_directory",
        type=str,
        help="The path to the directory where the files will be copied.",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=str,
        default="20%",
        help="The number or percentage of files to copy (default: 20%%).",
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
        type=str,
        help="Comma-separated list of directories to exclude from copying.",
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
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Perform a dry run without copying any files.",
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
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_files(
    directory: str, recursive: bool, exclude_dirs: Optional[str]
) -> List[Path]:
    """
    Collects all files from the directory, excluding specified directories.
    """
    path = Path(directory)
    if not path.is_dir():
        logging.error(
            f"Source directory '{directory}' does not exist or is not a directory."
        )
        sys.exit(1)

    if exclude_dirs:
        exclude_paths = set(
            (path / ex_dir.strip()).resolve() for ex_dir in exclude_dirs.split(",")
        )
    else:
        exclude_paths = set()

    if recursive:
        all_files = path.rglob("*")
    else:
        all_files = path.glob("*")

    collected_files: List[Path] = []
    for file in all_files:
        if file.is_file():
            # Exclude if any of the exclude_paths are in the file's parents
            if any(excluded in file.parents for excluded in exclude_paths):
                logging.debug(f"Excluding file '{file}'")
                continue
            collected_files.append(file)

    logging.info(f"Collected {len(collected_files)} file(s) from '{directory}'")
    return collected_files


def parse_amount_to_copy(amount_str: str, total_files: int) -> int:
    """
    Parses the user-specified amount (raw number or percentage) and returns
    the number of files to copy.
    """
    if total_files <= 0:
        return 0

    if amount_str.endswith("%"):
        try:
            percentage = float(amount_str[:-1])
            files_to_copy = int((percentage / 100) * total_files)
            logging.debug(
                f"Interpreted '{amount_str}' as {percentage}% of {total_files} files: {files_to_copy} files."
            )
            return max(1, files_to_copy)
        except ValueError:
            logging.error(f"Invalid percentage value: '{amount_str}'")
            sys.exit(1)
    else:
        try:
            files_to_copy = int(amount_str)
            logging.debug(
                f"Interpreted '{amount_str}' as a raw number: {files_to_copy} files."
            )
            return max(1, files_to_copy)
        except ValueError:
            logging.error(f"Invalid number value: '{amount_str}'")
            sys.exit(1)


def copy_random_files(
    files: List[Path],
    num_files_to_copy: int,
    destination: str,
    dry_run: bool,
) -> None:
    """
    Copies a specified number of random files from the list to the destination.
    """
    if not files:
        logging.info("No files to copy.")
        return

    destination_path = Path(destination)
    if not destination_path.exists():
        try:
            if not dry_run:
                destination_path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created destination directory: '{destination_path}'")
        except Exception as e:
            logging.error(
                f"Failed to create destination directory '{destination_path}': {e}"
            )
            sys.exit(1)

    num_files_to_copy = min(num_files_to_copy, len(files))
    sys_random = random.SystemRandom()
    selected_files = sys_random.sample(files, num_files_to_copy)

    logging.info(f"Copying {num_files_to_copy} file(s) to '{destination}'")
    for file in selected_files:
        dest_file = destination_path / file.name
        try:
            if not dry_run:
                shutil.copy2(file, dest_file)
            logging.info(f"Copied '{file}' to '{dest_file}'")
        except Exception as e:
            logging.error(f"Failed to copy '{file}' to '{dest_file}': {e}")


def main() -> None:
    """
    Main function to orchestrate the random file copying process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    # Collect files
    files = collect_files(
        directory=args.source_directory,
        recursive=args.recursive,
        exclude_dirs=args.exclude,
    )

    if not files:
        logging.warning("No files found to copy.")
        sys.exit(0)

    # Determine how many files to copy
    num_files_to_copy = parse_amount_to_copy(args.number, len(files))

    # Copy the files
    copy_random_files(
        files=files,
        num_files_to_copy=num_files_to_copy,
        destination=args.destination_directory,
        dry_run=args.dry_run,
    )

    logging.info("Random file copying process completed successfully.")


if __name__ == "__main__":
    main()
