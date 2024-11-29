#!/usr/bin/env python3

# -------------------------------------------------------
# Script: remove_duplicate_files.py
#
# Description:
# This script scans a specified directory (and optionally subdirectories)
# for duplicate files and removes them based on the specified criteria.
# It uses file hashing to detect duplicates and allows the user to specify
# the hashing algorithm. Users can also choose which duplicate file to keep
# (e.g., oldest, newest, biggest, smallest).
#
# Usage:
# ./remove_duplicate_files.py [directory] [options]
#
# - [directory]: The path to the directory to scan for duplicate files.
#
# Options:
# -a, --algorithm ALGORITHM     Hashing algorithm to use. Choices: "md5", "sha1", "sha256", etc. (default: "md5")
# -r, --recursive               Scan directories recursively.
# -k, --keep CRITERIA           Criteria for keeping a file. Choices: "oldest", "newest", "biggest", "smallest". (default: "oldest")
# -e, --exclude EXCLUDE_DIRS    Comma-separated list of directories to exclude from scanning.
# -v, --verbose                 Enable verbose logging (INFO level).
# -vv, --debug                  Enable debug logging (DEBUG level).
# -n, --dry-run                 Perform a dry run without deleting any files.
#
# Template: ubuntu22.04
#
# Requirements:
# - tqdm (install via: pip install tqdm)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import hashlib

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Find and remove duplicate files in a directory.'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='The path to the directory to scan for duplicate files.'
    )
    parser.add_argument(
        '-a',
        '--algorithm',
        type=str,
        default='md5',
        help='Hashing algorithm to use. Choices: "md5", "sha1", "sha256", etc. (default: "md5")'
    )
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Scan directories recursively.'
    )
    parser.add_argument(
        '-k',
        '--keep',
        type=str,
        default='oldest',
        choices=['oldest', 'newest', 'biggest', 'smallest'],
        help='Criteria for keeping a file. Choices: "oldest", "newest", "biggest", "smallest". (default: "oldest")'
    )
    parser.add_argument(
        '-e',
        '--exclude',
        type=str,
        help='Comma-separated list of directories to exclude from scanning.'
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
        '-n',
        '--dry-run',
        action='store_true',
        help='Perform a dry run without deleting any files.'
    )
    args = parser.parse_args()

    # Validate the hashing algorithm
    try:
        hashlib.new(args.algorithm)
    except ValueError:
        parser.error(f"Hashing algorithm '{args.algorithm}' is not supported.")

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

    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def collect_files(directory: str, recursive: bool, exclude_dirs: Optional[str]) -> List[Path]:
    """
    Collects all files from the directory, excluding specified directories.
    """
    files: List[Path] = []
    path = Path(directory)

    if not path.is_dir():
        logging.error(f"Directory '{directory}' does not exist or is not a directory.")
        sys.exit(1)

    if exclude_dirs:
        exclude_paths = set((path / ex_dir.strip()).resolve() for ex_dir in exclude_dirs.split(','))
    else:
        exclude_paths = set()

    if recursive:
        all_files = path.rglob('*')
    else:
        all_files = path.glob('*')

    for file in all_files:
        if file.is_file():
            if any(excluded in file.parents for excluded in exclude_paths):
                logging.debug(f"Excluding file '{file}'")
                continue
            files.append(file)

    logging.info(f"Collected {len(files)} file(s) from '{directory}'")
    return files


def compute_hash(file_path: Path, algorithm: str) -> str:
    """
    Computes the hash of a file using the specified algorithm.
    """
    hash_func = hashlib.new(algorithm)
    try:
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        file_hash = hash_func.hexdigest()
        logging.debug(f"Computed {algorithm} hash for '{file_path}': {file_hash}")
        return file_hash
    except Exception as e:
        logging.error(f"Failed to compute hash for '{file_path}': {e}")
        return ''


def find_duplicates(files: List[Path], algorithm: str) -> Dict[str, List[Path]]:
    """
    Finds duplicate files based on the hash of their content.
    Returns a dictionary mapping from hash to list of files with that hash.
    """
    hashes: Dict[str, List[Path]] = {}
    for file in tqdm(files, desc="Computing file hashes", unit="file"):
        file_hash = compute_hash(file, algorithm)
        if file_hash:
            hashes.setdefault(file_hash, []).append(file)
    duplicates = {hash: paths for hash, paths in hashes.items() if len(paths) > 1}
    logging.info(f"Found {len(duplicates)} sets of duplicates.")
    return duplicates


def select_file_to_keep(files: List[Path], criteria: str) -> Path:
    """
    Selects one file to keep from a list of files based on the specified criteria.
    """
    if criteria == 'oldest':
        file_to_keep = min(files, key=lambda f: f.stat().st_mtime)
    elif criteria == 'newest':
        file_to_keep = max(files, key=lambda f: f.stat().st_mtime)
    elif criteria == 'biggest':
        file_to_keep = max(files, key=lambda f: f.stat().st_size)
    elif criteria == 'smallest':
        file_to_keep = min(files, key=lambda f: f.stat().st_size)
    else:
        logging.error(f"Invalid criteria '{criteria}'. Defaulting to keep the first file.")
        file_to_keep = files[0]

    logging.debug(f"Selected file to keep based on '{criteria}': '{file_to_keep}'")
    return file_to_keep


def delete_duplicates(duplicates: Dict[str, List[Path]], criteria: str, dry_run: bool) -> None:
    """
    Deletes duplicate files based on the specified criteria.
    """
    total_deleted = 0
    for file_hash, files in duplicates.items():
        file_to_keep = select_file_to_keep(files, criteria)
        files_to_delete = [f for f in files if f != file_to_keep]
        for file in files_to_delete:
            try:
                if not dry_run:
                    file.unlink()
                logging.info(f"Deleted duplicate file '{file}'")
                total_deleted += 1
            except Exception as e:
                logging.error(f"Failed to delete file '{file}': {e}")
    logging.info(f"Deleted {total_deleted} duplicate file(s).")


def main() -> None:
    """
    Main function to orchestrate the duplicate file removal process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)
    logging.info("Starting Duplicate File Removal Script.")

    # Collect files
    files = collect_files(args.directory, args.recursive, args.exclude)

    if not files:
        logging.warning("No files found to process.")
        sys.exit(0)

    # Find duplicates
    duplicates = find_duplicates(files, args.algorithm)

    if not duplicates:
        logging.info("No duplicate files found.")
        sys.exit(0)

    # Delete duplicates
    delete_duplicates(duplicates, args.keep, args.dry_run)

    logging.info("Duplicate file removal process completed successfully.")


if __name__ == '__main__':
    main()
