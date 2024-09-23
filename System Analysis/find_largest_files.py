#!/usr/bin/env python3

# -------------------------------------------------------
# Script: find_largest_files.py
#
# Description:
# This script searches for the largest files in a specified directory,
# optionally filtering by file types, sizes, or other criteria.
# It recursively scans the directory and outputs the file details
# in a human-readable format or just the file paths.
#
# Usage:
# ./find_largest_files.py [options] [directory]
#
# - [directory]: The directory to search for files. Defaults to the current directory if not provided.
#
# Options:
#   -n NUM_FILES, --num-files NUM_FILES   The number of largest files to display. Defaults to 10.
#   -p, --paths-only                      Output only the file paths, for easy use in other scripts.
#   -t FILE_TYPES, --types FILE_TYPES     Comma-separated list of file extensions to include (e.g., 'txt,pdf,jpg').
#   -e, --exclude EXCLUDE_PATTERNS        Comma-separated list of patterns to exclude (e.g., '*.log,*.tmp').
#   -s SIZE_FILTER, --size SIZE_FILTER    Filter files by size (e.g., '>1M', '<500K', '=2G').
#   -R, --no-recursion                    Do not search directories recursively.
#   -S, --sort-by ATTRIBUTE               Sort files by attribute: size (default), name, mtime, ctime.
#   -r, --reverse                         Reverse the sort order.
#   -v, --verbose                         Enable verbose logging.
#   -h, --help                            Display this help message.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import fnmatch
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import operator


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Search for the largest files in a specified directory, with optional filtering and sorting.'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='The directory to search for files. Defaults to the current directory if not provided.'
    )
    parser.add_argument(
        '-n',
        '--num-files',
        type=int,
        default=10,
        help='The number of largest files to display. Defaults to 10.'
    )
    parser.add_argument(
        '-p',
        '--paths-only',
        action='store_true',
        help='Output only the file paths, for easy use in other scripts.'
    )
    parser.add_argument(
        '-t',
        '--types',
        type=str,
        help='Comma-separated list of file extensions to include (e.g., \'txt,pdf,jpg\').'
    )
    parser.add_argument(
        '-e',
        '--exclude',
        type=str,
        help='Comma-separated list of patterns to exclude (e.g., \'*.log,*.tmp\').'
    )
    parser.add_argument(
        '-s',
        '--size',
        type=str,
        help='Filter files by size (e.g., \'>1M\', \'<500K\', \'=2G\').'
    )
    parser.add_argument(
        '-R',
        '--no-recursion',
        action='store_true',
        help='Do not search directories recursively.'
    )
    parser.add_argument(
        '-S',
        '--sort-by',
        type=str,
        choices=['size', 'name', 'mtime', 'ctime'],
        default='size',
        help='Sort files by attribute: size (default), name, mtime, ctime.'
    )
    parser.add_argument(
        '-r',
        '--reverse',
        action='store_true',
        help='Reverse the sort order.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    args = parser.parse_args()
    return args


def setup_logging(verbose: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def parse_size_filter(size_filter_str: str) -> Optional[Tuple[str, int]]:
    """
    Parses a size filter string and returns a tuple of (operator, size in bytes)
    e.g., '>1M' -> ('>', 1048576)
    """
    pattern = r'([<>]=?|=)\s*([\d\.]+)\s*([KMGTP]?B?)'
    match = re.match(pattern, size_filter_str.strip(), re.IGNORECASE)
    if not match:
        logging.error(f"Invalid size filter format: '{size_filter_str}'")
        return None
    op, num, unit = match.groups()
    num = float(num)
    unit = unit.upper()
    unit_multipliers = {
        '': 1,
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024**2,
        'MB': 1024**2,
        'G': 1024**3,
        'GB': 1024**3,
        'T': 1024**4,
        'TB': 1024**4,
        'P': 1024**5,
        'PB': 1024**5,
    }
    multiplier = unit_multipliers.get(unit, None)
    if multiplier is None:
        logging.error(f"Invalid size unit in size filter: '{unit}'")
        return None
    size_in_bytes = int(num * multiplier)
    return (op, size_in_bytes)


def collect_files(directory: str, recursive: bool, include_extensions: Optional[List[str]], exclude_patterns: Optional[List[str]]) -> List[Path]:
    """
    Collects files from the directory, applying inclusion and exclusion filters.
    """
    path = Path(directory)
    if not path.is_dir():
        logging.error(f"Directory '{directory}' does not exist.")
        sys.exit(1)
    if recursive:
        files = list(path.rglob('*'))
    else:
        files = list(path.glob('*'))
    # Filter files
    files = [f for f in files if f.is_file()]
    if include_extensions:
        files = [f for f in files if f.suffix.lower().lstrip('.') in include_extensions]
    if exclude_patterns:
        for pattern in exclude_patterns:
            files = [f for f in files if not fnmatch.fnmatch(f.name, pattern)]
    return files


def filter_files_by_size(files: List[Path], size_filter: Optional[Tuple[str, int]]) -> List[Path]:
    """
    Filters the files by size based on the size filter.
    """
    if not size_filter:
        return files
    op_str, size_in_bytes = size_filter
    ops = {
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le,
        '=': operator.eq,
    }
    op_func = ops.get(op_str)
    if not op_func:
        logging.error(f"Invalid size filter operator: '{op_str}'")
        return files
    filtered_files = []
    for f in files:
        try:
            file_size = f.stat().st_size
            if op_func(file_size, size_in_bytes):
                filtered_files.append(f)
        except Exception as e:
            logging.warning(f"Could not get size for file '{f}': {e}")
    return filtered_files


def get_file_info(file: Path) -> dict:
    """
    Retrieves file information: size, path, creation date, modification date.
    """
    try:
        stat = file.stat()
        size = stat.st_size
        ctime = datetime.fromtimestamp(stat.st_ctime)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        return {
            'path': file,
            'size': size,
            'ctime': ctime,
            'mtime': mtime,
            'name': file.name,
        }
    except Exception as e:
        logging.warning(f"Could not get info for file '{file}': {e}")
        return {}


def sort_files(files_info: List[dict], sort_by: str, reverse: bool) -> List[dict]:
    """
    Sorts the files based on the specified attribute.
    """
    key_funcs = {
        'size': lambda x: x.get('size', 0),
        'name': lambda x: x.get('name', ''),
        'mtime': lambda x: x.get('mtime', datetime.min),
        'ctime': lambda x: x.get('ctime', datetime.min),
    }
    key_func = key_funcs.get(sort_by)
    if not key_func:
        logging.error(f"Invalid sort attribute: '{sort_by}'")
        return files_info
    sorted_files = sorted(files_info, key=key_func, reverse=not reverse)
    return sorted_files


def humanize_size(size_in_bytes: int) -> str:
    """
    Converts a file size in bytes to a human-readable string.
    """
    original_size = size_in_bytes
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{original_size} B"


def display_files(files_info: List[dict], paths_only: bool, num_files: int) -> None:
    """
    Displays the file information.
    """
    for file_info in files_info[:num_files]:
        if not file_info:
            continue
        path = file_info.get('path')
        if paths_only:
            print(path)
        else:
            size = file_info.get('size', 0)
            ctime = file_info.get('ctime')
            mtime = file_info.get('mtime')
            human_size = humanize_size(size)
            ctime_str = ctime.strftime('%Y-%m-%d %H:%M:%S') if ctime else 'Unavailable'
            mtime_str = mtime.strftime('%Y-%m-%d %H:%M:%S') if mtime else 'Unavailable'
            print(f"Size: {human_size}, File: {path}")
            print(f"Creation Date: {ctime_str}, Modification Date: {mtime_str}")
            print("---------------------------------------")


def main() -> None:
    """
    Main function to orchestrate the file search and display process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    directory = args.directory
    num_files = args.num_files
    paths_only = args.paths_only
    include_extensions = None
    exclude_patterns = None
    size_filter = None
    recursive = not args.no_recursion
    sort_by = args.sort_by
    reverse = args.reverse

    # Process include_extensions
    if args.types:
        include_extensions = [ext.lower().lstrip('.') for ext in args.types.split(',')]

    # Process exclude_patterns
    if args.exclude:
        exclude_patterns = [pattern.strip() for pattern in args.exclude.split(',')]

    # Parse size filter
    if args.size:
        size_filter = parse_size_filter(args.size)
        if not size_filter:
            logging.error("Invalid size filter specified.")
            sys.exit(1)

    # Collect files
    files = collect_files(directory, recursive, include_extensions, exclude_patterns)
    if not files:
        logging.warning("No files found matching the criteria.")
        sys.exit(0)

    # Filter files by size
    files = filter_files_by_size(files, size_filter)
    if not files:
        logging.warning("No files found after applying size filter.")
        sys.exit(0)

    # Get file info
    files_info = [get_file_info(f) for f in files]

    # Sort files
    files_info = sort_files(files_info, sort_by, reverse)

    # Display results
    display_files(files_info, paths_only, num_files)


if __name__ == '__main__':
    main()
