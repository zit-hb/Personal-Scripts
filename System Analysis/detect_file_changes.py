#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_file_changes.py
#
# Description:
# This script scans one or more specified directories (optionally recursively)
# and collects metadata for each file. It can read old metadata from a JSON file
# ("--input") and compare it against new metadata to detect changes. It can also
# write new metadata to a JSON file ("--output").
#
# Usage:
#   ./detect_file_changes.py [directories...] [options]
#
# Arguments:
#   [directories...]: Path(s) to the directory(ies) to scan (default: current directory).
#
# Options:
#   -i, --input INPUT_JSON        Path to the JSON file containing old metadata for comparison.
#   -o, --output OUTPUT_JSON      Path to the JSON file where new metadata should be written.
#   -a, --algorithm ALGORITHM     Hashing algorithm(s) to use. Can be specified multiple times (default: sha512).
#   -I, --include PATTERN         Regex pattern(s) to include. Can be specified multiple times.
#   -E, --exclude PATTERN         Regex pattern(s) to exclude. Can be specified multiple times.
#   -r, --recursive               Scan directories recursively.
#   -s, --check-size              Only check changes in file size.
#   -t, --check-time              Only check changes in file creation and modification times.
#   -p, --check-permissions       Only check changes in file permissions.
#   -l, --check-capabilities      Only check changes in file capabilities.
#   -m, --check-hash              Only check changes in file content hash(es).
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#
# Template: ubuntu22.04
#
# Requirements:
#   - prettytable (install via: pip install prettytable==3.12.0)
#   - getcap (install via: apt-get install -y libcap2-bin)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import re
import hashlib
import stat
import subprocess
from dataclasses import dataclass, asdict

# Try to import pretty table or fallback to plain formatting
try:
    from prettytable import PrettyTable

    USE_PRETTYTABLE = True
except ImportError:
    USE_PRETTYTABLE = False


@dataclass
class FileMetadata:
    """
    Dataclass to store file metadata.
    """

    path: str
    size: int
    ctime: float
    mtime: float
    permissions: str
    capabilities: Optional[str]
    hashes: Dict[str, str]


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and validates them.
    Returns an argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check for file changes by comparing metadata."
    )

    parser.add_argument(
        "directories",
        type=str,
        nargs="*",
        help="Path(s) to the directory(ies) to scan. Defaults to current directory.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the JSON file containing old metadata for comparison.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the JSON file where new metadata should be written.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        action="append",
        default=[],
        help="Hashing algorithm(s) to use.",
    )
    parser.add_argument(
        "-I",
        "--include",
        type=str,
        action="append",
        default=[],
        help="Regex pattern(s) to include.",
    )
    parser.add_argument(
        "-E",
        "--exclude",
        type=str,
        action="append",
        default=[],
        help="Regex pattern(s) to exclude.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan directories recursively.",
    )
    parser.add_argument(
        "-s",
        "--check-size",
        action="store_true",
        help="Only check changes in file size.",
    )
    parser.add_argument(
        "-t",
        "--check-time",
        action="store_true",
        help="Only check changes in file creation and modification times.",
    )
    parser.add_argument(
        "-p",
        "--check-permissions",
        action="store_true",
        help="Only check changes in file permissions.",
    )
    parser.add_argument(
        "-l",
        "--check-capabilities",
        action="store_true",
        help="Only check changes in file capabilities.",
    )
    parser.add_argument(
        "-m",
        "--check-hash",
        action="store_true",
        help="Only check changes in file content hash(es).",
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

    if not args.input and not args.output:
        parser.error("You must provide at least --input or --output (or both).")

    # If no hashing algorithms are specified, default to sha512
    if not args.algorithm:
        args.algorithm = ["sha512"]

    # Validate hashing algorithms
    for algo in args.algorithm:
        try:
            hashlib.new(algo)
        except ValueError:
            parser.error(f"Hashing algorithm '{algo}' is not supported.")

    return args


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration based on verbose/debug flags.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def collect_files(
    directories: List[str],
    recursive: bool,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> List[Path]:
    """
    Collects all files from the given directories (one or more),
    applying include/exclude regex filters, and returns a list of Path objects.
    """
    all_files: List[Path] = []

    include_regexes = (
        [re.compile(pattern) for pattern in include_patterns]
        if include_patterns
        else []
    )
    exclude_regexes = (
        [re.compile(pattern) for pattern in exclude_patterns]
        if exclude_patterns
        else []
    )

    # If no directories are specified, default to current directory
    if not directories:
        directories = ["."]

    def matches_any_regex(text: str, regex_list: List[re.Pattern]) -> bool:
        """
        Returns True if 'text' matches at least one of the regexes in 'regex_list'.
        """
        return any(regex.search(text) for regex in regex_list)

    for dir_str in directories:
        path = Path(dir_str)

        if not path.is_dir():
            logging.error(f"'{dir_str}' does not exist or is not a directory.")
            continue

        files_iter = path.rglob("*") if recursive else path.glob("*")

        for f in files_iter:
            if f.is_file():
                full_path_str = str(f.resolve())

                # If includes are specified, at least one must match
                if include_regexes and not matches_any_regex(
                    full_path_str, include_regexes
                ):
                    logging.debug(
                        f"Excluding '{full_path_str}' (no match for --include)."
                    )
                    continue

                # If excludes are specified, none may match
                if exclude_regexes and matches_any_regex(
                    full_path_str, exclude_regexes
                ):
                    logging.debug(f"Excluding '{full_path_str}' (matched --exclude).")
                    continue

                all_files.append(f)

    logging.info(f"Collected {len(all_files)} file(s) from directories {directories}.")
    return all_files


def get_file_permissions(file_path: Path) -> str:
    """
    Returns the file permissions of 'file_path' as a human-readable string (e.g., '-rwxr-xr--').
    """
    st = file_path.stat()
    return stat.filemode(st.st_mode)


def get_file_capabilities(file_path: Path) -> Optional[str]:
    """
    Returns the Linux file capabilities for 'file_path', if supported, otherwise None.
    Uses the 'getcap' command on systems where it is available.
    """
    try:
        result = subprocess.run(
            ["getcap", str(file_path)], capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            # Example output: "/path/to/file = cap_net_bind_service+ep"
            return result.stdout.strip().split("=", 1)[-1].strip()
    except FileNotFoundError:
        # 'getcap' not installed
        pass
    return None


def compute_hashes(file_path: Path, algorithms: List[str]) -> Dict[str, str]:
    """
    Computes and returns the specified hashes of the file located at 'file_path'.
    Returns a dictionary {algorithm: hex_digest}.
    """
    hashes_result: Dict[str, str] = {}
    try:
        data = file_path.read_bytes()
    except OSError as e:
        logging.error(f"Failed to read file '{file_path}': {e}")
        return hashes_result

    for algo in algorithms:
        try:
            h = hashlib.new(algo)
            h.update(data)
            hashes_result[algo] = h.hexdigest()
        except ValueError:
            logging.warning(f"Skipping unsupported algorithm '{algo}'.")

    return hashes_result


def gather_metadata(
    files: List[Path], algorithms: List[str]
) -> Dict[str, FileMetadata]:
    """
    Gathers and returns metadata (size, times, permissions, capabilities, hashes) for a list of files.
    The return value is a dict keyed by absolute file path (string) -> FileMetadata object.
    """
    metadata: Dict[str, FileMetadata] = {}
    for f in files:
        try:
            st = f.stat()
            md = FileMetadata(
                path=str(f.resolve()),
                size=st.st_size,
                ctime=st.st_ctime,
                mtime=st.st_mtime,
                permissions=get_file_permissions(f),
                capabilities=get_file_capabilities(f),
                hashes=compute_hashes(f, algorithms),
            )
            metadata[md.path] = md
        except OSError as e:
            logging.error(f"Failed to get metadata for '{f}': {e}")
    return metadata


def load_metadata_from_json(json_path: Path) -> Dict[str, FileMetadata]:
    """
    Loads file metadata from a JSON file at 'json_path'.
    Returns a dict of path string -> FileMetadata.
    If loading fails, returns an empty dict.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        loaded_md: Dict[str, FileMetadata] = {}
        for path_str, md_dict in raw_data.get("metadata", {}).items():
            loaded_md[path_str] = FileMetadata(
                path=md_dict["path"],
                size=md_dict["size"],
                ctime=md_dict["ctime"],
                mtime=md_dict["mtime"],
                permissions=md_dict["permissions"],
                capabilities=md_dict["capabilities"],
                hashes=md_dict["hashes"],
            )
        return loaded_md
    except (OSError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load metadata from '{json_path}': {e}")
        return {}


def detect_changes(
    old: Dict[str, FileMetadata],
    new: Dict[str, FileMetadata],
    check_size: bool,
    check_time: bool,
    check_perms: bool,
    check_caps: bool,
    check_hash: bool,
) -> List[Dict[str, Any]]:
    """
    Detects added, removed, and changed files by comparing 'old' metadata to 'new' metadata.
    Returns a list of change objects in the form:
        [
          {
            'path': <file_path>,
            'type': 'added' | 'removed' | 'changed',
            'details': { ... }
          },
          ...
        ]
    """
    changes: List[Dict[str, Any]] = []

    old_paths = set(old.keys())
    new_paths = set(new.keys())

    # Files that exist only in new data => 'added'
    added = new_paths - old_paths
    # Files that exist only in old data => 'removed'
    removed = old_paths - new_paths

    for path_str in added:
        changes.append({"path": path_str, "type": "added", "details": {}})

    for path_str in removed:
        changes.append({"path": path_str, "type": "removed", "details": {}})

    # Files present in both => potentially 'changed'
    intersected = old_paths & new_paths
    for path_str in intersected:
        o = old[path_str]
        n = new[path_str]

        detail_changes = compare_file_metadata(
            o,
            n,
            check_size=check_size,
            check_time=check_time,
            check_perms=check_perms,
            check_caps=check_caps,
            check_hash=check_hash,
        )

        if detail_changes:
            changes.append(
                {"path": path_str, "type": "changed", "details": detail_changes}
            )

    return changes


def compare_file_metadata(
    old_md: FileMetadata,
    new_md: FileMetadata,
    check_size: bool,
    check_time: bool,
    check_perms: bool,
    check_caps: bool,
    check_hash: bool,
) -> Dict[str, Any]:
    """
    Compares two FileMetadata objects (old_md vs. new_md) for a single file.
    Returns a dict describing differences found in size, times, permissions,
    capabilities, and/or hashes, depending on which check_* flags are enabled.
    If none of the check_* flags are set, all fields are compared by default.
    """
    detail_changes: Dict[str, Any] = {}
    compare_all = all_not_explicit(
        check_size, check_time, check_perms, check_caps, check_hash
    )

    # Compare size
    if check_size or compare_all:
        size_diff = compare_field(old_md.size, new_md.size)
        if size_diff:
            detail_changes["size"] = size_diff

    # Compare ctime/mtime
    if check_time or compare_all:
        ctime_diff = compare_field(old_md.ctime, new_md.ctime)
        if ctime_diff:
            detail_changes["ctime"] = ctime_diff
        mtime_diff = compare_field(old_md.mtime, new_md.mtime)
        if mtime_diff:
            detail_changes["mtime"] = mtime_diff

    # Compare permissions
    if check_perms or compare_all:
        perms_diff = compare_field(old_md.permissions, new_md.permissions)
        if perms_diff:
            detail_changes["permissions"] = perms_diff

    # Compare capabilities
    if check_caps or compare_all:
        caps_diff = compare_field(old_md.capabilities, new_md.capabilities)
        if caps_diff:
            detail_changes["capabilities"] = caps_diff

    # Compare hashes
    if check_hash or compare_all:
        hash_diffs = compare_hashes(old_md.hashes, new_md.hashes)
        if hash_diffs:
            detail_changes["hashes"] = hash_diffs

    return detail_changes


def compare_field(old_val: Any, new_val: Any) -> Optional[tuple]:
    """
    Compares 'old_val' and 'new_val' for a specific metadata field.
    Returns a tuple (old_val, new_val) if they differ, otherwise None.
    """
    if old_val != new_val:
        return (old_val, new_val)
    return None


def compare_hashes(
    old_hashes: Dict[str, str], new_hashes: Dict[str, str]
) -> Dict[str, tuple]:
    """
    Compares two hash dictionaries, returning a dict of all algorithms whose values differ.
    Format: {algo: (old_val, new_val), ...}
    """
    hash_diffs: Dict[str, tuple] = {}
    all_algos = set(old_hashes.keys()) | set(new_hashes.keys())
    for algo in all_algos:
        old_val = old_hashes.get(algo)
        new_val = new_hashes.get(algo)
        if old_val != new_val:
            hash_diffs[algo] = (old_val, new_val)
    return hash_diffs


def all_not_explicit(
    check_size: bool,
    check_time: bool,
    check_perms: bool,
    check_caps: bool,
    check_hash: bool,
) -> bool:
    """
    Returns True if none of the "check_*" flags are set, meaning we check all fields by default.
    """
    return not (check_size or check_time or check_perms or check_caps or check_hash)


def expand_change_details(changes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Takes the raw 'changes' list and transforms it into a list of rows focusing
    on a single changed attribute per row. The returned structure is:
      [
        {
          'path':      <str>,
          'type':      <'added'|'removed'|'changed'>,
          'attribute': <str>,
          'old':       <str>,
          'new':       <str>
        },
        ...
      ]

    If a file was 'added' or 'removed', it will produce a single row with empty
    'attribute', 'old', and 'new' fields.
    If a file was 'changed', each differing attribute is a separate row.
    """
    expanded = []
    for c in changes:
        ctype = c["type"]
        path = c["path"]
        details = c.get("details", {})

        # If file is added/removed => single row, no attributes.
        if ctype in ("added", "removed"):
            expanded.append(
                {"path": path, "type": ctype, "attribute": "", "old": "", "new": ""}
            )
            continue

        # For changed, we create one row for each differing attribute.
        # Example detail structure:
        #   {
        #       'size': (old_size, new_size),
        #       'mtime': (old_mtime, new_mtime),
        #       'hashes': { 'md5': (old_val, new_val), 'sha512': (old_val, new_val) }
        #   }
        for attr, diff in details.items():
            if attr == "hashes":
                # For multiple changed hashes, produce multiple rows
                for algo, algo_diff in diff.items():
                    old_val, new_val = algo_diff
                    expanded.append(
                        {
                            "path": path,
                            "type": ctype,
                            "attribute": f"hash:{algo}",
                            "old": str(old_val) if old_val is not None else "",
                            "new": str(new_val) if new_val is not None else "",
                        }
                    )
            else:
                old_val, new_val = diff
                expanded.append(
                    {
                        "path": path,
                        "type": ctype,
                        "attribute": attr,
                        "old": str(old_val),
                        "new": str(new_val),
                    }
                )

    return expanded


def print_changes(changes: List[Dict[str, Any]]) -> None:
    """
    Prints the detected changes to stdout. Each changed attribute is its own row
    in the PrettyTable or in the plain text output.
    """
    if not changes:
        logging.info("No changes detected.")
        return

    # Transform the raw changes into row-based structures
    rows = expand_change_details(changes)

    if USE_PRETTYTABLE:
        table = PrettyTable()
        table.field_names = [
            "Path",
            "Change Type",
            "Attribute",
            "Old Value",
            "New Value",
        ]
        for row in rows:
            table.add_row(
                [row["path"], row["type"], row["attribute"], row["old"], row["new"]]
            )
        print(table)

    else:
        print("Changes detected:")
        for row in rows:
            print(f"Path: {row['path']}")
            print(f"  Type: {row['type']}")
            if row["attribute"]:
                print(f"  Attribute: {row['attribute']}")
                print(f"    Old: {row['old']}")
                print(f"    New: {row['new']}")
            print("")


def save_metadata_to_json(
    json_path: Path,
    metadata: Dict[str, FileMetadata],
    changes: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Saves the given metadata (and optional changes) to a JSON file at 'json_path'.
    The structure is:
        {
            "metadata": {
                "<path>": { ... },
                ...
            },
            "changes": [ ... ]
        }
    """
    data_to_save = {
        "metadata": {path_str: asdict(md) for path_str, md in metadata.items()}
    }
    if changes is not None:
        data_to_save["changes"] = changes

    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        logging.info(f"Successfully wrote metadata to '{json_path}'.")
    except OSError as e:
        logging.error(f"Failed to write metadata to '{json_path}': {e}")


def main() -> None:
    """
    Main function to orchestrate the file change checking process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    files = collect_files(
        directories=args.directories,
        recursive=args.recursive,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )

    new_metadata = gather_metadata(files, args.algorithm)

    changes: List[Dict[str, Any]] = []
    if args.input:
        old_metadata = load_metadata_from_json(Path(args.input))
        changes = detect_changes(
            old_metadata,
            new_metadata,
            check_size=args.check_size,
            check_time=args.check_time,
            check_perms=args.check_permissions,
            check_caps=args.check_capabilities,
            check_hash=args.check_hash,
        )
        print_changes(changes)

    if args.output:
        save_metadata_to_json(Path(args.output), new_metadata, changes)

    if changes:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
