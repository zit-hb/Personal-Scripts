#!/usr/bin/env python3

# -------------------------------------------------------
# Script: query_pypi.py
#
# Description:
#   Query and interact with the PyPI repository via its JSON API.
#
# Usage:
#   ./query_pypi.py [options] <command> [command options]
#
# Commands:
#   show        Show detailed information about a package (richly formatted).
#   list        List available versions of a package.
#   download    Download a specific version of a package.
#
# Options:
#   -u, --user-agent        Custom User-Agent string for HTTP requests.
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - requests (install via: pip install requests==2.31.0)
#   - packaging (install via: pip install packaging==23.1)
#   - rich (install via: pip install rich==13.7.1)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from packaging.version import Version

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.pretty import Pretty
from rich.text import Text
from rich.rule import Rule


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Interact with the PyPI JSON API.")
    parser.add_argument(
        "-u",
        "--user-agent",
        type=str,
        default="query_pypi.py (+https://pypi.org/)",
        help="Custom User-Agent string for HTTP requests (default: 'query_pypi.py (+https://pypi.org/)').",
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Show package information.")
    show_parser.add_argument("package", type=str, help="Name of the package.")
    show_parser.add_argument(
        "-V",
        "--version",
        type=str,
        help="Package version (default: latest).",
    )
    show_parser.add_argument(
        "-D",
        "--description",
        action="store_true",
        help="Include long description in the output.",
    )

    list_parser = subparsers.add_parser("list", help="List all versions of a package.")
    list_parser.add_argument("package", type=str, help="Name of the package.")
    list_parser.add_argument(
        "-o",
        "--order",
        choices=["release", "version"],
        default="release",
        help="Order by 'release' date or 'version' (default: release).",
    )
    list_parser.add_argument(
        "-d",
        "--direction",
        choices=["asc", "desc"],
        default="asc",
        help="Sort direction: 'asc' or 'desc' (default: asc).",
    )

    download_parser = subparsers.add_parser("download", help="Download a package file.")
    download_parser.add_argument("package", type=str, help="Name of the package.")
    download_parser.add_argument(
        "-V",
        "--version",
        type=str,
        help="Package version (default: latest).",
    )
    download_parser.add_argument(
        "-D",
        "--dest",
        type=str,
        default=".",
        help="Destination directory (default: current).",
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


def _http_get(url: str, user_agent: str, **kwargs: Any) -> requests.Response:
    """
    Wrapper around requests.get with sane defaults and error handling.
    """
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", user_agent)
    timeout = kwargs.pop("timeout", 15)
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching {url}: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching {url}: {e}")
        sys.exit(1)


def get_package_data(
    package: str, version: Optional[str], user_agent: str
) -> Dict[str, Any]:
    """
    Retrieves JSON data for a package (and optional version).
    """
    base_url = f"https://pypi.org/pypi/{package}"
    url = f"{base_url}/{version}/json" if version else f"{base_url}/json"
    logging.debug(f"Fetching PyPI JSON: {url}")
    response = _http_get(url, user_agent=user_agent)
    return response.json()


def _parse_iso8601(dt: Optional[str]) -> Optional[datetime]:
    """
    Parse PyPI's ISO 8601 timestamp into a timezone-aware datetime (UTC).
    Returns None if dt is falsy or invalid.
    """
    if not dt:
        return None
    try:
        # Examples: '2023-09-10T15:01:02.123456Z'
        # Python's fromisoformat doesn't accept trailing 'Z' directly.
        if dt.endswith("Z"):
            dt = dt[:-1] + "+00:00"
        d = datetime.fromisoformat(dt)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def _is_nonempty(value: Any) -> bool:
    """
    Check whether a value should be rendered (not None/empty).
    """
    if value is None:
        return False
    if isinstance(value, (str, bytes)) and value.strip() == "":
        return False
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return False
    return True


def _fmt_bool(value: Any) -> str:
    """
    Convert a boolean-like value into a human-readable string.
    """
    return "true" if bool(value) else "false"


def _make_scalar_panel(
    info: Dict[str, Any], scalar_keys_order: List[str]
) -> Optional[Panel]:
    """
    Create a rich Panel containing scalar fields from the package info section.
    """
    scalar_table = Table(show_header=False, expand=True, box=None, pad_edge=False)
    scalar_table.add_column(justify="right", style="bold")
    scalar_table.add_column(style="")

    for key in scalar_keys_order:
        value = info.get(key)
        if not _is_nonempty(value):
            continue
        if isinstance(value, bool):
            value_str = _fmt_bool(value)
        else:
            value_str = str(value)
        scalar_table.add_row(key.replace("_", " ").title(), value_str)

    if scalar_table.row_count:
        return Panel(scalar_table, title="Info", expand=True)
    return None


def _make_simple_list_table(title: str, items: Any, *, expand: bool) -> Optional[Table]:
    """
    Create a simple list table (e.g., for classifiers or extras).
    """
    if _is_nonempty(items) and isinstance(items, list):
        table = Table(title=title, show_header=False, expand=expand)
        table.add_column()
        for it in items:
            table.add_row(str(it))
        return table
    return None


def _make_requires_dist_table(requires_dist: Any) -> Optional[Table]:
    """
    Create a table for the 'requires_dist' (dependencies) field.
    """
    if _is_nonempty(requires_dist) and isinstance(requires_dist, list):
        req_table = Table(title="Requires Dist", show_header=False, expand=True)
        req_table.add_column()
        for r in requires_dist:
            req_table.add_row(str(r))
        return req_table
    return None


def _make_downloads_table(downloads: Any) -> Optional[Table]:
    """
    Create a table for legacy download statistics if present.
    """
    if _is_nonempty(downloads) and isinstance(downloads, dict):
        dl_table = Table(title="Downloads (legacy)", show_header=True)
        dl_table.add_column("Period", style="bold")
        dl_table.add_column("Count")
        for k, v in downloads.items():
            dl_table.add_row(str(k), str(v))
        return dl_table
    return None


def _make_description_panel(description: Any, content_type: str) -> Optional[Panel]:
    """
    Create a panel for the package description (Markdown or plain text).
    """
    if _is_nonempty(description):
        if "markdown" in (content_type or "").lower():
            return Panel(Markdown(description), title="Description", expand=True)
        else:
            return Panel(Text(description), title="Description", expand=True)
    return None


def _collect_remaining_fields(info: Dict[str, Any], shown_keys: set) -> Dict[str, Any]:
    """
    Collect any remaining non-empty fields not shown in earlier panels.
    """
    remaining: Dict[str, Any] = {}
    for k, v in info.items():
        if k in shown_keys:
            continue
        if _is_nonempty(v):
            remaining[k] = v
    return remaining


def _make_remaining_panel(remaining: Dict[str, Any]) -> Optional[Panel]:
    """
    Create a panel showing any remaining key-value pairs from package info.
    """
    if remaining:
        return Panel(
            Pretty(remaining, indent_guides=True),
            title="Other Info Fields",
            expand=True,
        )
    return None


def handle_show(
    package: str, version: Optional[str], user_agent: str, include_description: bool
) -> None:
    """
    Display full `info` section using rich, omitting unset/empty fields
    and formatting collections nicely. The description is only shown if
    include_description=True.
    """
    console = Console()
    data = get_package_data(package, version, user_agent)
    info: Dict[str, Any] = data.get("info", {}) or {}

    name = info.get("name") or package
    ver = info.get("version") or (version or "")
    title = f"[bold]{name}[/bold] {ver}".strip()
    console.print(Rule(title))

    scalar_keys_order = [
        "summary",
        "license",
        "requires_python",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "home_page",
        "project_url",
        "package_url",
        "release_url",
        "docs_url",
        "download_url",
        "bugtrack_url",
        "platform",
        "dynamic",
        "yanked",
        "yanked_reason",
    ]

    # Scalars
    scalar_panel = _make_scalar_panel(info, scalar_keys_order)
    if scalar_panel:
        console.print(scalar_panel)

    # Classifiers
    cls_table = _make_simple_list_table(
        "Classifiers", info.get("classifiers"), expand=True
    )
    if cls_table:
        console.print(cls_table)

    # Extras
    extras_table = _make_simple_list_table(
        "Extras", info.get("provides_extra"), expand=False
    )
    if extras_table:
        console.print(extras_table)

    # Requirements
    req_table = _make_requires_dist_table(info.get("requires_dist"))
    if req_table:
        console.print(req_table)

    # Downloads
    dl_table = _make_downloads_table(info.get("downloads"))
    if dl_table:
        console.print(dl_table)

    # Description (only if requested)
    if include_description:
        description_panel = _make_description_panel(
            info.get("description"),
            (info.get("description_content_type") or "").lower(),
        )
        if description_panel:
            console.print(description_panel)

    # Remaining fields (do not show project_urls anywhere)
    shown_keys: set = set(scalar_keys_order) | {
        "project_urls",
        "classifiers",
        "provides_extra",
        "requires_dist",
        "downloads",
        "description",
        "description_content_type",
        "name",
        "version",
    }
    remaining = _collect_remaining_fields(info, shown_keys)
    remaining_panel = _make_remaining_panel(remaining)
    if remaining_panel:
        console.print(remaining_panel)


def _pick_release_upload_time(files: Iterable[Dict[str, Any]]) -> Optional[datetime]:
    """
    For a release's files, find the *latest* upload timestamp as the release time.
    """
    best: Optional[datetime] = None
    for f in files or []:
        dt = _parse_iso8601(f.get("upload_time_iso_8601") or f.get("upload_time"))
        if dt and (best is None or dt > best):
            best = dt
    return best


def handle_list_versions(
    package: str, order: str, direction: str, user_agent: str
) -> None:
    """
    List all versions of a package, sorted by release date or version number.
    """
    data = get_package_data(package, None, user_agent)
    releases = data.get("releases", {}) or {}

    entries: List[Tuple[Version, Optional[datetime], str]] = []
    for ver_str, files in releases.items():
        try:
            ver_obj = Version(ver_str)
        except Exception:
            continue
        release_dt = _pick_release_upload_time(files)
        entries.append((ver_obj, release_dt, ver_str))

    if order == "release":
        entries.sort(
            key=lambda x: (
                x[1] is None,
                x[1] or datetime.min.replace(tzinfo=timezone.utc),
            )
        )
    else:
        entries.sort(key=lambda x: x[0])

    if direction == "desc":
        entries.reverse()

    for _, dt, ver_str in entries:
        dt_str = dt.isoformat() if dt else ""
        print(f"{ver_str}    {dt_str}")


def handle_download(
    package: str, version: Optional[str], destination: str, user_agent: str
) -> None:
    """
    Download a package file (wheel or sdist) from PyPI.
    """
    data = get_package_data(package, version, user_agent)
    files = data.get("urls", []) or []
    if not files:
        logging.error("No downloadable files found.")
        sys.exit(1)

    wheel = next((f for f in files if f.get("packagetype") == "bdist_wheel"), None)
    source = next((f for f in files if f.get("packagetype") == "sdist"), None)
    chosen = wheel or source or files[0]

    url = chosen.get("url")
    filename = chosen.get("filename")
    if not url or not filename:
        logging.error("Invalid file metadata from PyPI.")
        sys.exit(1)

    os.makedirs(destination, exist_ok=True)
    dest_path = os.path.join(destination, filename)
    logging.info(f"Downloading {filename} -> {dest_path}")

    with _http_get(url, user_agent=user_agent, stream=True) as resp:
        try:
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        fh.write(chunk)
        except OSError as e:
            logging.error(f"Failed to write file: {e}")
            sys.exit(1)

    logging.info("Download completed.")


def main() -> None:
    """
    Main function to orchestrate the PyPI query workflow.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.command == "show":
        handle_show(
            args.package,
            getattr(args, "version", None),
            args.user_agent,
            args.description,
        )
    elif args.command == "list":
        handle_list_versions(args.package, args.order, args.direction, args.user_agent)
    elif args.command == "download":
        handle_download(
            args.package,
            getattr(args, "version", None),
            args.dest,
            args.user_agent,
        )


if __name__ == "__main__":
    main()
