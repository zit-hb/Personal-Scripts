#!/usr/bin/env python3

# -------------------------------------------------------
# Script: manage_headers.py
#
# Description:
# Manages headers of the Python scripts.
#
# Usage:
# ./manage_headers.py [subcommand] [options]
#
# Subcommands:
#   update_pip                    Update 'pip' install commands in script headers with version numbers.
#
# Options:
#   -r, --recursive               Recursively process directories.
#   -s, --version-source SOURCE   Source of version numbers: "latest" or "installed". Default: "latest".
#   -u, --update-all              Update all packages, even those with existing versions.
#   -x, --exclude-package PKG     Exclude a package from updating. Can be used multiple times.
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#
# Template: ubuntu22.04
#
# Requirements:
# - requests (install via: pip install requests==2.32.3)
# - packaging (install via: pip install packaging==24.2)
#
# -------------------------------------------------------

import argparse
import logging
import os
import re
import shlex
import sys
import requests
from typing import List, Tuple, Optional

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

import packaging.requirements
import packaging.version
import packaging.specifiers


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Modifies pip install commands in script headers to include version numbers.'
    )
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    # Subcommand: update_pip
    update_pip_parser = subparsers.add_parser(
        'update_pip',
        help='Update pip install commands in script headers with version numbers.'
    )
    update_pip_parser.add_argument(
        'path',
        help='Path to a script file or directory to process.'
    )
    update_pip_parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Recursively process directories.'
    )
    update_pip_parser.add_argument(
        '-s',
        '--version-source',
        choices=['latest', 'installed'],
        default='latest',
        help='Source of version numbers: "latest" or "installed".'
    )
    update_pip_parser.add_argument(
        '-u',
        '--update-all',
        action='store_true',
        help='Update all packages, even those with existing versions.'
    )
    update_pip_parser.add_argument(
        '-x',
        '--exclude-package',
        action='append',
        help='Exclude a package from updating. Can be used multiple times.'
    )
    update_pip_parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    update_pip_parser.add_argument(
        '-vv',
        '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )

    args = parser.parse_args()
    return args


def setup_logging(verbose: bool, debug: bool) -> None:
    """Sets up logging configuration."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def collect_script_paths(path: str, recursive: bool) -> List[str]:
    """Collects a list of script file paths from the given path."""
    script_paths = []
    if os.path.isfile(path):
        script_paths.append(path)
    elif os.path.isdir(path):
        if recursive:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        script_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(path):
                if file.endswith('.py'):
                    script_paths.append(os.path.join(path, file))
    else:
        logging.error(f"The path '{path}' is neither a file nor a directory.")
        sys.exit(1)
    return script_paths


def read_script(script_path: str) -> Tuple[List[str], List[str]]:
    """Reads the script and returns a tuple of (header_lines, rest_lines)."""
    try:
        with open(script_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Script '{script_path}' not found.")
        sys.exit(1)
    header_lines = []
    rest_lines = []
    in_header = True
    for line in lines:
        if in_header and line.strip().startswith('#'):
            header_lines.append(line)
        elif in_header and line.strip() == '':
            header_lines.append(line)
        else:
            in_header = False
            rest_lines.append(line)
    return header_lines, rest_lines


def find_requirements_section(header_lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Finds the start and end indices of the 'Requirements' section in the header."""
    requirements_start = None
    requirements_end = None
    for idx, line in enumerate(header_lines):
        content = line.strip('#').strip()
        if content.startswith('Requirements:'):
            requirements_start = idx
            continue
        if requirements_start is not None:
            if content == '' or content.startswith('---'):
                requirements_end = idx
                break
    if requirements_start is not None and requirements_end is None:
        requirements_end = len(header_lines)
    return requirements_start, requirements_end


def extract_install_command(line: str) -> Optional[str]:
    """Extracts the pip install command from a line in the Requirements section."""
    match = re.search(r'\(install via:\s*(.*?)\)', line.strip('#').strip())
    if match:
        command = match.group(1).strip()
        tokens = shlex.split(command)
        if any(token in ('pip', 'pip3', 'python', 'python3') for token in tokens) and 'pip' in tokens:
            return command
        elif 'pip install' in command or command.startswith('pip ') or command.startswith('pip3 ') or command.startswith('python -m pip') or command.startswith('python3 -m pip'):
            return command
        else:
            return None
    else:
        return None


def update_requirements_lines(requirements_lines: List[str], args) -> List[str]:
    """Processes the requirements lines and returns updated lines."""
    updated_requirements_lines = []
    for line in requirements_lines:
        install_command = extract_install_command(line)
        if install_command:
            updated_install_command = process_pip_install_command(install_command, args)
            # Reconstruct the line with the updated install command
            new_line = re.sub(
                r'\(install via:.*?\)',
                f'(install via: {updated_install_command})',
                line
            )
            updated_requirements_lines.append(new_line)
        else:
            # If no pip install command is found, leave the line as is.
            updated_requirements_lines.append(line)
    return updated_requirements_lines


def parse_pip_install_command(install_command: str) -> Tuple[List[str], List[str], List[str]]:
    """Parses a pip install command and returns a tuple of (prefix, options, package_specifiers)."""
    tokens = shlex.split(install_command)
    package_specifiers = []
    options = []
    prefix = []
    skip_tokens = {'pip', 'pip3', 'python', 'python3'}
    options_with_args = {'-r', '--requirement', '-f', '--find-links', '-i', '--index-url', '--extra-index-url', '--trusted-host', '--cert', '--client-cert', '--proxy'}
    i = 0
    # Collect initial 'pip', 'pip3', 'python', 'python3', etc.
    while i < len(tokens) and (tokens[i] in skip_tokens or tokens[i:i+3] == ['python', '-m', 'pip']):
        prefix.append(tokens[i])
        i += 1
        if tokens[i-1] == 'python' and i < len(tokens) and tokens[i] == '-m' and tokens[i+1] == 'pip':
            prefix.extend(['-m', 'pip'])
            i += 2
    # Collect 'install'
    if i < len(tokens) and tokens[i] == 'install':
        prefix.append(tokens[i])
        i += 1
    # Now process the rest
    while i < len(tokens):
        token = tokens[i]
        if token.startswith('-'):
            options.append(token)
            if token in options_with_args:
                i += 1
                if i < len(tokens):
                    options.append(tokens[i])
                else:
                    logging.warning(f"Option '{token}' expects an argument, but none was provided.")
            i += 1
        else:
            package_specifiers.append(token)
            i += 1
    return prefix, options, package_specifiers


def parse_package_specifier(specifier: str) -> Tuple[str, Optional[str]]:
    """Parses a package specifier and returns the package name and version specifier."""
    try:
        req = packaging.requirements.Requirement(specifier)
        package_name = req.name
        version_spec = str(req.specifier) if req.specifier else None
        return package_name, version_spec
    except Exception as e:
        logging.warning(f"Failed to parse package specifier '{specifier}': {e}")
        return specifier, None


def get_latest_version(package_name: str) -> Optional[str]:
    """Gets the latest stable version of a package from PyPI."""
    try:
        url = f'https://pypi.org/pypi/{package_name}/json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            versions = list(data['releases'].keys())
            versions = [
                v for v in versions
                if not any(c in v for c in ['a', 'b', 'rc', 'dev', 'post'])
            ]
            versions.sort(key=packaging.version.parse, reverse=True)
            if versions:
                return versions[0]
        else:
            logging.error(f"Failed to get version info for package '{package_name}' from PyPI.")
    except Exception as e:
        logging.error(f"Error getting latest version for package '{package_name}': {e}")
    return None


def get_installed_version(package_name: str) -> Optional[str]:
    """Gets the installed version of a package."""
    try:
        version = importlib_metadata.version(package_name)
        return version
    except importlib_metadata.PackageNotFoundError:
        logging.error(f"Package '{package_name}' is not installed.")
        return None
    except Exception as e:
        logging.error(f"Error getting installed version for package '{package_name}': {e}")
        return None


def process_pip_install_command(install_command: str, args: argparse.Namespace) -> str:
    """Processes a pip install command, updating package versions as per the options."""
    prefix, options, package_specifiers = parse_pip_install_command(install_command)
    updated_specifiers = []
    exclude_packages = args.exclude_package if args.exclude_package else []
    for spec in package_specifiers:
        package_name, version_spec = parse_package_specifier(spec)
        if package_name in exclude_packages:
            update_package = False
        elif args.update_all:
            update_package = True
        elif not version_spec:
            update_package = True
        else:
            update_package = False
        if update_package:
            if args.version_source == 'installed':
                version = get_installed_version(package_name)
            else:
                version = get_latest_version(package_name)
            if version:
                # Preserve any extras or environment markers
                try:
                    req = packaging.requirements.Requirement(spec)
                    if req.extras:
                        extras = f"[{','.join(req.extras)}]"
                    else:
                        extras = ''
                    updated_spec = f"{package_name}{extras}=={version}"
                    updated_specifiers.append(updated_spec)
                    logging.info(f"Updated '{package_name}' to version '{version}'")
                except Exception as e:
                    updated_specifiers.append(spec)
                    logging.warning(f"Could not parse specifier '{spec}': {e}")
            else:
                updated_specifiers.append(spec)
                logging.warning(f"Could not get version for package '{package_name}', leaving as is.")
        else:
            updated_specifiers.append(spec)
    updated_command = ' '.join(prefix + options + updated_specifiers)
    return updated_command


def process_script(script_path: str, args: argparse.Namespace):
    """
    Processes a single script file to update its package versions in the Requirements section.
    """
    header_lines, rest_lines = read_script(script_path)

    requirements_start, requirements_end = find_requirements_section(header_lines)
    if requirements_start is None:
        logging.info(f"No 'Requirements' section found in script '{script_path}'. Skipping.")
        return
    requirements_lines = header_lines[requirements_start+1:requirements_end]
    updated_requirements_lines = update_requirements_lines(requirements_lines, args)
    new_header_lines = header_lines[:requirements_start+1] + updated_requirements_lines + header_lines[requirements_end:]
    with open(script_path, 'w') as f:
        f.writelines(new_header_lines + rest_lines)
    logging.info(f"Updated script '{script_path}'")


def main():
    """
    Main function to orchestrate the package management process.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    if args.subcommand == 'update_pip':
        script_paths = collect_script_paths(args.path, args.recursive)
        for script_path in script_paths:
            process_script(script_path, args)
    else:
        logging.error('Unknown subcommand.')
        sys.exit(1)


if __name__ == '__main__':
    main()
