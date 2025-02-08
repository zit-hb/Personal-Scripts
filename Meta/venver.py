#!/usr/bin/env python3

# -------------------------------------------------------
# Script: venver.py
#
# Description:
# This script allows you to execute another Python script inside a virtual environment (venv).
# It can parse the header of the target script to extract pip install requirements, creates or
# reuses the specified venv, installs the requirements, and finally executes the target script
# within this environment.
#
# Usage:
#   ./venver.py [options] [target_script] -- [script_args]
#
# Arguments:
#   - [target_script]: The path to the Python script to execute inside the venv.
#   - [script_args]:   Arguments to pass to the target script.
#
# Options:
#   -V, --venv VENV_DIR      Specify the directory for the virtual environment.
#                            If not provided, a venv will be automatically created
#                            in the cache directory.
#   -c, --cache PATH         Path to a directory to use as a cache (default: ~/.cache/buchwald).
#   -s, --skip-install       Do not install any dependencies. Just use the existing venv.
#   -v, --verbose            Enable verbose logging (INFO level) and show pip output.
#   -vv, --debug             Enable debug logging (DEBUG level).
#   -N, --no-cache           Remove the existing venv directory if it exists, then create a new one.
#   -f, --force              Force removal of the existing venv directory even if it doesn't look like a venv.
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List
from dataclasses import dataclass


@dataclass
class ScriptRequirements:
    """Holds the pip install requirements parsed from a script."""

    install_commands: List[str]


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the venv script.
    """
    parser = argparse.ArgumentParser(
        description="Run a Python script in a virtual environment, installing "
        "pip requirements from the script header."
    )
    parser.add_argument(
        "target_script",
        type=str,
        nargs="?",
        help="The target Python script to execute inside the venv.",
    )
    parser.add_argument(
        "-V",
        "--venv",
        type=str,
        help=(
            "Directory for the virtual environment. If not provided, a location in the "
            "cache directory is used (subdirectory based on the script name)."
        ),
    )
    parser.add_argument(
        "-c",
        "--cache",
        type=str,
        default=os.path.expanduser("~/.cache/buchwald"),
        help="Path to a directory to use as a cache (default: ~/.cache/buchwald).",
    )
    parser.add_argument(
        "-s",
        "--skip-install",
        action="store_true",
        help="Skip installing any dependencies into the venv.",
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
        "-N",
        "--no-cache",
        action="store_true",
        help="Remove the existing venv directory if it exists, then create a new one.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force removal of the existing venv directory even if it doesn't look like a venv.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the target script inside the venv.",
    )

    args = parser.parse_args()

    # If the user used '--' before specifying the script, treat the first item in script_args
    # as the target_script if target_script wasn't set.
    if not args.target_script and args.script_args:
        args.target_script = args.script_args.pop(0)

    return args


def setup_logging(verbose: bool, debug: bool) -> None:
    """
    Sets up logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_script_header(script_path: str) -> ScriptRequirements:
    """
    Parses the script header to extract pip install commands.

    We collect all lines that start with '#', then look for a section beginning with
    'Requirements:'. We consider each subsequent line until an empty line or a line
    starting with dashes as part of the requirements, and extract install commands
    of the form '(install via: pip install X)'.
    """
    install_commands = []
    try:
        with open(script_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Target script '{script_path}' not found.")
        sys.exit(1)

    # Gather all comment lines
    header_lines = [line for line in lines if line.startswith("#")]
    requirements_started = False

    for line in header_lines:
        line_content = line.lstrip("#").strip()

        # If the line starts with "Requirements:", subsequent lines are potential requirements
        if line_content.startswith("Requirements:"):
            requirements_started = True
            continue

        if requirements_started:
            # If we encounter a blank line or a line with dashes, stop processing requirements
            if line_content == "" or line_content.startswith("-----"):
                break

            # Look for `(install via: pip install something)`
            match = re.search(r"\(install via:\s*(pip install.*?)\)", line_content)
            if match:
                cmd = match.group(1).strip()
                # Remove 'sudo' if present (just to standardize)
                if cmd.startswith("sudo "):
                    cmd = cmd[len("sudo ") :]
                logging.debug(f"Found install command: '{cmd}'")
                install_commands.append(cmd)
            else:
                logging.debug(
                    f"No '(install via: ...)' found in line: '{line_content}'"
                )

    logging.info(f"Extracted install commands: {install_commands}")
    return ScriptRequirements(install_commands=install_commands)


def is_venv_directory(path: str) -> bool:
    """
    Checks if the given path looks like a virtual environment directory.
    """
    if not os.path.isdir(path):
        return False

    # Common indicator is pyvenv.cfg:
    pyvenv_cfg = os.path.join(path, "pyvenv.cfg")
    if os.path.isfile(pyvenv_cfg):
        return True

    # Additionally check for typical python binary locations:
    python_unix = os.path.join(path, "bin", "python")
    python_windows = os.path.join(path, "Scripts", "python.exe")
    if os.path.isfile(python_unix) or os.path.isfile(python_windows):
        return True

    return False


def remove_venv_if_requested(venv_path: str, no_cache: bool, force: bool) -> None:
    """
    Removes the venv directory if --no-cache was specified and it exists.
    Only removes the directory unconditionally if --force is also set,
    otherwise checks if it looks like a venv directory.
    """
    if not no_cache:
        return

    if os.path.isdir(venv_path):
        if is_venv_directory(venv_path) or force:
            logging.info(f"Removing existing virtual environment at '{venv_path}'")
            shutil.rmtree(venv_path)
        else:
            logging.warning(
                f"'{venv_path}' does not appear to be a venv directory. "
                "Use --force to remove it anyway."
            )
    else:
        logging.debug(f"No venv directory found at '{venv_path}' to remove.")


def create_or_load_venv(
    venv_path: str,
    skip_install: bool,
    no_cache: bool,
    force: bool,
    verbose: bool,
    debug: bool,
) -> None:
    """
    Removes the venv directory if requested, then creates a new virtual environment if
    it doesn't exist, or uses the existing one. If skip_install is True, do not attempt
    to install or upgrade pip.
    """
    # Remove existing venv if --no-cache was specified
    remove_venv_if_requested(venv_path, no_cache, force)

    if not os.path.isdir(venv_path):
        logging.info(f"Creating new virtual environment at '{venv_path}'")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    else:
        logging.info(f"Using existing virtual environment at '{venv_path}'")

    if not skip_install:
        # If --verbose or --debug, show pip output. Otherwise, capture it unless there's an error.
        show_pip_output = verbose or debug

        pip_path = os.path.join(venv_path, "bin", "pip")
        if os.name == "nt":  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip.exe")

        upgrade_cmd = [pip_path, "install", "--upgrade", "pip"]

        try:
            if show_pip_output:
                subprocess.run(upgrade_cmd, check=True)
            else:
                subprocess.run(
                    upgrade_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        except subprocess.CalledProcessError as e:
            if not show_pip_output and e.stderr:
                logging.error(e.stderr)
            raise


def install_packages(
    venv_path: str, install_commands: List[str], verbose: bool, debug: bool
) -> None:
    """
    Installs packages into the venv by running the collected pip install commands.
    Only show pip output if verbose or debug is enabled, otherwise capture and show errors on failure.
    """
    if not install_commands:
        logging.debug("No packages to install.")
        return

    show_pip_output = verbose or debug
    pip_path = os.path.join(venv_path, "bin", "pip")
    if os.name == "nt":  # Windows
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")

    for cmd in install_commands:
        cmd_parts = cmd.split()
        if cmd_parts[:2] == ["pip", "install"]:
            cmd_parts[0] = pip_path
            logging.info(f"Installing with command: {' '.join(cmd_parts)}")

            try:
                if show_pip_output:
                    subprocess.run(cmd_parts, check=True)
                else:
                    subprocess.run(
                        cmd_parts,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
            except subprocess.CalledProcessError as e:
                if not show_pip_output and e.stderr:
                    logging.error(e.stderr)
                raise
        else:
            logging.warning(f"Unknown install command format, skipping: {cmd}")


def run_script_in_venv(
    venv_path: str, script_path: str, script_args: List[str], verbose: bool, debug: bool
) -> int:
    """
    Executes the script using the Python interpreter from the venv.
    Always show the script's stdout/stderr in real time.
    """
    python_path = os.path.join(venv_path, "bin", "python")
    if os.name == "nt":  # Windows
        python_path = os.path.join(venv_path, "Scripts", "python.exe")

    cmd = [python_path, script_path] + script_args
    logging.info(f"Running script with command: {' '.join(cmd)}")

    # Always show the script's output
    result = subprocess.run(cmd)
    return result.returncode


def get_venv_path(user_venv: str, cache_dir: str, script_path: str) -> str:
    """
    Determine the actual venv path based on user arguments. If the user provided
    --venv, that is returned. Otherwise, construct a path in the cache directory.

    For example, if script_path is "foo.py", the resulting venv path in the cache
    would be something like: ~/.cache/buchwald/foo/venv
    """
    if user_venv:
        return user_venv

    # If no --venv is provided, build a path inside the cache.
    # Extract script name without extension.
    base = os.path.basename(script_path)
    root, _ = os.path.splitext(base)
    return os.path.join(cache_dir, root, "venv")


def process_single_script(
    script_path: str,
    venv_path: str,
    skip_install: bool,
    no_cache: bool,
    force: bool,
    verbose: bool,
    debug: bool,
    script_args: List[str],
) -> bool:
    """
    Processes a single script: parse header, create/load venv, install packages (unless skipped),
    run script.
    """
    logging.info(f"Processing script '{script_path}'")
    reqs = parse_script_header(script_path)

    try:
        create_or_load_venv(
            venv_path=venv_path,
            skip_install=skip_install,
            no_cache=no_cache,
            force=force,
            verbose=verbose,
            debug=debug,
        )
        if not skip_install:
            install_packages(
                venv_path, reqs.install_commands, verbose=verbose, debug=debug
            )
        return (
            run_script_in_venv(venv_path, script_path, script_args, verbose, debug) == 0
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating or installing packages in venv: {e}")
        return False


def main() -> None:
    """
    Main function to orchestrate the venv wrapper process.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    if not args.target_script:
        logging.error("No target script specified.")
        sys.exit(2)

    # Determine the effective venv path: either user-specified or within the cache.
    effective_venv_path = get_venv_path(args.venv, args.cache, args.target_script)

    # Run the specified script.
    success = process_single_script(
        script_path=args.target_script,
        venv_path=effective_venv_path,
        skip_install=args.skip_install,
        no_cache=args.no_cache,
        force=args.force,
        verbose=args.verbose,
        debug=args.debug,
        script_args=args.script_args,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
