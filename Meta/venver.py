#!/usr/bin/env python3

# -------------------------------------------------------
# Script: venver.py
#
# Description:
# This script allows you to execute another Python script inside a virtual environment (venv).
# It parses the header of the target script to extract its requirements, creates or reuses the
# specified venv, installs the requirements, and finally executes the target script within this
# environment.
#
# Requirements are only installed when they changed since the last run, so repeated invocations
# of the same script start without any network round trips. Use --reinstall to force a refresh.
#
# Requirements that pip cannot satisfy (for example 'apt-get install ...') are reported as
# warnings, because a virtual environment cannot provide them. Run such scripts through
# docker.py instead.
#
# Usage:
#   ./venver.py [options] [target_script] [script_args]
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
#   -r, --reinstall          Install the requirements even if they are already up to date.
#   -v, --verbose            Enable verbose logging (INFO level).
#   -vv, --debug             Enable debug logging (DEBUG level).
#   -N, --no-cache           Remove the existing venv directory if it exists, then create a new one.
#   -f, --force              Force removal of the existing venv directory even if it doesn't look like a venv.
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List

# Name of the marker file inside a venv that records which requirements are installed.
REQUIREMENTS_STAMP = ".venver-requirements"

# Requirements starting with one of these prefixes can be installed into a venv.
PIP_COMMAND_PREFIXES = ("pip install", "pip3 install")


@dataclass
class ScriptRequirements:
    """Holds the requirements parsed from a script."""

    install_commands: List[str] = field(default_factory=list)
    unsupported_commands: List[str] = field(default_factory=list)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the venv script.
    """
    parser = argparse.ArgumentParser(
        description="Run a Python script in a virtual environment, installing "
        "pip requirements from the script header."
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
        "-r",
        "--reinstall",
        action="store_true",
        help="Install the requirements even if they are already up to date.",
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
        "target_script_and_args",
        nargs=argparse.REMAINDER,
        help="The path to the Python script and any arguments to pass to it.",
    )

    return parser.parse_args()


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


def iter_header_lines(lines: List[str]) -> List[str]:
    """
    Returns the comment lines that make up the script header.

    The header is everything before the first line of actual code, so comments that
    happen to appear further down in the script are never mistaken for requirements.
    """
    header_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#!"):
            continue
        if not stripped.startswith("#"):
            break
        header_lines.append(stripped)
    return header_lines


def parse_script_header(script_path: str) -> ScriptRequirements:
    """
    Parses the script header to extract its requirements.

    We look for a section beginning with 'Requirements:' and consider each subsequent line
    until an empty comment or a line starting with dashes as part of the requirements.
    Install commands are given in the form '(install via: pip install X)'. Commands that pip
    cannot run are collected separately so that the caller can report them.
    """
    requirements = ScriptRequirements()
    try:
        with open(script_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logging.error(f"Could not read target script '{script_path}': {e}")
        sys.exit(1)

    requirements_started = False
    for line in iter_header_lines(lines):
        line_content = line.lstrip("#").strip()

        # If the line starts with "Requirements:", subsequent lines are potential requirements
        if line_content.startswith("Requirements:"):
            requirements_started = True
            continue

        if not requirements_started:
            continue

        # If we encounter an empty comment or a line with dashes, stop processing requirements
        if line_content == "" or line_content.startswith("-----"):
            break

        # Look for `(install via: ...)`
        match = re.search(r"\(install via:\s*(.*?)\)", line_content)
        if not match:
            logging.debug(f"No '(install via: ...)' found in line: '{line_content}'")
            continue

        cmd = match.group(1).strip()
        # Remove 'sudo' if present (just to standardize)
        if cmd.startswith("sudo "):
            cmd = cmd[len("sudo ") :]

        if cmd.startswith(PIP_COMMAND_PREFIXES):
            logging.debug(f"Found install command: '{cmd}'")
            requirements.install_commands.append(cmd)
        else:
            logging.debug(f"Found requirement that pip cannot install: '{cmd}'")
            requirements.unsupported_commands.append(cmd)

    logging.info(f"Extracted install commands: {requirements.install_commands}")
    return requirements


def warn_about_unsupported_requirements(requirements: ScriptRequirements) -> None:
    """
    Warns about requirements that cannot be installed into a virtual environment.
    """
    if not requirements.unsupported_commands:
        return

    logging.warning(
        "The script declares %d requirement(s) that pip cannot install. The script may "
        "fail unless they are already present on this system. Consider running it through "
        "docker.py instead.",
        len(requirements.unsupported_commands),
    )
    for cmd in requirements.unsupported_commands:
        logging.warning(f"  Not installed: {cmd}")


def venv_bin_dir(venv_path: str) -> str:
    """
    Returns the directory holding the executables of the given venv.
    """
    return os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin")


def venv_executable(venv_path: str, name: str) -> str:
    """
    Returns the path to an executable inside the given venv.
    """
    if os.name == "nt":
        name += ".exe"
    return os.path.join(venv_bin_dir(venv_path), name)


def is_venv_directory(path: str) -> bool:
    """
    Checks if the given path looks like a virtual environment directory.
    """
    if not os.path.isdir(path):
        return False

    # Common indicator is pyvenv.cfg:
    if os.path.isfile(os.path.join(path, "pyvenv.cfg")):
        return True

    # Additionally check for typical python binary locations:
    python_unix = os.path.join(path, "bin", "python")
    python_windows = os.path.join(path, "Scripts", "python.exe")
    return os.path.isfile(python_unix) or os.path.isfile(python_windows)


def remove_venv_if_requested(venv_path: str, no_cache: bool, force: bool) -> None:
    """
    Removes the venv directory if --no-cache was specified and it exists.
    Only removes the directory unconditionally if --force is also set,
    otherwise checks if it looks like a venv directory.
    """
    if not no_cache:
        return

    if not os.path.isdir(venv_path):
        logging.debug(f"No venv directory found at '{venv_path}' to remove.")
        return

    if is_venv_directory(venv_path) or force:
        logging.info(f"Removing existing virtual environment at '{venv_path}'")
        shutil.rmtree(venv_path)
    else:
        logging.warning(
            f"'{venv_path}' does not appear to be a venv directory. "
            "Use --force to remove it anyway."
        )


def create_or_load_venv(venv_path: str, no_cache: bool, force: bool) -> None:
    """
    Removes the venv directory if requested, then creates a new virtual environment if
    it doesn't exist, or uses the existing one.
    """
    remove_venv_if_requested(venv_path, no_cache, force)

    if os.path.isdir(venv_path):
        logging.info(f"Using existing virtual environment at '{venv_path}'")
        return

    logging.info(f"Creating new virtual environment at '{venv_path}'")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)


def run_pip_command(cmd: List[str], show_pip_output: bool) -> None:
    """
    Runs a pip command, showing its output only when requested. On failure the captured
    output is logged so that errors are never silently swallowed.
    """
    if show_pip_output:
        subprocess.run(cmd, check=True)
        return

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            logging.error(e.stdout.strip())
        if e.stderr:
            logging.error(e.stderr.strip())
        raise


def calculate_requirements_digest(install_commands: List[str]) -> str:
    """
    Returns a digest identifying the given set of requirements. The interpreter version is
    part of the digest so that a Python upgrade triggers a fresh install.
    """
    payload = "\n".join([sys.version, *install_commands])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_requirements_stamp(venv_path: str) -> str:
    """
    Returns the digest of the requirements currently installed in the venv, or an empty
    string if it is unknown.
    """
    stamp_path = os.path.join(venv_path, REQUIREMENTS_STAMP)
    try:
        with open(stamp_path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def write_requirements_stamp(venv_path: str, digest: str) -> None:
    """
    Records which requirements are installed in the venv.
    """
    stamp_path = os.path.join(venv_path, REQUIREMENTS_STAMP)
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(digest)
    except OSError as e:
        logging.warning(f"Could not write the requirements stamp: {e}")


def clear_requirements_stamp(venv_path: str) -> None:
    """
    Removes the requirements stamp so that an interrupted install is not mistaken for a
    complete one on the next run.
    """
    stamp_path = os.path.join(venv_path, REQUIREMENTS_STAMP)
    try:
        os.remove(stamp_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logging.warning(f"Could not remove the requirements stamp: {e}")


def install_packages(
    venv_path: str, install_commands: List[str], reinstall: bool, show_pip_output: bool
) -> None:
    """
    Installs the requirements into the venv, skipping the work entirely when the venv
    already holds exactly these requirements.
    """
    digest = calculate_requirements_digest(install_commands)
    if not reinstall and read_requirements_stamp(venv_path) == digest:
        logging.info("Requirements are already up to date, skipping installation.")
        return

    # Invalidate the stamp first so that an interrupted install is detected next time.
    clear_requirements_stamp(venv_path)

    pip_path = venv_executable(venv_path, "pip")
    logging.info("Upgrading pip in the virtual environment")
    run_pip_command([pip_path, "install", "--upgrade", "pip"], show_pip_output)

    for cmd in install_commands:
        cmd_parts = shlex.split(cmd)
        # Replace the leading 'pip'/'pip3' with the pip of this venv.
        cmd_parts[0] = pip_path
        logging.info(f"Installing with command: {shlex.join(cmd_parts)}")
        run_pip_command(cmd_parts, show_pip_output)

    write_requirements_stamp(venv_path, digest)


def build_script_environment(venv_path: str) -> dict:
    """
    Returns the environment for the target script, with the venv activated. This makes
    tools that the script shells out to resolve to the venv as well.
    """
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = os.path.abspath(venv_path)
    env["PATH"] = os.pathsep.join(
        [venv_bin_dir(os.path.abspath(venv_path)), env.get("PATH", "")]
    )
    # PYTHONHOME would override the venv, so make sure it is not inherited.
    env.pop("PYTHONHOME", None)
    return env


def run_script_in_venv(venv_path: str, script_path: str, script_args: List[str]) -> int:
    """
    Executes the script using the Python interpreter from the venv.
    Always show the script's stdout/stderr in real time.
    """
    python_path = venv_executable(venv_path, "python")
    cmd = [python_path, script_path, *script_args]
    logging.info(f"Running script with command: {shlex.join(cmd)}")

    # Always show the script's output
    result = subprocess.run(cmd, env=build_script_environment(venv_path), check=False)
    return result.returncode


def get_venv_path(user_venv: str, cache_dir: str, script_path: str) -> str:
    """
    Determine the actual venv path based on user arguments. If the user provided
    --venv, that is returned. Otherwise, construct a path in the cache directory.

    The directory name contains a digest of the resolved script path so that two scripts
    that merely share a file name do not end up sharing a venv.

    For example, if script_path is "foo.py", the resulting venv path in the cache
    would be something like: ~/.cache/buchwald/foo-1a2b3c4d5e6f/venv
    """
    if user_venv:
        return user_venv

    # If no --venv is provided, build a path inside the cache.
    resolved = os.path.realpath(script_path)
    root, _ = os.path.splitext(os.path.basename(resolved))
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"{root}-{digest}", "venv")


def process_single_script(
    script_path: str,
    venv_path: str,
    skip_install: bool,
    reinstall: bool,
    no_cache: bool,
    force: bool,
    show_pip_output: bool,
    script_args: List[str],
) -> int:
    """
    Processes a single script: parse header, create/load venv, install packages (unless skipped),
    run script. Returns the exit code of the target script, or a non-zero code if the
    environment could not be prepared.
    """
    logging.info(f"Processing script '{script_path}'")
    reqs = parse_script_header(script_path)
    warn_about_unsupported_requirements(reqs)

    try:
        create_or_load_venv(venv_path=venv_path, no_cache=no_cache, force=force)
        if not skip_install:
            install_packages(
                venv_path, reqs.install_commands, reinstall, show_pip_output
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating or installing packages in venv: {e}")
        return 1

    return run_script_in_venv(venv_path, script_path, script_args)


def main() -> None:
    """
    Main function to orchestrate the venv wrapper process.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    # Ensure we have something to run
    if not args.target_script_and_args:
        logging.error("No target script specified.")
        sys.exit(2)

    # Remove a leading '--' if present (so that all following args go to the script)
    if args.target_script_and_args[0] == "--":
        args.target_script_and_args = args.target_script_and_args[1:]
        if not args.target_script_and_args:
            logging.error("No target script specified.")
            sys.exit(2)

    # Split off the first as the script path, the rest as script args
    target_script = args.target_script_and_args[0]
    script_args = args.target_script_and_args[1:]

    # Determine the effective venv path: either user-specified or within the cache.
    effective_venv_path = get_venv_path(args.venv, args.cache, target_script)

    # Run the specified script.
    exit_code = process_single_script(
        script_path=target_script,
        venv_path=effective_venv_path,
        skip_install=args.skip_install,
        reinstall=args.reinstall,
        no_cache=args.no_cache,
        force=args.force,
        show_pip_output=args.verbose or args.debug,
        script_args=script_args,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
