#!/usr/bin/env python3

# -------------------------------------------------------
# Script: stable-diffusion-webui.py
#
# Description:
# This script runs AUTOMATIC1111's stable-diffusion-webui inside a Docker
# container, accessed via a web browser. By default, NVIDIA GPU support is
# enabled (if available). The script automatically downloads the
# stable-diffusion-webui source archive and then builds and runs the container
# if no suitable image exists. You can force a rebuild with --no-cache, which
# will download and rebuild even if an image already exists.
#
# Usage:
#   ./stable-diffusion-webui.py [options]
#
# Options:
#   -r, --release RELEASE   stable-diffusion-webui release/tag to download
#                           (default: 1.10.0).
#   -d, --data-dir DIR      Host directory for stable-diffusion-webui code/data
#                           (default: ~/.cache/buchwald/stable-diffusion-webui/data).
#   -u, --user-id UID       UID of the 'stable' user inside the container
#                           (default: current user's UID).
#   -H, --host HOST         Host/IP to bind for the published port
#                           (default: 127.0.0.1).
#   -p, --port PORT         Host port to map to container's port 7860
#                           (default: 7860).
#   -G, --no-gpu            Disable NVIDIA GPU usage (enabled by default).
#   -N, --no-cache          Force rebuilding the Docker image, even if one
#                           exists.
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import sys
import subprocess
import os
import tempfile
import shutil
from pathlib import Path

DOCKERFILE_CONTENT = """\
# Dockerfile for stable-diffusion-webui
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG UID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    bc \\
    wget \\
    git \\
    python3 \\
    python3-venv \\
    libgl1 \\
    libglib2.0-0 \\
    google-perftools \\
 && rm -rf /var/lib/apt/lists/*

# Create a user 'stable' with a configurable UID and home directory
RUN useradd -u ${UID} -m -s /bin/bash stable

# Set work directory (where we'll mount stable-diffusion-webui code/data)
WORKDIR /opt/stable-diffusion-webui

# Switch to user 'stable' for runtime
USER stable

# By default, start the webui
CMD ["bash", "./webui.sh"]
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for building and running stable-diffusion-webui.
    """
    parser = argparse.ArgumentParser(
        description="Build and run stable-diffusion-webui in a Docker container."
    )
    parser.add_argument(
        "-r",
        "--release",
        default="1.10.0",
        help="stable-diffusion-webui release/tag to download (default: 1.10.0).",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="~/.cache/buchwald/stable-diffusion-webui/data",
        help=(
            "Host directory for stable-diffusion-webui code/data "
            "(default: ~/.cache/buchwald/stable-diffusion-webui/data)."
        ),
    )
    parser.add_argument(
        "-u",
        "--user-id",
        default=str(os.getuid()),
        help=(
            "UID of the 'stable' user inside the container "
            "(default: current user's UID)."
        ),
    )
    parser.add_argument(
        "-H",
        "--host",
        default="127.0.0.1",
        help="Host/IP to bind for the published port (default: 127.0.0.1).",
    )
    parser.add_argument(
        "-p",
        "--port",
        default="7860",
        help="Host port to map to container's port 7860 (default: 7860).",
    )
    parser.add_argument(
        "-G",
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable NVIDIA GPU usage (enabled by default).",
    )
    parser.add_argument(
        "-N",
        "--no-cache",
        action="store_true",
        help="Force rebuilding the Docker image, even if one exists.",
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

    # If --debug is set, it overrides --verbose
    if args.debug:
        args.verbose = True

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


def image_exists(image_tag: str) -> bool:
    """
    Checks if a Docker image with the given tag already exists locally.
    Returns True if it does, False otherwise.
    """
    logging.debug(f"Checking if image '{image_tag}' exists locally...")
    cmd = ["docker", "images", "-q", image_tag]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # If the command failed for some reason, assume image doesn't exist
        logging.debug("Failed to check Docker images; assuming image does not exist.")
        return False

    return bool(result.stdout.strip())


def download_and_unpack_webui(release: str, data_dir: str) -> None:
    """
    Downloads and unpacks stable-diffusion-webui (for the given release)
    to the specified data directory.
    After unpacking, modifies webui-user.sh to set clone_dir=".".
    """
    # Expand user in data_dir
    data_dir_expanded = str(Path(data_dir).expanduser())
    os.makedirs(data_dir_expanded, exist_ok=True)

    # Check if webui.sh already exists in data_dir (a simple check to see if we need to re-download)
    webui_script_path = os.path.join(data_dir_expanded, "webui.sh")
    if os.path.isfile(webui_script_path):
        logging.info("webui.sh already exists in data directory, skipping download.")
        return

    archive_name = f"v{release}.tar.gz"
    archive_url = f"https://github.com/AUTOMATIC1111/stable-diffusion-webui/archive/refs/tags/{archive_name}"

    logging.info(f"Downloading stable-diffusion-webui from {archive_url}...")
    with tempfile.TemporaryDirectory(prefix="sd_webui_download_") as tmp_dir:
        archive_path = os.path.join(tmp_dir, archive_name)

        cmd_wget = ["wget", "-q", "-O", archive_path, archive_url]
        result = subprocess.run(cmd_wget)
        if result.returncode != 0:
            logging.error("Failed to download the stable-diffusion-webui archive.")
            sys.exit(result.returncode)

        logging.info("Unpacking stable-diffusion-webui archive...")
        cmd_tar = [
            "tar",
            "xf",
            archive_path,
            "--strip-components=1",
            "-C",
            data_dir_expanded,
        ]
        result = subprocess.run(cmd_tar)
        if result.returncode != 0:
            logging.error("Failed to extract the stable-diffusion-webui archive.")
            sys.exit(result.returncode)

    # Modify webui-user.sh so that #clone_dir="stable-diffusion-webui" is changed to clone_dir="."
    webui_user_sh_path = os.path.join(data_dir_expanded, "webui-user.sh")
    if os.path.isfile(webui_user_sh_path):
        logging.info("Updating 'clone_dir' in webui-user.sh to '.'")
        with open(webui_user_sh_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(
            '#clone_dir="stable-diffusion-webui"', 'clone_dir="."'
        )
        with open(webui_user_sh_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        logging.warning("webui-user.sh not found; skipping 'clone_dir' update.")


def build_image(image_tag: str, user_id: str, no_cache: bool, verbose: bool) -> None:
    """
    Builds the Docker image for stable-diffusion-webui using a minimal Dockerfile
    that installs dependencies, creates a 'stable' user with the specified UID,
    and sets up the environment. The 'no_cache' parameter forces rebuilding.
    """
    logging.info(f"Building Docker image '{image_tag}' (UID={user_id})...")

    build_context = tempfile.mkdtemp(prefix="sd_webui_build_")
    try:
        # Write our Dockerfile
        dockerfile_path = os.path.join(build_context, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as df:
            df.write(DOCKERFILE_CONTENT)

        cmd_build = [
            "docker",
            "build",
            "-t",
            image_tag,
            "--build-arg",
            f"UID={user_id}",
            build_context,
        ]
        if no_cache:
            cmd_build.insert(2, "--no-cache")

        if verbose:
            logging.debug(f"Running command: {' '.join(cmd_build)}")
            result = subprocess.run(cmd_build)
        else:
            logging.debug(f"Running command (no verbose output): {' '.join(cmd_build)}")
            result = subprocess.run(cmd_build, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error("Failed to build Docker image.")
            if not verbose:
                logging.error(result.stderr)
            sys.exit(result.returncode)

        logging.info(f"Successfully built Docker image '{image_tag}'.")

    finally:
        shutil.rmtree(build_context, ignore_errors=True)


def run_container(
    image_tag: str, data_dir: str, gpu: bool, host: str, port: str
) -> None:
    """
    Runs the stable-diffusion-webui Docker container with the given parameters.
    The container listens on port 7860 internally; we map (host:port -> 7860).
    The stable-diffusion-webui code/data is mounted from data_dir -> /opt/stable-diffusion-webui.
    """
    data_dir_expanded = str(Path(data_dir).expanduser())
    os.makedirs(data_dir_expanded, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-p",
        f"{host}:{port}:7860",
        "-v",
        f"{data_dir_expanded}:/opt/stable-diffusion-webui",
        "-e",
        "COMMANDLINE_ARGS=--listen",
    ]

    if gpu:
        cmd.append("--gpus=all")

    cmd.append(image_tag)

    logging.debug(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def main() -> None:
    """
    Main function to orchestrate building and running the stable-diffusion-webui
    Docker container.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    image_tag = f"stable-diffusion-webui:{args.release}"

    # 1) Download stable-diffusion-webui (if needed) to data_dir
    download_and_unpack_webui(release=args.release, data_dir=args.data_dir)

    # 2) Check if the image exists
    if image_exists(image_tag):
        if args.no_cache:
            logging.info(
                f"Image '{image_tag}' already exists, but --no-cache was specified. Rebuilding."
            )
            build_image(
                image_tag=image_tag,
                user_id=args.user_id,
                no_cache=args.no_cache,
                verbose=args.verbose,
            )
        else:
            logging.info(f"Image '{image_tag}' already exists. Skipping rebuild.")
    else:
        logging.info(f"Image '{image_tag}' does not exist locally. Building it now...")
        build_image(
            image_tag=image_tag,
            user_id=args.user_id,
            no_cache=args.no_cache,
            verbose=args.verbose,
        )

    # 3) Run the container
    try:
        run_container(
            image_tag=image_tag,
            data_dir=args.data_dir,
            gpu=args.gpu,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")


if __name__ == "__main__":
    main()
