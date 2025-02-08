#!/usr/bin/env python3

# -------------------------------------------------------
# Script: fooocus.py
#
# Description:
# This script runs Fooocus inside a Docker container. Fooocus is accessed
# via a web browser. By default, NVIDIA GPU support is enabled (if available).
# The script automatically downloads the Fooocus source archive and then builds
# and runs the container if no suitable image exists. You can force a rebuild
# with --no-cache, which will download and rebuild even if an image already
# exists.
#
# Usage:
#   ./fooocus.py [options]
#
# Options:
#   -r, --release RELEASE   Fooocus release/tag to download (default: 2.5.5).
#   -d, --data-dir DIR      Host directory for persistent Fooocus data (default: ~/.cache/buchwald/fooocus/data).
#   -H, --host HOST         Host/IP to bind for the published port (default: 127.0.0.1).
#   -p, --port PORT         Host port to map to container's port 7865 (default: 7865).
#   -G, --no-gpu            Disable NVIDIA GPU usage (enabled by default).
#   -N, --no-cache          Force rebuilding the Docker image, even if one exists.
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


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for building and running Fooocus.
    """
    parser = argparse.ArgumentParser(
        description="Build and run Fooocus in a Docker container."
    )
    parser.add_argument(
        "-r",
        "--release",
        default="2.5.5",
        help="Fooocus release/tag to download (default: 2.5.5).",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="~/.cache/buchwald/fooocus/data",
        help="Host directory for persistent Fooocus data (default: ~/.cache/buchwald/fooocus/data).",
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
        default="7865",
        help="Host port to map to container's port 7865 (default: 7865).",
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


def build_image(image_tag: str, release: str, verbose: bool) -> None:
    """
    Builds the Docker image for Fooocus using the specified release/tag.
    Always rebuilds, ignoring any existing image. 'image_tag' is used
    when tagging the Docker image, and 'release' is used for downloading.
    """
    logging.info(f"Building Docker image '{image_tag}' (RELEASE={release})...")

    build_context = tempfile.mkdtemp(prefix="fooocus_build_")
    try:
        archive_name = f"v{release}.tar.gz"
        archive_url = (
            f"https://github.com/lllyasviel/Fooocus/archive/refs/tags/{archive_name}"
        )
        archive_path = os.path.join(build_context, archive_name)

        logging.debug(f"Downloading Fooocus archive from {archive_url}")
        cmd_wget = ["wget", "-q", "-O", archive_path, archive_url]
        result = subprocess.run(cmd_wget)
        if result.returncode != 0:
            logging.error("Failed to download the Fooocus archive.")
            sys.exit(result.returncode)

        logging.debug("Extracting Fooocus archive...")
        cmd_tar = ["tar", "xf", archive_path, "-C", build_context]
        result = subprocess.run(cmd_tar)
        if result.returncode != 0:
            logging.error("Failed to extract the Fooocus archive.")
            sys.exit(result.returncode)

        fooocus_dir = os.path.join(build_context, f"Fooocus-{release}")

        cmd_build = ["docker", "build", "-t", image_tag, fooocus_dir]

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
    Runs the Fooocus Docker container with the given parameters.
    The container listens on port 7865 internally; we map (host:port -> 7865).
    """
    data_dir_expanded = str(Path(data_dir).expanduser())
    os.makedirs(data_dir_expanded, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-p",
        f"{host}:{port}:7865",
        "-v",
        f"{data_dir_expanded}:/content/data",
        "-e",
        "CMDARGS=--listen",  # Additional arguments for Fooocus's launch.py
        "-e",
        "DATADIR=/content/data",  # Directory for models/outputs
        "-e",
        "config_path=/content/data/config.txt",
        "-e",
        "config_example_path=/content/data/config_modification_tutorial.txt",
        "-e",
        "path_checkpoints=/content/data/models/checkpoints/",
        "-e",
        "path_loras=/content/data/models/loras/",
        "-e",
        "path_embeddings=/content/data/models/embeddings/",
        "-e",
        "path_vae_approx=/content/data/models/vae_approx/",
        "-e",
        "path_upscale_models=/content/data/models/upscale_models/",
        "-e",
        "path_inpaint=/content/data/models/inpaint/",
        "-e",
        "path_controlnet=/content/data/models/controlnet/",
        "-e",
        "path_clip_vision=/content/data/models/clip_vision/",
        "-e",
        "path_fooocus_expansion=/content/data/models/prompt_expansion/fooocus_expansion/",
        "-e",
        "path_outputs=/content/app/outputs/",
    ]

    if gpu:
        cmd.append("--gpus=all")

    cmd.append(image_tag)

    logging.debug(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def main() -> None:
    """
    Main function to orchestrate building and running the Fooocus Docker container.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    # Construct the Docker image tag once here
    image_tag = f"fooocus:{args.release}"

    # Check if the image exists
    if image_exists(image_tag):
        if args.no_cache:
            logging.info(
                f"Image '{image_tag}' already exists, but --no-cache was specified. Rebuilding."
            )
            build_image(image_tag, args.release, args.verbose)
        else:
            logging.info(f"Image '{image_tag}' already exists. Skipping rebuild.")
    else:
        logging.info(f"Image '{image_tag}' does not exist locally. Building it now...")
        build_image(image_tag, args.release, args.verbose)

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
    sys.exit(main())
