#!/usr/bin/env python3

# -------------------------------------------------------
# Script: lm-studio.py
#
# Description:
# This script runs LM-Studio inside a Docker container. LM-Studio is a graphical
# user interface application for Large Language Models. By default, NVIDIA GPU
# support is enabled (if available). The script automatically sets up the environment
# so that the GUI can be displayed on your host system. It temporarily configures
# xhost to allow the Docker container to connect to the local X server, and reverts
# this change once the container stops.
#
# Warning about xhost:
#   This script may run 'xhost +local:...' by default to allow the Docker
#   container to connect to your local X server. This can have security
#   implications, so use it with caution. You can disable this behavior
#   with the --no-xhost / -X option. If xhost was changed, it will automatically
#   be reverted once the Docker container stops.
#
# Usage:
# ./lm-studio.py [options]
#
# Options:
#   -a, --arch ARCH           LM Studio architecture (default: x64)
#   -r, --release RELEASE     LM Studio version (default: 0.3.9-6)
#   -d, --data-dir DIR        Host directory for persistent LM Studio data (default: ~/.lmstudio)
#   -u, --user-id UID         UID of the 'lmstudio' user inside the container (default: current user's UID)
#   -G, --no-gpu              Disable NVIDIA GPU usage (enabled by default)
#   -s, --skip-build          Skip building the Docker image (use existing image)
#   -x, --xhost-name NAME     Argument to pass to 'xhost +local:NAME' (default: docker)
#   -X, --no-xhost            Do not run 'xhost +local:NAME' to allow local connections
#   -v, --verbose             Enable verbose logging (INFO level)
#   -vv, --debug              Enable debug logging (DEBUG level)
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

DOCKERFILE_CONTENT = r"""\
# Use Ubuntu 22.04 as the base image.
FROM ubuntu:22.04

# Declare build-time variables.
ARG BUILD_ARCH=x64
ARG RELEASE=0.3.9-6
ARG USER_ID=1000

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install required packages.
# Note: dbus is added here to reduce errors like "Failed to connect to the bus".
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      ca-certificates \
      dbus \
      libfuse2 \
      libglib2.0-0 \
      libnss3-dev \
      libgdk-pixbuf2.0-dev \
      libgtk-3-dev \
      libxss-dev \
      libasound2 \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create an application directory, add a non-root user with configurable UID, and ensure ownership.
RUN mkdir -p /app && \
    useradd -u ${USER_ID} -ms /bin/bash lmstudio && \
    chown -R lmstudio:lmstudio /app && \
    mkdir -p /home/lmstudio/.lmstudio && \
    chown -R lmstudio:lmstudio /home/lmstudio

# Declare /home/lmstudio/.lmstudio as a volume.
VOLUME /home/lmstudio/.lmstudio

# Set the working directory.
WORKDIR /app

# Download the LM Studio AppImage, rename it, and make it executable.
RUN wget "https://installers.lmstudio.ai/linux/${BUILD_ARCH}/${RELEASE}/LM-Studio-${RELEASE}-${BUILD_ARCH}.AppImage" && \
    mv "LM-Studio-${RELEASE}-${BUILD_ARCH}.AppImage" LM-Studio.AppImage && \
    chmod +x LM-Studio.AppImage

# Switch to the non-root user.
USER lmstudio

# Set the entrypoint so that the AppImage is executed with the desired options.
ENTRYPOINT ["./LM-Studio.AppImage", "--appimage-extract-and-run", "--no-sandbox"]
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for building and running LM Studio.
    """
    user_id_default = os.getuid()

    parser = argparse.ArgumentParser(
        description="Build and run LM Studio in a Docker container with GUI support."
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="x64",
        help="LM Studio architecture (default: x64).",
    )
    parser.add_argument(
        "-r",
        "--release",
        default="0.3.9-6",
        help="LM Studio version (default: 0.3.9-6).",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default="~/.lmstudio",
        help="Host directory for persistent LM Studio data (default: ~/.lmstudio).",
    )
    parser.add_argument(
        "-u",
        "--user-id",
        type=int,
        default=user_id_default,
        help="UID of the 'lmstudio' user inside the container (default: current user's UID).",
    )
    parser.add_argument(
        "-G",
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable NVIDIA GPU usage (enabled by default).",
    )
    parser.add_argument(
        "-s",
        "--skip-build",
        action="store_true",
        help="Skip building the Docker image (use existing image).",
    )
    parser.add_argument(
        "-x",
        "--xhost-name",
        default="docker",
        help="Argument to pass to 'xhost +local:NAME' (default: docker).",
    )
    parser.add_argument(
        "-X",
        "--no-xhost",
        action="store_true",
        help="Do not run 'xhost +local:NAME' to allow local connections.",
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


def build_image(arch: str, release: str, user_id: int, verbose: bool = False) -> None:
    """
    Builds the Docker image for LM Studio using the specified architecture, release, and user ID.
    """
    image_tag = f"lm-studio:{release}-{arch}"
    logging.info(
        f"Building Docker image '{image_tag}' (ARCH={arch}, RELEASE={release}, UID={user_id})..."
    )

    # Create a temporary directory for the Docker build context
    build_context = tempfile.mkdtemp(prefix="lmstudio_build_")

    try:
        # Write the Dockerfile into the temp directory
        dockerfile_path = os.path.join(build_context, "Dockerfile")
        with open(dockerfile_path, "w") as df:
            df.write(DOCKERFILE_CONTENT)

        # Prepare the build command
        cmd = [
            "docker",
            "build",
            "--build-arg",
            f"BUILD_ARCH={arch}",
            "--build-arg",
            f"RELEASE={release}",
            "--build-arg",
            f"USER_ID={user_id}",
            "-t",
            image_tag,
            build_context,
        ]

        # Run the build command
        if verbose:
            # Show build logs in real-time
            logging.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logging.error("Failed to build Docker image.")
                sys.exit(result.returncode)
        else:
            # Only show logs if there's an error
            logging.debug(f"Running command (no verbose output): {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error("Failed to build Docker image.")
                logging.error(result.stderr)
                sys.exit(result.returncode)

        logging.info(f"Successfully built Docker image '{image_tag}'.")

    finally:
        # Clean up the temporary build context
        shutil.rmtree(build_context, ignore_errors=True)


def ensure_xhost(xhost_name: str) -> None:
    """
    Ensures xhost is configured to allow Docker containers to connect to the X server.
    """
    try:
        logging.debug("Configuring xhost to allow Docker containers to display GUI...")
        subprocess.run(["xhost", f"+local:{xhost_name}"], check=True)
    except FileNotFoundError:
        logging.warning("xhost command not found. GUI applications may not display.")
    except subprocess.CalledProcessError:
        logging.warning("Failed to configure xhost. GUI applications may not display.")


def revert_xhost(xhost_name: str) -> None:
    """
    Reverts xhost permissions that were previously granted, removing the local connection for Docker.
    """
    try:
        logging.debug(
            "Removing xhost permission for Docker containers to display GUI..."
        )
        subprocess.run(["xhost", f"-local:{xhost_name}"], check=True)
    except FileNotFoundError:
        logging.warning("xhost command not found. Cannot remove GUI permissions.")
    except subprocess.CalledProcessError:
        logging.warning(
            "Failed to remove xhost permission. GUI permissions may remain."
        )


def run_container(
    arch: str, release: str, data_dir: str, gpu: bool, user_id: int
) -> None:
    """
    Runs the LM Studio Docker container with the given parameters.
    """
    image_tag = f"lm-studio:{release}-{arch}"

    # Expand ~ in data_dir
    data_dir_expanded = str(Path(data_dir).expanduser())
    os.makedirs(data_dir_expanded, exist_ok=True)

    # Construct the base docker run command
    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-e",
        f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
        "-v",
        "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v",
        f"{data_dir_expanded}:/home/lmstudio/.lmstudio",
    ]

    # If GPU is enabled, add the necessary flags
    if gpu:
        cmd.append("--gpus=all")

    cmd.append(image_tag)

    logging.debug(f"Running command: {' '.join(cmd)}")

    # We want the GUI app's stdout/stderr to be shown in our console, so do not capture output.
    subprocess.run(cmd)


def main() -> None:
    """
    Main function to orchestrate building and running the LM Studio Docker container.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    # Build the Docker image unless --skip-build is provided
    if not args.skip_build:
        build_image(args.arch, args.release, args.user_id, verbose=args.verbose)
    else:
        logging.info("Skipping Docker image build (using existing image).")

    # Keep track of whether we changed xhost so we can revert it
    changed_xhost = False

    try:
        # Optionally configure xhost
        if not args.no_xhost:
            ensure_xhost(args.xhost_name)
            changed_xhost = True

        # Run the container
        run_container(
            arch=args.arch,
            release=args.release,
            data_dir=args.data_dir,
            gpu=args.gpu,
            user_id=args.user_id,
        )

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")

    finally:
        # Revert xhost changes if we made any
        if changed_xhost:
            revert_xhost(args.xhost_name)


if __name__ == "__main__":
    sys.exit(main())
