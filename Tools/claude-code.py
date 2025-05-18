#!/usr/bin/env python3

# -------------------------------------------------------
# Script: claude-code.py
#
# Description:
# Runs the Anthropic Claude CLI tool inside a Docker container.
#
# Usage:
#   ./claude-code.py [options] -- [claude arguments]
#
# Arguments:
#   - claude_args: Arguments passed to the claude-code CLI.
#
# Options:
#   -V, --version VERSION     Claude Code CLI version to install (default: 0.2.117).
#   -m, --mount VOLUME        Bind-mount a volume (format: host_path:container_path).
#                             Can be specified multiple times.
#   -d, --data PATH           Shortcut for --mount PATH:/data.
#   -H, --claude-home PATH    Directory to mount as Claude's home (default: ~/.cache/buchwald/claude-code/home).
#   -u, --uid UID             User ID to run the container as inside Docker (default: current user).
#   -e, --env ENV             Set environment variable in the container (format: VAR=value).
#                             Can be specified multiple times.
#   -v, --verbose             Enable verbose logging (INFO level).
#   -vv, --debug              Enable debug logging (DEBUG level).
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from typing import List

DOCKERFILE_TEMPLATE: str = """FROM node:20.19.2-bullseye-slim
RUN apt-get update && apt-get install -y --no-install-recommends git
RUN mkdir /data
WORKDIR /data
RUN npm install -g @anthropic-ai/claude-code@{version}
ENTRYPOINT ["claude"]
CMD ["--help"]
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Anthropic Claude Code CLI inside Docker."
    )
    parser.add_argument(
        "-V",
        "--version",
        type=str,
        default="0.2.117",
        help="Claude Code CLI version to install (default: 0.2.117).",
    )
    parser.add_argument(
        "-m",
        "--mount",
        action="append",
        dest="volumes",
        default=[],
        metavar="VOLUME",
        help="Bind-mount a volume (format: host_path:container_path). "
        "Can be specified multiple times.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        metavar="PATH",
        help="Shortcut for --mount PATH:/data.",
    )
    parser.add_argument(
        "-H",
        "--claude-home",
        type=str,
        default=os.path.expanduser("~/.cache/buchwald/claude-code/home"),
        metavar="PATH",
        help="Directory to mount as Claude's home (default: ~/.cache/buchwald/claude-code/home).",
    )
    parser.add_argument(
        "-u",
        "--uid",
        type=int,
        default=os.geteuid(),
        help="User ID to run the container as inside Docker (default: current user).",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        dest="envs",
        default=[],
        metavar="ENV",
        help="Set environment variable in the container (format: VAR=value). "
        "Can be specified multiple times.",
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
        dest="debug",
        help="Enable debug logging (DEBUG level).",
    )
    parser.add_argument(
        "claude_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the claude CLI.",
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


def build_docker_image(
    dockerfile_content: str, image_tag: str, debug: bool = False
) -> bool:
    """
    Builds the Docker image with the given tag.
    Only shows output if debug is True.
    """
    try:
        with tempfile.TemporaryDirectory() as build_dir:
            dockerfile_path = os.path.join(build_dir, "Dockerfile")
            with open(dockerfile_path, "w") as dockerfile:
                dockerfile.write(dockerfile_content)
            logging.info("Building Docker image '%s'.", image_tag)
            stdout = sys.stdout if debug else subprocess.DEVNULL
            stderr = sys.stderr if debug else subprocess.DEVNULL
            subprocess.run(
                ["docker", "build", "-t", image_tag, build_dir],
                check=True,
                stdout=stdout,
                stderr=stderr,
            )
        return True
    except subprocess.CalledProcessError as error:
        logging.error("Docker build failed: %s", error)
        return False


def run_claude_container(
    image_tag: str,
    claude_args: List[str],
    volumes: List[str],
    user_id: int,
    envs: List[str],
) -> int:
    """
    Runs the Claude Code CLI inside the Docker container.
    """
    cmd: List[str] = ["docker", "run", "--rm", "-it", "-u", str(user_id)]
    for volume in volumes:
        cmd.extend(["-v", volume])
    for env in envs:
        cmd.extend(["-e", env])
    cmd.extend([image_tag] + claude_args)
    logging.info("Running Claude Code CLI: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as error:
        logging.error("Failed to run container: %s", error)
        return 1


def main() -> None:
    """
    Main function to orchestrate the Dockerized Claude Code CLI workflow.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.data:
        args.volumes.append(f"{args.data}:/data")

    # Ensure claude-home directory exists
    claude_home_host = os.path.abspath(os.path.expanduser(args.claude_home))
    os.makedirs(claude_home_host, exist_ok=True)
    claude_home_container = "/home/user"
    # Mount claude-home as container's home, and set HOME env
    args.volumes.append(f"{claude_home_host}:{claude_home_container}")
    args.envs.append(f"HOME={claude_home_container}")

    image_tag = f"claude_code_cli:{args.version}"
    dockerfile_content = DOCKERFILE_TEMPLATE.format(version=args.version)

    if not build_docker_image(dockerfile_content, image_tag, debug=args.debug):
        sys.exit(1)

    exit_code = run_claude_container(
        image_tag,
        args.claude_args,
        args.volumes,
        args.uid,
        args.envs,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
