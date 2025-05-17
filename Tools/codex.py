#!/usr/bin/env python3

# -------------------------------------------------------
# Script: codex.py
#
# Description:
# Runs the OpenAI Codex CLI tool inside a Docker container.
#
# Usage:
#   ./codex.py [options] -- [codex arguments]
#
# Arguments:
#   - codex_args: Arguments passed to the codex CLI.
#
# Options:
#   -V, --version VERSION     Codex CLI version to install (default: 0.1.2505161800).
#   -m, --mount VOLUME        Bind-mount a volume (format: host_path:container_path).
#                             Can be specified multiple times.
#   --uid UID                 User ID to run the container as inside Docker
#                             (default: current user).
#   -d, --data PATH           Shortcut for --mount PATH:/data.
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

DOCKERFILE_TEMPLATE: str = """FROM node:20-slim
RUN mkdir /data
WORKDIR /data
RUN npm install -g @openai/codex@{version}
ENTRYPOINT ["codex"]
CMD ["--help"]
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run OpenAI Codex CLI inside Docker.")
    parser.add_argument(
        "-V",
        "--version",
        type=str,
        default="0.1.2505161800",
        help="Codex CLI version to install (default: 0.1.2505161800).",
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
        "--uid",
        type=int,
        default=os.geteuid(),
        help="User ID to run the container as inside Docker (default: current user).",
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
        "codex_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the codex CLI.",
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


def build_docker_image(dockerfile_content: str, image_tag: str) -> bool:
    """
    Builds the Docker image with the given tag.
    """
    try:
        with tempfile.TemporaryDirectory() as build_dir:
            dockerfile_path = os.path.join(build_dir, "Dockerfile")
            with open(dockerfile_path, "w") as dockerfile:
                dockerfile.write(dockerfile_content)
            logging.info("Building Docker image '%s'.", image_tag)
            subprocess.run(
                ["docker", "build", "-t", image_tag, build_dir],
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        return True
    except subprocess.CalledProcessError as error:
        logging.error("Docker build failed: %s", error)
        return False


def run_codex_container(
    image_tag: str,
    codex_args: List[str],
    volumes: List[str],
    user_id: int,
) -> int:
    """
    Runs the Codex CLI inside the Docker container.
    """
    cmd: List[str] = ["docker", "run", "--rm", "-it", "-u", str(user_id)]
    for volume in volumes:
        cmd.extend(["-v", volume])
    cmd.extend([image_tag] + codex_args)
    logging.info("Running Codex CLI: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as error:
        logging.error("Failed to run container: %s", error)
        return 1


def main() -> None:
    """
    Main function to orchestrate the Dockerized Codex CLI workflow.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.data:
        args.volumes.append(f"{args.data}:/data")

    image_tag = f"codex_cli:{args.version}"
    dockerfile_content = DOCKERFILE_TEMPLATE.format(version=args.version)

    if not build_docker_image(dockerfile_content, image_tag):
        sys.exit(1)

    exit_code = run_codex_container(image_tag, args.codex_args, args.volumes, args.uid)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
