#!/usr/bin/env python3

# -------------------------------------------------------
# Script: docker.py
#
# Description:
# This script allows you to execute another script inside a Docker container.
# It can parse the header of the target script to extract a specified template
# and requirements. It generates a Dockerfile based on the selected or default
# template and builds and runs the Docker container.
#
# The container runs as the invoking user and gets a writable home directory inside
# the cache directory, so downloaded models and other caches survive between runs.
#
# Environment variables are handed to Docker through a private env file instead of the
# command line, so secrets do not show up in the process list of the host.
#
# Usage:
#   ./docker.py [options] [target_script] [script_args]
#
# Arguments:
#   - [target_script]: The path to the target script to execute inside the Docker container.
#   - [script_args]: Arguments to pass to the target script inside the Docker container.
#
# Options:
#   -t, --template TEMPLATE_NAME      Dockerfile template to use.
#                                     If not specified, the template from the script header is used.
#   -i, --input-dockerfile PATH       Path to an existing Dockerfile to use.
#   -o, --output-dockerfile PATH      Path to save the generated Dockerfile.
#   -V, --volume VOLUME               Additional volumes to mount into the Docker container. Can be specified multiple times.
#   -d, --data PATH                   Mounts a host directory to /data in the container.
#   -e, --env ENV_VAR                 Environment variables to set in the Docker container. Can be specified multiple times.
#                                     Accepts 'KEY=VALUE' or a bare 'KEY' to pass a variable through from the host.
#   -P, --privileged                  Start the Docker container in privileged mode.
#   -G, --gpu                         Enable GPU access for the Docker container.
#   -N, --no-cache                    Do not use cache when building the Docker image.
#   -v, --verbose                     Enable verbose logging (INFO level).
#   -vv, --debug                      Enable debug logging (DEBUG level).
#   -T, --test PATH                   File or directory to test scripts.
#   -c, --cache PATH                  Path to a directory to use as a cache (default: ~/.cache/buchwald).
#   -Y, --no-tty                      Disable TTY mode even if stdout is a terminal.
#   -p, --port PORT_MAPPING           Forward ports from the Docker container (e.g., 8080:8080). Can be specified multiple times.
#   -u, --user-id UID                 UID of the user inside the container (default: current user's UID)
#   -S, --shm-size SIZE               Set the size of /dev/shm (e.g., 1g for 1 gigabyte).
#
# Requirements:
# - Docker must be installed and running on the host system.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import hashlib
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

# Home directory of the user inside the container. It is backed by a directory in the
# cache so that anything a script caches under '~' is reused on the next run.
CONTAINER_HOME = "/home/buchwald"

# Subdirectory of the cache directory that backs CONTAINER_HOME.
CONTAINER_HOME_CACHE_SUBDIR = "container-home"

TEMPLATES = {
    "ubuntu22.04": {
        "docker_run_options": [],
        "dockerfile_template": """
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \\
    apt-get install -y python3 python3-pip
RUN apt-get update && \\
    [INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
""",
    },
    "ubuntu24.04": {
        "docker_run_options": [],
        "dockerfile_template": """
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \\
    apt-get install -y python3 python3-pip python3-venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update && \\
    [INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
""",
    },
    "cuda12.4.1-ubuntu22.04": {
        "docker_run_options": [],
        "dockerfile_template": """
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \\
    apt-get install -y python3 python3-pip
RUN apt-get update && \\
    [INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
""",
    },
    "cuda11.3.1-ubuntu20.04": {
        "docker_run_options": [],
        "dockerfile_template": """
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \\
    apt-get install -y python3 python3-pip
RUN apt-get update && \\
    [INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
""",
    },
}


@dataclass
class ScriptHeader:
    template_name: Optional[str]
    install_commands: List[str]


@dataclass
class DockerfilePreparation:
    dockerfile_path: str
    docker_run_options: List[str]
    context_dir: str


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the docker_wrapper script.
    """
    parser = argparse.ArgumentParser(
        description="Docker wrapper script to execute a script inside a Docker container."
    )
    parser.add_argument(
        "-T",
        "--test",
        type=str,
        help="File or directory to test scripts.",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        choices=TEMPLATES.keys(),
        help=f"Dockerfile template to use. Available templates: {', '.join(TEMPLATES.keys())}",
    )
    parser.add_argument(
        "-i",
        "--input-dockerfile",
        type=str,
        help="Path to an existing Dockerfile to use.",
    )
    parser.add_argument(
        "-o",
        "--output-dockerfile",
        type=str,
        help="Path to save the generated Dockerfile.",
    )
    parser.add_argument(
        "-V",
        "--volume",
        action="append",
        help="Additional volumes to mount into the Docker container. Can be specified multiple times.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Mounts a host directory to /data in the container.",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        help="Environment variables to set in the Docker container. Can be specified multiple times.",
    )
    parser.add_argument(
        "-P",
        "--privileged",
        action="store_true",
        help="Start the Docker container in privileged mode.",
    )
    parser.add_argument(
        "-G",
        "--gpu",
        action="store_true",
        help="Enable GPU access for the Docker container.",
    )
    parser.add_argument(
        "-N",
        "--no-cache",
        action="store_true",
        help="Do not use cache when building the Docker image.",
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
        "-c",
        "--cache",
        type=str,
        default="~/.cache/buchwald",
        help="Path to a directory to use as a cache.",
    )
    parser.add_argument(
        "-Y",
        "--no-tty",
        action="store_true",
        help="Disable TTY mode even if stdout is a terminal.",
    )
    parser.add_argument(
        "-p",
        "--port",
        action="append",
        help="Forward ports from the Docker container (e.g., 8080:8080). Can be specified multiple times.",
    )
    parser.add_argument(
        "-u",
        "--user-id",
        type=int,
        default=os.getuid(),
        help="UID of the user inside the container (default: current user's UID)",
    )
    parser.add_argument(
        "-S",
        "--shm-size",
        type=str,
        help="Set the size of /dev/shm (e.g., 1g for 1 gigabyte).",
    )
    parser.add_argument(
        "target_script_and_args",
        nargs=argparse.REMAINDER,
        help="The target script plus any arguments to pass inside the Docker container.",
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


def parse_script_header(script_path: str) -> ScriptHeader:
    """
    Parses the script header to extract the template name and install commands.
    """
    template_name = None
    install_commands = []
    try:
        with open(script_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logging.error(f"Could not read target script '{script_path}': {e}")
        sys.exit(1)

    requirements_started = False
    for line in iter_header_lines(lines):
        line_content = line.lstrip("#").strip()
        # Check for Template line
        if line_content.startswith("Template:"):
            template_name = line_content[len("Template:") :].strip()
            logging.debug(f"Found template in script header: '{template_name}'")
            continue
        if line_content.startswith("Requirements:"):
            requirements_started = True
            continue
        if not requirements_started:
            continue
        if line_content == "" or line_content.startswith("-----"):
            break
        # Match any line containing '(install via: ...)'
        match = re.search(r"\(install via:\s*(.*?)\)", line_content)
        if match:
            install_command = match.group(1).strip()
            # Remove 'sudo' if present
            if install_command.startswith("sudo "):
                install_command = install_command[len("sudo ") :]
            logging.debug(f"Found install command: '{install_command}'")
            install_commands.append(install_command)
        else:
            logging.debug(f"No install command found in line: '{line_content}'")
    logging.info(f"Extracted template: {template_name}")
    logging.info(f"Extracted install commands: {install_commands}")
    return ScriptHeader(template_name=template_name, install_commands=install_commands)


def generate_dockerfile(
    template_name: str, install_commands: List[str], dockerfile_path: str
) -> None:
    """
    Generates the Dockerfile based on the selected template and install commands.
    """
    if template_name not in TEMPLATES:
        logging.error(f"Template '{template_name}' is not supported.")
        sys.exit(1)
    template_info = TEMPLATES[template_name]
    dockerfile_content = template_info["dockerfile_template"].lstrip()

    # Prepend pip upgrade command to install_commands
    if template_name == "ubuntu24.04":
        pip_upgrade_cmd = "pip install --upgrade pip"
    else:
        pip_upgrade_cmd = "pip3 install --upgrade pip"

    install_cmds = " && ".join([pip_upgrade_cmd, *install_commands])

    # Replace the placeholder
    dockerfile_content = dockerfile_content.replace("[INSTALL_COMMANDS]", install_cmds)

    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    logging.info(f"Dockerfile generated at '{dockerfile_path}'")


def run_command(cmd: List[str], show_output: bool, error_message: str) -> int:
    """
    Runs a command and returns its exit code. Output is streamed when show_output is set,
    otherwise it is captured and only logged if the command failed, so that failures are
    never reported without an explanation.
    """
    if show_output:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error(f"{error_message} Exit code: {result.returncode}")
        return result.returncode

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logging.error(f"{error_message} Exit code: {result.returncode}")
        if result.stdout:
            logging.error(result.stdout.strip())
        if result.stderr:
            logging.error(result.stderr.strip())
    return result.returncode


def build_docker_image(
    context_dir: str,
    dockerfile_path: str,
    image_tag: str,
    no_cache: bool,
    show_output: bool,
) -> int:
    """
    Builds the Docker image using the Dockerfile.
    """
    cmd = ["docker", "build", "-f", dockerfile_path, "-t", image_tag]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(context_dir)
    logging.info(f"Building Docker image with tag '{image_tag}'")

    return run_command(cmd, show_output, "Failed to build Docker image.")


def run_container_command(cmd: List[str], show_output: bool, test_mode: bool) -> int:
    """
    Runs the docker run command. Outside of test mode the container is attached to the
    terminal, because the script may be interactive.
    """
    if not test_mode:
        return subprocess.run(cmd, check=False).returncode

    return run_command(cmd, show_output, "Failed to run Docker container.")


def prepare_container_home(cache_path: str) -> Optional[str]:
    """
    Creates the host directory that backs the home directory of the container user and
    returns it, or None if it could not be created.
    """
    home_path = os.path.join(cache_path, CONTAINER_HOME_CACHE_SUBDIR)
    try:
        os.makedirs(home_path, exist_ok=True)
    except OSError as e:
        logging.warning(
            f"Could not create the container home directory '{home_path}': {e}. "
            "Caches will not be preserved between runs."
        )
        return None
    return home_path


def write_env_file(env_vars: List[str], directory: str) -> str:
    """
    Writes the environment variables into a private file for 'docker run --env-file' and
    returns its path. Passing them this way keeps secrets out of the host's process list.
    """
    env_file_path = os.path.join(directory, "env")
    fd = os.open(env_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for env_var in env_vars:
            f.write(f"{env_var}\n")
    return env_file_path


def build_docker_run_command(
    docker_run_options: List[str],
    image_tag: str,
    target_script_path: str,
    volumes: Optional[List[str]],
    data_path: Optional[str],
    env_file: Optional[str],
    privileged: bool,
    gpu: bool,
    script_args: List[str],
    container_home: Optional[str],
    tty_mode: bool,
    user_id: Optional[int],
    shm_size: Optional[str],
    ports: Optional[List[str]],
) -> List[str]:
    """
    Assembles the 'docker run' command line.
    """
    cmd = ["docker", "run", "--rm"]
    cmd += docker_run_options

    if privileged:
        cmd += ["--privileged"]

    if gpu:
        cmd += ["--gpus", "all"]

    if tty_mode:
        cmd += ["-t"]

    cmd += ["-i"]

    if env_file:
        cmd += ["--env-file", env_file]

    if ports:
        for port_mapping in ports:
            cmd += ["-p", port_mapping]

    if user_id is not None:
        cmd += ["--user", str(user_id)]

    if shm_size is not None:
        cmd += ["--shm-size", shm_size]

    script_name = os.path.basename(target_script_path)
    cmd += ["-v", f"{os.path.abspath(target_script_path)}:/app/{script_name}:ro"]

    if volumes:
        for vol in volumes:
            cmd += ["-v", vol]

    if data_path:
        cmd += ["-v", f"{os.path.abspath(data_path)}:/data"]

    if container_home:
        # The container user is usually not part of /etc/passwd, so HOME has to be set
        # explicitly. Without this it defaults to '/', which is not writable.
        cmd += ["-v", f"{container_home}:{CONTAINER_HOME}"]
        cmd += ["-e", f"HOME={CONTAINER_HOME}"]
        cmd += ["-e", f"XDG_CACHE_HOME={CONTAINER_HOME}/.cache"]

    cmd += [image_tag, f"/app/{script_name}", *script_args]
    return cmd


def normalize_script_name(script_path: str) -> str:
    """
    Normalizes the script path to be used as part of the Docker image tag.
    """
    script_name = os.path.splitext(script_path)[0]
    script_name = re.sub(r"\W", "_", script_name).lower().strip("_")
    return script_name or "script"


def build_image_tag(script_path: str) -> str:
    """
    Returns the Docker image tag for the given script. The tag contains a digest of the
    resolved script path so that two scripts that merely share a file name do not
    overwrite each other's image, and it is namespaced to avoid clashing with unrelated
    images on the host.
    """
    resolved = os.path.realpath(script_path)
    name = normalize_script_name(os.path.basename(resolved))
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return f"buchwald/{name}:{digest}"


def prepare_dockerfile(
    args: argparse.Namespace, script_path: str, tmpdir: str
) -> Optional[DockerfilePreparation]:
    """
    Prepares the Dockerfile and returns a DockerfilePreparation object, or None if no
    usable Dockerfile could be determined.
    """
    if args.input_dockerfile:
        dockerfile_path = args.input_dockerfile
        context_dir = os.path.dirname(os.path.abspath(dockerfile_path))
        logging.info(
            f"Using input Dockerfile at '{dockerfile_path}' with context '{context_dir}'"
        )
        if not os.path.isfile(dockerfile_path):
            logging.error(f"Input Dockerfile '{dockerfile_path}' does not exist.")
            return None
        return DockerfilePreparation(
            dockerfile_path=dockerfile_path,
            docker_run_options=[],
            context_dir=context_dir,
        )

    header = parse_script_header(script_path)
    template_name = args.template or header.template_name
    if not template_name:
        logging.error(
            "No template specified. Please specify a template using '--template' or in the script header."
        )
        return None
    if template_name not in TEMPLATES:
        logging.error(
            f"Template '{template_name}' is not supported for script '{script_path}'."
        )
        return None

    dockerfile_path = os.path.join(tmpdir, "Dockerfile")
    generate_dockerfile(template_name, header.install_commands, dockerfile_path)
    if args.output_dockerfile:
        with (
            open(dockerfile_path, encoding="utf-8") as src,
            open(args.output_dockerfile, "w", encoding="utf-8") as dst,
        ):
            dst.write(src.read())
        logging.info(f"Dockerfile saved to '{args.output_dockerfile}'")

    return DockerfilePreparation(
        dockerfile_path=dockerfile_path,
        docker_run_options=TEMPLATES[template_name]["docker_run_options"],
        context_dir=tmpdir,
    )


def execute_script(
    args: argparse.Namespace,
    script_path: str,
    script_args: List[str],
    tty_mode: bool,
    test_mode: bool,
) -> int:
    """
    Processes a single script: prepare Dockerfile, build image, run container.
    Returns the exit code of the container, or a non-zero code if it could not be started.
    """
    show_output = args.verbose or args.debug
    expanded_cache_path = os.path.expanduser(args.cache) if args.cache else None

    # The build context and the env file live in separate private temporary directories,
    # so that the env file never becomes part of the Docker build context.
    with (
        tempfile.TemporaryDirectory() as build_dir,
        tempfile.TemporaryDirectory() as runtime_dir,
    ):
        prep = prepare_dockerfile(args, script_path, build_dir)
        if prep is None:
            return 1

        image_tag = build_image_tag(script_path)
        build_status = build_docker_image(
            context_dir=prep.context_dir,
            dockerfile_path=prep.dockerfile_path,
            image_tag=image_tag,
            no_cache=args.no_cache,
            show_output=show_output,
        )
        if build_status != 0:
            return build_status

        env_file = write_env_file(args.env, runtime_dir) if args.env else None
        container_home = (
            prepare_container_home(expanded_cache_path) if expanded_cache_path else None
        )

        cmd = build_docker_run_command(
            docker_run_options=prep.docker_run_options,
            image_tag=image_tag,
            target_script_path=script_path,
            volumes=args.volume,
            data_path=args.data,
            env_file=env_file,
            privileged=args.privileged,
            gpu=args.gpu,
            script_args=script_args,
            container_home=container_home,
            tty_mode=tty_mode,
            user_id=args.user_id,
            shm_size=args.shm_size,
            ports=args.port,
        )
        logging.info(f"Running Docker container with command: {shlex.join(cmd)}")

        return run_container_command(cmd, show_output, test_mode)


def collect_test_scripts(test_path: str) -> List[str]:
    """
    Returns the scripts to test for the given file or directory.
    """
    if os.path.isfile(test_path):
        return [test_path]

    if os.path.isdir(test_path):
        script_paths = []
        for root, _, files in os.walk(test_path):
            script_paths.extend(
                os.path.join(root, file) for file in files if file.endswith(".py")
            )
        return sorted(script_paths)

    logging.error(f"The test path '{test_path}' is neither a file nor a directory.")
    sys.exit(1)


def test_scripts(args: argparse.Namespace) -> int:
    """
    Tests scripts specified in args.test (file or directory).
    Outputs a summary of which tests succeeded and which failed.
    Returns the number of failed tests.
    """
    successes = []
    failures = []

    for script_path in collect_test_scripts(args.test):
        logging.info(f"Processing script '{script_path}'")
        header = parse_script_header(script_path)
        if not header.template_name and not args.template:
            logging.info(
                f"Skipping script '{script_path}' as it does not specify a template."
            )
            continue
        # TTY mode is disabled for tests, and scripts are invoked with '-h'.
        exit_code = execute_script(
            args, script_path, ["-h"], tty_mode=False, test_mode=True
        )
        if exit_code == 0:
            successes.append(script_path)
        else:
            failures.append(script_path)

    print(f"Total scripts tested: {len(successes) + len(failures)}")
    print(f"Successful tests: {len(successes)}")
    for script in successes:
        print(f"  {script}")
    print(f"Failed tests: {len(failures)}")
    for script in failures:
        print(f"  {script}")

    return len(failures)


def main() -> None:
    """
    Main function to orchestrate the Docker wrapper process.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    # Determine TTY mode: enabled if not no-tty and stdout is a TTY
    tty_mode = (not args.no_tty) and sys.stdout.isatty()

    # Check for test mode
    if args.test:
        sys.exit(1 if test_scripts(args) > 0 else 0)

    # If not test mode, we expect a script plus arguments in target_script_and_args
    if not args.target_script_and_args:
        logging.error("No target script specified. Please provide a script to execute.")
        sys.exit(2)

    # Separate the script from its arguments
    target_script = args.target_script_and_args[0]
    script_args = args.target_script_and_args[1:]

    sys.exit(
        execute_script(args, target_script, script_args, tty_mode, test_mode=False)
    )


if __name__ == "__main__":
    main()
