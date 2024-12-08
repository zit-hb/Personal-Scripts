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
# Usage:
# ./docker.py [options] [target_script] -- [script_args]
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
#   -P, --privileged                  Start the Docker container in privileged mode.
#   -G, --gpu                         Enable GPU access for the Docker container.
#   -N, --no-cache                    Do not use cache when building the Docker image.
#   -v, --verbose                     Enable verbose logging (INFO level).
#   -vv, --debug                      Enable debug logging (DEBUG level).
#   -T, --test PATH                   File or directory to test scripts.
#   -c, --cache PATH                  Path to a directory to use as a cache (default: ~/.cache/buchwald).
#   -Y, --no-tty                      Disable TTY mode even if stdout is a terminal.
#
# Requirements:
# - Docker must be installed and running on the host system.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional
from dataclasses import dataclass

TEMPLATES = {
    'ubuntu22.04': {
        'docker_run_options': [],
        'dockerfile_template': '''
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
[INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
''',
    },
    'cuda12.4.1-ubuntu22.04': {
        'docker_run_options': [],
        'dockerfile_template': '''
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
[INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
''',
    },
    'cuda11.3.1-ubuntu20.04': {
        'docker_run_options': [],
        'dockerfile_template': '''
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
[INSTALL_COMMANDS]
WORKDIR /app
ENTRYPOINT ["python3"]
''',
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
        description='Docker wrapper script to execute a script inside a Docker container.'
    )
    parser.add_argument(
        'target_script',
        type=str,
        nargs='?',
        help='The target script to execute inside the Docker container.'
    )
    parser.add_argument(
        '-T',
        '--test',
        type=str,
        help='File or directory to test scripts.'
    )
    parser.add_argument(
        '-t',
        '--template',
        type=str,
        choices=TEMPLATES.keys(),
        help=f'Dockerfile template to use. Available templates: {", ".join(TEMPLATES.keys())}'
    )
    parser.add_argument(
        '-i',
        '--input-dockerfile',
        type=str,
        help='Path to an existing Dockerfile to use.'
    )
    parser.add_argument(
        '-o',
        '--output-dockerfile',
        type=str,
        help='Path to save the generated Dockerfile.'
    )
    parser.add_argument(
        '-V',
        '--volume',
        action='append',
        help='Additional volumes to mount into the Docker container. Can be specified multiple times.'
    )
    parser.add_argument(
        '-d',
        '--data',
        type=str,
        help='Mounts a host directory to /data in the container.'
    )
    parser.add_argument(
        '-e',
        '--env',
        action='append',
        help='Environment variables to set in the Docker container. Can be specified multiple times.'
    )
    parser.add_argument(
        '-P',
        '--privileged',
        action='store_true',
        help='Start the Docker container in privileged mode.'
    )
    parser.add_argument(
        '-G',
        '--gpu',
        action='store_true',
        help='Enable GPU access for the Docker container.'
    )
    parser.add_argument(
        '-N',
        '--no-cache',
        action='store_true',
        help='Do not use cache when building the Docker image.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    parser.add_argument(
        '-vv',
        '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )
    parser.add_argument(
        '-c',
        '--cache',
        type=str,
        default='~/.cache/buchwald',
        help='Path to a directory to use as a cache.'
    )
    parser.add_argument(
        '-Y', '--no-tty',
        action='store_true',
        help='Disable TTY mode even if stdout is a terminal.'
    )
    parser.add_argument(
        'script_args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the target script inside the Docker container.'
    )

    args = parser.parse_args()
    if args.script_args and args.script_args[0] == '--':
        args.script_args = args.script_args[1:]

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
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')


def parse_script_header(script_path: str) -> ScriptHeader:
    """
    Parses the script header to extract the template name and install commands.
    """
    template_name = None
    install_commands = []
    try:
        with open(script_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Target script '{script_path}' not found.")
        sys.exit(1)
    header_lines = [line for line in lines if line.startswith('#')]
    requirements_started = False
    for line in header_lines:
        line_content = line.lstrip('#').strip()
        # Check for Template line
        if line_content.startswith('Template:'):
            template_name_candidate = line_content[len('Template:'):].strip()
            logging.debug(f"Found template in script header: '{template_name_candidate}'")
            template_name = template_name_candidate
            continue
        if line_content.startswith('Requirements:'):
            requirements_started = True
            continue
        if requirements_started:
            if line_content == '' or line_content.startswith('-----'):
                break
            # Match any line containing '(install via: ...)'
            match = re.search(r'\(install via:\s*(.*?)\)', line_content)
            if match:
                install_command = match.group(1).strip()
                # Remove 'sudo' if present
                if install_command.startswith('sudo '):
                    install_command = install_command[len('sudo '):]
                logging.debug(f"Found install command: '{install_command}'")
                install_commands.append(install_command)
            else:
                logging.debug(f"No install command found in line: '{line_content}'")
    logging.info(f"Extracted template: {template_name}")
    logging.info(f"Extracted install commands: {install_commands}")
    return ScriptHeader(template_name=template_name, install_commands=install_commands)


def generate_dockerfile(template_name: str, install_commands: List[str], dockerfile_path: str) -> None:
    """
    Generates the Dockerfile based on the selected template and install commands.
    """
    if template_name not in TEMPLATES:
        logging.error(f"Template '{template_name}' is not supported.")
        sys.exit(1)
    template_info = TEMPLATES[template_name]
    dockerfile_content = template_info['dockerfile_template'].lstrip()

    # Prepare the install commands
    if install_commands:
        install_cmds = '\n'.join([f'RUN {cmd}' for cmd in install_commands])
    else:
        install_cmds = ''

    # Replace the placeholder
    dockerfile_content = dockerfile_content.replace('[INSTALL_COMMANDS]', install_cmds)

    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    logging.info(f"Dockerfile generated at '{dockerfile_path}'")


def run_build_command(cmd: List[str], verbose: bool, tty_mode: bool, test_mode: bool) -> int:
    """
    Runs the docker build command.
    If verbose and tty_mode and not test_mode: run without capture (live output).
    Otherwise: run with capture. If verbose, log the output; if not verbose, do not log.
    """
    if verbose and tty_mode and not test_mode:
        # Live output (no capture)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logging.error(f"Failed to build Docker image. Exit code: {result.returncode}")
        return result.returncode
    else:
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            logging.info(result.stdout)
            logging.info(result.stderr)
        if result.returncode != 0:
            logging.error(f"Failed to build Docker image. Exit code: {result.returncode}")
        return result.returncode


def build_docker_image(
    context_dir: str,
    dockerfile_path: str,
    image_tag: str,
    no_cache: bool,
    verbose: bool,
    tty_mode: bool,
    test_mode: bool
) -> int:
    """
    Builds the Docker image using the Dockerfile.
    """
    cmd = ['docker', 'build', '-f', dockerfile_path, '-t', image_tag]
    if no_cache:
        cmd.append('--no-cache')
    cmd.append(context_dir)
    logging.info(f"Building Docker image with tag '{image_tag}'")

    return run_build_command(cmd, verbose, tty_mode, test_mode)


def run_container_command(cmd: List[str], verbose: bool, tty_mode: bool, test_mode: bool) -> int:
    """
    Runs the docker run command.
    If test_mode or not tty_mode: capture and if verbose log output, else no logging.
    Otherwise (no test_mode and tty_mode): run without capture (live output).
    """
    if test_mode or not tty_mode:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            logging.info(result.stdout)
            logging.info(result.stderr)
        return result.returncode
    else:
        # Live output
        result = subprocess.run(cmd)
        return result.returncode


def run_docker_container(
    docker_run_options: List[str],
    image_tag: str,
    target_script_path: str,
    volumes: Optional[List[str]],
    data_path: Optional[str],
    env_vars: Optional[List[str]],
    privileged: bool,
    gpu: bool,
    script_args: List[str],
    cache_path: Optional[str],
    verbose: bool,
    tty_mode: bool,
    test_mode: bool
) -> int:
    """
    Runs the Docker container with the specified image and options.
    """
    cmd = ['docker', 'run', '--rm']
    cmd += docker_run_options

    if privileged:
        cmd += ['--privileged']

    if gpu:
        cmd += ['--gpus', 'all']

    # If tty_mode and not test_mode, we will run with live output (no capture),
    # so we add '-it' to docker run in that scenario
    if tty_mode and not test_mode:
        cmd += ['-it']

    if env_vars:
        for env_var in env_vars:
            cmd += ['-e', env_var]

    script_name = os.path.basename(target_script_path)
    cmd += ['-v', f'{os.path.abspath(target_script_path)}:/app/{script_name}:ro']

    if volumes:
        for vol in volumes:
            cmd += ['-v', vol]

    if data_path:
        cmd += ['-v', f'{os.path.abspath(data_path)}:/data']

    if cache_path:
        expanded_cache_path = os.path.expanduser(cache_path)
        cmd += ['-v', f'{expanded_cache_path}:/root/.cache']

    cmd += [image_tag, f'/app/{script_name}'] + script_args

    logging.info(f"Running Docker container with command: {' '.join(cmd)}")

    return run_container_command(cmd, verbose, tty_mode, test_mode)


def normalize_script_name(script_path: str) -> str:
    """
    Normalizes the script path to be used as the Docker image tag.
    """
    script_name = os.path.splitext(script_path)[0]
    script_name = re.sub(r'\W', '_', script_name).lower()
    return script_name


def process_single_script(script_path: str, args: argparse.Namespace, tty_mode: bool, test_mode: bool) -> bool:
    """
    Processes a single script: parse header, generate Dockerfile, build image, run container.
    Returns True if succeeded, False otherwise.
    """
    # Parse the script header
    header = parse_script_header(script_path)
    if not header.template_name:
        logging.info(f"Skipping script '{script_path}' as it does not specify a template.")
        return False

    # Determine template
    template_name = args.template or header.template_name
    if template_name not in TEMPLATES:
        logging.error(f"Template '{template_name}' is not supported for script '{script_path}'.")
        return False

    # Create a temporary directory for Dockerfile
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, 'Dockerfile')
        generate_dockerfile(template_name, header.install_commands, dockerfile_path)

        if args.output_dockerfile:
            with open(dockerfile_path, 'r') as src, open(args.output_dockerfile, 'w') as dst:
                dst.write(src.read())
            logging.info(f"Dockerfile saved to '{args.output_dockerfile}'")

        image_tag = normalize_script_name(os.path.basename(script_path))

        # Build the Docker image
        build_status = build_docker_image(
            context_dir=tmpdir,
            dockerfile_path=dockerfile_path,
            image_tag=image_tag,
            no_cache=args.no_cache,
            verbose=args.verbose,
            tty_mode=tty_mode,
            test_mode=test_mode
        )
        if build_status != 0:
            return False

        docker_run_options = TEMPLATES[template_name]['docker_run_options']

        # Determine script_args for test mode
        script_args = ['-h'] if test_mode else args.script_args

        # Run the Docker container
        run_status = run_docker_container(
            docker_run_options=docker_run_options,
            image_tag=image_tag,
            target_script_path=script_path,
            volumes=args.volume,
            data_path=args.data,
            env_vars=args.env,
            privileged=args.privileged,
            gpu=args.gpu,
            script_args=script_args,
            cache_path=args.cache,
            verbose=args.verbose,
            tty_mode=tty_mode,
            test_mode=test_mode
        )

        return run_status == 0


def test_scripts(args: argparse.Namespace) -> int:
    """
    Tests scripts specified in args.test (file or directory).
    Outputs a summary of which tests succeeded and which failed.
    Returns the number of failed tests.
    """
    successes = []
    failures = []
    test_path: str = args.test
    script_paths = []

    if os.path.isfile(test_path):
        script_paths = [test_path]
    elif os.path.isdir(test_path):
        for root, _, files in os.walk(test_path):
            for file in files:
                if file.endswith('.py'):
                    script_paths.append(os.path.join(root, file))
    else:
        logging.error(f"The test path '{test_path}' is neither a file nor a directory.")
        sys.exit(1)

    # TTY mode disabled for tests
    tty_mode = False

    for script_path in script_paths:
        logging.info(f"Processing script '{script_path}'")
        header = parse_script_header(script_path)
        if not header.template_name:
            logging.info(f"Skipping script '{script_path}' as it does not specify a template.")
            continue
        success = process_single_script(script_path, args, tty_mode=tty_mode, test_mode=True)
        if success:
            successes.append(script_path)
        else:
            failures.append(script_path)

    total_tests = len(successes) + len(failures)
    print(f"Total scripts tested: {total_tests}")
    print(f"Successful tests: {len(successes)}")
    for script in successes:
        print(f"  {script}")
    print(f"Failed tests: {len(failures)}")
    for script in failures:
        print(f"  {script}")

    return len(failures)


def prepare_dockerfile(args: argparse.Namespace, tmpdir: str) -> DockerfilePreparation:
    """
    Prepares the Dockerfile and returns a DockerfilePreparation object.
    """
    if args.input_dockerfile:
        dockerfile_path = args.input_dockerfile
        context_dir = os.path.dirname(os.path.abspath(dockerfile_path))
        logging.info(f"Using input Dockerfile at '{dockerfile_path}' with context '{context_dir}'")
        if not os.path.isfile(dockerfile_path):
            logging.error(f"Input Dockerfile '{dockerfile_path}' does not exist.")
            sys.exit(1)
        docker_run_options = []
        return DockerfilePreparation(
            dockerfile_path=dockerfile_path,
            docker_run_options=docker_run_options,
            context_dir=context_dir
        )
    else:
        header = parse_script_header(args.target_script)
        template_name = args.template or header.template_name
        if not template_name:
            logging.error("No template specified. Please specify a template using '--template' or in the script header.")
            sys.exit(1)
        if template_name not in TEMPLATES:
            logging.error(f"Template '{template_name}' is not supported.")
            sys.exit(1)
        dockerfile_path = os.path.join(tmpdir, 'Dockerfile')
        generate_dockerfile(template_name, header.install_commands, dockerfile_path)
        if args.output_dockerfile:
            with open(dockerfile_path, 'r') as src, open(args.output_dockerfile, 'w') as dst:
                dst.write(src.read())
            logging.info(f"Dockerfile saved to '{args.output_dockerfile}'")
        context_dir = tmpdir
        docker_run_options = TEMPLATES[template_name]['docker_run_options']
        return DockerfilePreparation(
            dockerfile_path=dockerfile_path,
            docker_run_options=docker_run_options,
            context_dir=context_dir
        )


def main() -> None:
    """
    Main function to orchestrate the Docker wrapper process.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    # Determine TTY mode: enabled if not no-tty and stdout is a TTY
    tty_mode = (not args.no_tty) and sys.stdout.isatty()

    if not args.target_script and not args.test:
        logging.error("No target script specified. Please provide a script to execute.")
        sys.exit(2)
    elif args.test:
        failures = test_scripts(args)
        if failures > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        prep = prepare_dockerfile(args, tmpdir)
        image_tag = normalize_script_name(os.path.basename(args.target_script))
        test_mode = False

        # Build the Docker image
        build_status = build_docker_image(
            context_dir=prep.context_dir,
            dockerfile_path=prep.dockerfile_path,
            image_tag=image_tag,
            no_cache=args.no_cache,
            verbose=args.verbose,
            tty_mode=tty_mode,
            test_mode=test_mode
        )

        if build_status != 0:
            sys.exit(build_status)

        # Run the Docker container
        run_status = run_docker_container(
            docker_run_options=prep.docker_run_options,
            image_tag=image_tag,
            target_script_path=args.target_script,
            volumes=args.volume,
            data_path=args.data,
            env_vars=args.env,
            privileged=args.privileged,
            gpu=args.gpu,
            script_args=args.script_args,
            cache_path=args.cache,
            verbose=args.verbose,
            tty_mode=tty_mode,
            test_mode=test_mode
        )
        sys.exit(run_status)


if __name__ == '__main__':
    main()
