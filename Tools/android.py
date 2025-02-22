#!/usr/bin/env python3

# -------------------------------------------------------
# Script: android.py
#
# Description:
# This script helps build, run, and analyze Android projects in a Docker container.
#
# Warning about xhost:
#   This script may run 'xhost +local:...' to allow the container
#   to display the emulator window on your host. This can have security
#   implications. If xhost was changed, it is reverted after the
#   container exits. Use --no-xhost to skip this behavior.
#
# Usage:
# ./android.py [command] [options]
#
# Commands:
#   build        Build an Android project using Gradle.
#   run          Run an Android APK in an emulator.
#   analyze      Analyze an Android APK.
#
# Options:
#   Global:
#     -J, --java-version         Java version to install (default: 17).
#     -D, --skip-docker-build    Skip building the Docker image (use existing).
#     -A, --android-version      Android platform version (default: 35).
#     -S, --system-image         System image (default: google_apis).
#     -r, --arch                 Architecture (default: x86_64).
#     -B, --build-tools          Build-tools version (default: 35.0.0).
#     -u, --user-id              UID of the container user (default: current user's UID).
#     -v, --verbose              Enable verbose logging (INFO level).
#     -vv, --debug               Enable debug logging (DEBUG level).
#
#   Build:
#     -p, --project-dir          Path to the Android project (default: current dir).
#     -o, --apk-output           Host path where the newly built debug APK is copied (optional).
#     -l, --lint-tests           Run lint checks and tests (default: off).
#     -G, --gradle-task          Custom Gradle task(s) to run (e.g. ':app:assembleDebug').
#
#   Run:
#     -i, --apk-path             Path to existing APK (required).
#     -m, --main-activity        Launcher activity (if omitted, auto-detected from the APK).
#     -N, --avd-name             Name of the AVD to create/run (default: a random name).
#     -L, --no-logcat            Disable streaming logcat output.
#     -X, --no-xhost             Do not configure xhost for GUI access.
#     -x, --xhost-name           Argument to pass to 'xhost +local:NAME' (default: docker).
#
#   Analyze:
#     -i, --apk-path             Path to existing APK (required).
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
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ActivityInfo:
    """
    Holds information about a single activity in the APK.
    """

    name: str = ""
    label: str = ""
    icon: str = ""


@dataclass
class ApkMetadata:
    """
    Holds metadata extracted from 'aapt dump badging' for an APK.
    """

    package_name: str = ""
    version_code: str = ""
    version_name: str = ""
    compile_sdk_version: str = ""
    compile_sdk_version_codename: str = ""
    sdk_version: str = ""
    target_sdk_version: str = ""
    permissions: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    application_label: str = ""
    launchable_activity: str = ""
    activities: List[ActivityInfo] = field(default_factory=list)


def get_java_package(version: str) -> str:
    """
    Returns the Ubuntu package name for the given Java version (8, 11, or 17).
    Raises ValueError if the version is unknown.
    """
    if version == "8":
        return "openjdk-8-jdk"
    elif version == "11":
        return "openjdk-11-jdk"
    elif version == "17":
        return "openjdk-17-jdk"
    else:
        raise ValueError(f"Unsupported Java version: {version}. Use 8, 11, or 17.")


DOCKERFILE_TEMPLATE = r"""\
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ANDROID_VERSION=35
ARG SYSTEM_IMAGE="google_apis"
ARG ARCH="x86_64"
ARG BUILD_TOOLS="35.0.0"
ARG JAVA_PACKAGE="openjdk-17-jdk"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    gradle \
    libgl1-mesa-dev \
    libpulse0 \
    libnss3 \
    libx11-dev \
    libxkbcommon-x11-0 \
    libxkbfile1 \
    libxcb-cursor0 \
    libx11-xcb1 \
    libxcb-xinerama0 \
    libxrender1 \
    libxi6 \
    libxrandr2 \
    libqt5widgets5 \
    libqt5gui5 \
    libqt5core5a \
    adb \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends $JAVA_PACKAGE && rm -rf /var/lib/apt/lists/*

ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_SDK_ROOT/emulator:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin

RUN mkdir -p $ANDROID_SDK_ROOT/cmdline-tools \
    && cd $ANDROID_SDK_ROOT/cmdline-tools \
    && wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip -O cmdline-tools.zip \
    && unzip cmdline-tools.zip -d . \
    && rm cmdline-tools.zip \
    && mkdir -p latest \
    && mv cmdline-tools/* latest/ || true

RUN yes | sdkmanager --licenses

RUN sdkmanager \
    "platform-tools" \
    "emulator" \
    "system-images;android-${ANDROID_VERSION};${SYSTEM_IMAGE};${ARCH}" \
    "platforms;android-${ANDROID_VERSION}" \
    "build-tools;${BUILD_TOOLS}"

ENV PATH=$PATH:$ANDROID_SDK_ROOT/build-tools/${BUILD_TOOLS}

RUN mkdir -p /tmp/gradle && chmod 777 /tmp/gradle
ENV GRADLE_USER_HOME=/tmp/gradle

WORKDIR /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
"""

ENTRYPOINT_SCRIPT = r"""#!/bin/bash
set -e

if [ "$MODE" = "build" ]; then
  echo "Building the project in /app..."

  BUILD_TASK="${GRADLE_TASK:-assembleDebug}"
  if [ "$LINT_TESTS" = "true" ]; then
    BUILD_TASK="${GRADLE_TASK:-build}"
  fi

  # If we have a gradlew, use it; otherwise use system 'gradle'.
  if [ -f "/app/gradlew" ]; then
    chmod +x "/app/gradlew"
    echo "Executing './gradlew $BUILD_TASK' ..."
    (cd /app && ./gradlew $BUILD_TASK)
  else
    echo "Executing 'gradle $BUILD_TASK' ..."
    (cd /app && gradle $BUILD_TASK)
  fi

  echo "Build complete."

  # If APK_OUTPUT is specified, attempt to locate exactly one .apk file in /app
  # and copy it out. If multiple or none are found, provide an appropriate message.
  if [ -n "$APK_OUTPUT" ]; then
    echo "Looking for the generated APK file(s) in /app..."
    FOUND_APKS=$(find /app -type f -name "*.apk" | sort)
    NUM_APKS=$(echo "$FOUND_APKS" | wc -l)
    if [ "$NUM_APKS" -gt 1 ]; then
      echo "ERROR: Multiple APK files found:"
      echo "$FOUND_APKS"
      echo "Please specify a single build variant or manage the APKs manually."
      exit 1
    elif [ "$NUM_APKS" -eq 0 ]; then
      echo "WARNING: No .apk files found in /app."
    else
      SRC_APK="$FOUND_APKS"
      echo "Copying $SRC_APK to $APK_OUTPUT"
      cp "$SRC_APK" "$APK_OUTPUT"
    fi
  fi

  exit 0
fi

if [ "$MODE" = "run" ]; then
  echo "Creating AVD named '$AVD_NAME' with android-$ANDROID_VERSION $SYSTEM_IMAGE $ARCH..."
  echo "no" | avdmanager create avd -n "$AVD_NAME" -k "system-images;android-$ANDROID_VERSION;$SYSTEM_IMAGE;$ARCH" --force

  echo "Starting emulator..."
  emulator -avd "$AVD_NAME" -gpu auto -accel auto -no-snapshot -no-snapshot-save &
  EMULATOR_PID=$!

  echo "Waiting for emulator to be online..."
  adb wait-for-device

  boot_completed=""
  while [ "$boot_completed" != "1" ]; do
    boot_completed=$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r\n')
    sleep 1
  done
  echo "Emulator booted."

  if [ -z "$APK_PATH" ]; then
    echo "ERROR: No APK_PATH specified."
    wait $EMULATOR_PID
    exit 1
  fi

  echo "Installing APK: $APK_PATH"
  adb install -r "$APK_PATH"

  if [ -z "$MAIN_ACTIVITY" ]; then
    echo "ERROR: No MAIN_ACTIVITY specified. Cannot launch activity."
    wait $EMULATOR_PID
    exit 1
  fi

  echo "Launching activity: $MAIN_ACTIVITY"
  adb shell am start -n "$MAIN_ACTIVITY"

  if [ "$LOGCAT_ENABLED" = "true" ]; then
    echo "Starting adb logcat in background..."
    # When we exit (Ctrl+C), kill both emulator & logcat
    trap "kill $EMULATOR_PID $LOGCAT_PID 2>/dev/null || true" EXIT

    adb logcat &
    LOGCAT_PID=$!
    echo "Emulator is running with logcat. Press Ctrl+C to stop..."
  else
    echo "Logcat disabled."
    # Only kill the emulator on exit
    trap "kill $EMULATOR_PID 2>/dev/null || true" EXIT
    echo "Emulator is running. Press Ctrl+C to stop..."
  fi

  # Wait for the emulator process so that the container keeps running
  wait $EMULATOR_PID
  exit 0
fi

echo "ERROR: Unknown MODE value: '$MODE'"
exit 1
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments with subcommands and global options.
    """
    parser = argparse.ArgumentParser(
        description="Build, run, or analyze an Android app in a Dockerized environment."
    )

    user_id_default = os.getuid()

    # Global options
    parser.add_argument(
        "-J",
        "--java-version",
        default="17",
        choices=("8", "11", "17"),
        help="Java version to install (default: 17).",
    )
    parser.add_argument(
        "-D",
        "--skip-docker-build",
        action="store_true",
        help="Skip building the Docker image (use existing).",
    )
    parser.add_argument(
        "-A",
        "--android-version",
        default="35",
        help="Android platform version (default: 35).",
    )
    parser.add_argument(
        "-S",
        "--system-image",
        default="google_apis",
        help="System image (default: google_apis).",
    )
    parser.add_argument(
        "-r",
        "--arch",
        default="x86_64",
        help="Architecture (default: x86_64).",
    )
    parser.add_argument(
        "-B",
        "--build-tools",
        default="35.0.0",
        help="Build-tools version (default: 35.0.0).",
    )
    parser.add_argument(
        "-u",
        "--user-id",
        type=int,
        default=user_id_default,
        help="UID of the container user (default: current user's UID).",
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

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # build subcommand
    parser_build = subparsers.add_parser(
        "build", help="Build the Android project using Gradle."
    )
    parser_build.add_argument(
        "-p",
        "--project-dir",
        default=".",
        help="Path to the Android project (default: current dir).",
    )
    parser_build.add_argument(
        "-o",
        "--apk-output",
        help="Host path where the newly built debug APK is copied (optional).",
    )
    parser_build.add_argument(
        "-l",
        "--lint-tests",
        action="store_true",
        help="Run lint checks and tests (default: off).",
    )
    parser_build.add_argument(
        "-G",
        "--gradle-task",
        default=None,
        help="Custom Gradle task(s) to run (e.g. ':app:assembleDebug'). "
        "By default uses 'assembleDebug', or 'build' when lint-tests is on.",
    )

    # run subcommand
    parser_run = subparsers.add_parser("run", help="Run an Android APK in an emulator.")
    parser_run.add_argument(
        "-i",
        "--apk-path",
        required=True,
        help="Path to existing APK (required).",
    )
    parser_run.add_argument(
        "-m",
        "--main-activity",
        default=None,
        help="Launcher activity (if omitted, auto-detected from the APK).",
    )
    parser_run.add_argument(
        "-N",
        "--avd-name",
        default=None,
        help="Name of the AVD to create/run (default: a random name).",
    )
    parser_run.add_argument(
        "-L",
        "--no-logcat",
        action="store_true",
        help="Disable streaming logcat output.",
    )
    parser_run.add_argument(
        "-X",
        "--no-xhost",
        action="store_true",
        help="Do not configure xhost for GUI access.",
    )
    parser_run.add_argument(
        "-x",
        "--xhost-name",
        default="docker",
        help="Argument to pass to 'xhost +local:NAME' (default: docker).",
    )

    # analyze subcommand
    parser_analyze = subparsers.add_parser("analyze", help="Analyze an Android APK.")
    parser_analyze.add_argument(
        "-i",
        "--apk-path",
        required=True,
        help="Path to existing APK (required).",
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
        level = logging.ERROR

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_docker_image(
    android_version: str,
    system_image: str,
    arch: str,
    build_tools: str,
    java_version: str,
    verbose: bool,
) -> str:
    """
    Builds the Docker image with the specified Android and Java versions,
    unless --skip-docker-build is used. Returns the image tag created.
    """
    java_package = get_java_package(java_version)
    image_tag = (
        f"android-emulator:{android_version}-{system_image}-{arch}-java{java_version}"
    )
    logging.info(f"Building Docker image: {image_tag}")

    build_context = tempfile.mkdtemp(prefix="android_emulator_")

    try:
        # Create a Dockerfile from our template
        dockerfile_content = DOCKERFILE_TEMPLATE.replace("openjdk-17-jdk", java_package)
        dockerfile_path = os.path.join(build_context, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        # Write the entrypoint script
        entrypoint_path = os.path.join(build_context, "entrypoint.sh")
        with open(entrypoint_path, "w") as f:
            f.write(ENTRYPOINT_SCRIPT)

        cmd = [
            "docker",
            "build",
            "-t",
            image_tag,
            "--build-arg",
            f"ANDROID_VERSION={android_version}",
            "--build-arg",
            f"SYSTEM_IMAGE={system_image}",
            "--build-arg",
            f"ARCH={arch}",
            "--build-arg",
            f"BUILD_TOOLS={build_tools}",
            "--build-arg",
            f"JAVA_PACKAGE={java_package}",
            build_context,
        ]

        logging.debug(f"Running: {' '.join(cmd)}")
        if verbose:
            result = subprocess.run(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error("Docker build failed.")
            if not verbose:
                logging.error(result.stderr)
            sys.exit(result.returncode)

        logging.info(f"Docker image built: {image_tag}")
    finally:
        shutil.rmtree(build_context, ignore_errors=True)

    return image_tag


def ensure_xhost(xhost_name: str) -> None:
    """
    Ensures xhost is configured to allow Docker containers to display a GUI
    on the host, by running 'xhost +local:<xhost_name>'.
    """
    try:
        subprocess.run(["xhost", f"+local:{xhost_name}"], check=True)
    except FileNotFoundError:
        logging.warning("xhost not found. Emulator window may not appear.")
    except subprocess.CalledProcessError:
        logging.warning("Failed to configure xhost permissions.")


def revert_xhost(xhost_name: str) -> None:
    """
    Reverts xhost permissions previously granted by removing local
    Docker container access, via 'xhost -local:<xhost_name>'.
    """
    try:
        subprocess.run(["xhost", f"-local:{xhost_name}"], check=True)
    except FileNotFoundError:
        logging.warning("xhost not found. Cannot revert permissions.")
    except subprocess.CalledProcessError:
        logging.warning("Failed to revert xhost permissions.")


def run_container(
    mode: str,
    image_tag: str,
    project_dir: str,
    apk_path: str,
    main_activity: str,
    no_xhost: bool,
    xhost_name: str,
    apk_output: str,
    lint_tests: bool = False,
    avd_name: str = "",
    android_version: str = "",
    system_image: str = "",
    arch: str = "",
    logcat_enabled: bool = True,
    user_id: int = 0,
    gradle_task: str = None,
) -> None:
    """
    Runs a Docker container in one of three modes:
      - 'build': mount the project directory and build.
      - 'run': launch the emulator and install/run the specified APK.
      - 'analyze': run 'aapt dump badging' on the specified APK (handled differently).
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-e",
        f"MODE={mode}",
    ]

    if mode == "run":
        cmd += ["--privileged"]
        cmd += ["--device", "/dev/kvm"]
    else:
        cmd += ["-u", str(user_id)]

    if mode == "build":
        if lint_tests:
            cmd += ["-e", "LINT_TESTS=true"]
        else:
            cmd += ["-e", "LINT_TESTS=false"]

        if gradle_task:
            cmd += ["-e", f"GRADLE_TASK={gradle_task}"]

        proj_abs = str(Path(project_dir).resolve())
        cmd += ["-v", f"{proj_abs}:/app"]

        if apk_output:
            output_file_abs = Path(apk_output).resolve()
            output_dir_abs = output_file_abs.parent
            output_dir_abs.mkdir(parents=True, exist_ok=True)

            cmd += [
                "-v",
                f"{output_dir_abs}:/output",
                "-e",
                f"APK_OUTPUT=/output/{output_file_abs.name}",
            ]

    elif mode == "run":
        apk_abs = str(Path(apk_path).resolve())
        cmd += [
            "-v",
            f"{apk_abs}:/app/app-debug.apk",
            "-e",
            "APK_PATH=/app/app-debug.apk",
        ]
        if main_activity:
            cmd += ["-e", f"MAIN_ACTIVITY={main_activity}"]

        cmd += [
            "-e",
            f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
            "-v",
            "/tmp/.X11-unix:/tmp/.X11-unix",
            "-e",
            f"AVD_NAME={avd_name}",
            "-e",
            f"ANDROID_VERSION={android_version}",
            "-e",
            f"SYSTEM_IMAGE={system_image}",
            "-e",
            f"ARCH={arch}",
            "-e",
            f"LOGCAT_ENABLED={'true' if logcat_enabled else 'false'}",
        ]

    cmd.append(image_tag)

    changed_xhost = False
    if mode == "run" and not no_xhost:
        ensure_xhost(xhost_name)
        changed_xhost = True

    try:
        logging.debug(f"Running container: {' '.join(cmd)}")
        subprocess.run(cmd)
    finally:
        if changed_xhost:
            revert_xhost(xhost_name)


def handle_build(args: argparse.Namespace) -> None:
    """
    Handles the 'build' sub-command by possibly building the Docker image,
    then running a container that mounts the project directory and runs Gradle build.
    If --apk-output is specified, the resulting APK is copied out (if exactly one).
    """
    if not args.skip_docker_build:
        image_tag = build_docker_image(
            args.android_version,
            args.system_image,
            args.arch,
            args.build_tools,
            args.java_version,
            (args.verbose or args.debug),
        )
    else:
        image_tag = (
            f"android-emulator:{args.android_version}-"
            f"{args.system_image}-{args.arch}-java{args.java_version}"
        )
        logging.info(f"Skipping Docker build. Using existing image: {image_tag}")

    run_container(
        mode="build",
        image_tag=image_tag,
        project_dir=args.project_dir,
        apk_path="",
        main_activity="",
        no_xhost=False,  # Not relevant for build
        xhost_name="",
        apk_output=args.apk_output or "",
        lint_tests=args.lint_tests,
        user_id=args.user_id,
        gradle_task=args.gradle_task,
    )


def _parse_main_activity_in_container(
    apk_path: str, image_tag: str, user_id: int = 0
) -> str:
    """
    Runs a temporary Docker container with the given image to execute:
        aapt dump badging <apk>
    Then parses out package name & launchable-activity. Returns am-start style string.
    If not found, returns an empty string.
    """
    apk_abs = str(Path(apk_path).resolve())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        str(user_id),
        "-v",
        f"{apk_abs}:/tmp/app.apk",
        image_tag,
        "aapt",
        "dump",
        "badging",
        "/tmp/app.apk",
    ]
    logging.debug(f"Parsing main activity using command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run 'aapt dump badging' inside container.")
        logging.error(e.stderr)
        return ""

    package_name = ""
    activity_name = ""
    for line in output.splitlines():
        if line.startswith("package:"):
            match = re.search(r"name='([^']+)'", line)
            if match:
                package_name = match.group(1)
        elif line.startswith("launchable-activity:"):
            match = re.search(r"name='([^']+)'", line)
            if match:
                activity_name = match.group(1)

    if not package_name or not activity_name:
        logging.warning("Could not determine package or launchable-activity from APK.")
        return ""

    # Construct typical am start format: com.example/.MainActivity
    if "/" in activity_name:
        return activity_name
    if activity_name.startswith("."):
        return f"{package_name}/{activity_name}"
    if activity_name.startswith(package_name):
        suffix = activity_name[len(package_name) :]
        if not suffix.startswith("."):
            suffix = "." + suffix
        return f"{package_name}/{suffix}"
    return f"{package_name}/{activity_name}"


def handle_run(args: argparse.Namespace) -> None:
    """
    Handles the 'run' sub-command by possibly building the Docker image,
    then running a container that starts the emulator and installs/launches the APK.
    If main activity is not specified, it is auto-detected from the APK.
    """
    if not args.skip_docker_build:
        image_tag = build_docker_image(
            args.android_version,
            args.system_image,
            args.arch,
            args.build_tools,
            args.java_version,
            (args.verbose or args.debug),
        )
    else:
        image_tag = (
            f"android-emulator:{args.android_version}-"
            f"{args.system_image}-{args.arch}-java{args.java_version}"
        )
        logging.info(f"Skipping Docker build. Using existing image: {image_tag}")

    if not args.avd_name:
        random_name = "avd-" + str(uuid.uuid4())[:8]
        logging.info(f"No AVD name specified. Generated random name: {random_name}")
        args.avd_name = random_name

    if not args.main_activity:
        logging.info(
            "No main activity specified; attempting to discover from the APK..."
        )
        detected_activity = _parse_main_activity_in_container(
            args.apk_path, image_tag, args.user_id
        )
        if not detected_activity:
            logging.error(
                "Could not determine the main launcher activity from the APK. "
                "Please specify --main-activity explicitly."
            )
            sys.exit(1)
        args.main_activity = detected_activity
        logging.info(f"Detected main activity: {args.main_activity}")

    run_container(
        mode="run",
        image_tag=image_tag,
        project_dir=".",
        apk_path=args.apk_path,
        main_activity=args.main_activity,
        no_xhost=args.no_xhost,
        xhost_name=args.xhost_name,
        apk_output="",
        avd_name=args.avd_name,
        android_version=args.android_version,
        system_image=args.system_image,
        arch=args.arch,
        logcat_enabled=(not args.no_logcat),
        user_id=args.user_id,
    )


def _parse_badging_info(badging_output: str) -> ApkMetadata:
    """
    Given 'aapt dump badging' output, parse out various APK metadata.
    Returns an ApkMetadata object with discovered fields.
    """
    metadata = ApkMetadata()

    for line in badging_output.splitlines():
        line = line.strip()
        if line.startswith("package:"):
            m_name = re.search(r"name='([^']+)'", line)
            m_vcode = re.search(r"versionCode='([^']+)'", line)
            m_vname = re.search(r"versionName='([^']+)'", line)
            m_cskv = re.search(r"compileSdkVersion='([^']+)'", line)
            m_cskvc = re.search(r"compileSdkVersionCodename='([^']+)'", line)
            if m_name:
                metadata.package_name = m_name.group(1)
            if m_vcode:
                metadata.version_code = m_vcode.group(1)
            if m_vname:
                metadata.version_name = m_vname.group(1)
            if m_cskv:
                metadata.compile_sdk_version = m_cskv.group(1)
            if m_cskvc:
                metadata.compile_sdk_version_codename = m_cskvc.group(1)

        elif line.startswith("sdkVersion:"):
            m_sdk = re.search(r"sdkVersion:'([^']+)'", line)
            if m_sdk:
                metadata.sdk_version = m_sdk.group(1)

        elif line.startswith("targetSdkVersion:"):
            m_tsdk = re.search(r"targetSdkVersion:'([^']+)'", line)
            if m_tsdk:
                metadata.target_sdk_version = m_tsdk.group(1)

        elif line.startswith("uses-permission:"):
            m_perm = re.search(r"name='([^']+)'", line)
            if m_perm:
                metadata.permissions.append(m_perm.group(1))

        elif line.startswith("uses-feature:"):
            m_feat = re.search(r"name='([^']+)'", line)
            if m_feat:
                metadata.features.append(m_feat.group(1))

        elif line.startswith("application-label:"):
            m_label = re.search(r"application-label:'([^']+)'", line)
            if m_label:
                metadata.application_label = m_label.group(1)

        elif line.startswith("launchable-activity:"):
            m_lact = re.search(r"name='([^']+)'", line)
            if m_lact:
                metadata.launchable_activity = m_lact.group(1)

        elif line.startswith("activity:"):
            m_act = re.search(r"name='([^']+)'", line)
            m_label = re.search(r"label='([^']+)'", line)
            m_icon = re.search(r"icon='([^']+)'", line)
            act_name = m_act.group(1) if m_act else ""
            act_label = m_label.group(1) if m_label else ""
            act_icon = m_icon.group(1) if m_icon else ""
            metadata.activities.append(
                ActivityInfo(name=act_name, label=act_label, icon=act_icon)
            )

    return metadata


def handle_analyze(args: argparse.Namespace) -> None:
    """
    Handles the 'analyze' sub-command by possibly building the Docker image,
    then running 'aapt dump badging' inside a container on the specified APK.
    Prints a structured summary of discovered metadata.
    """
    if not args.skip_docker_build:
        image_tag = build_docker_image(
            args.android_version,
            args.system_image,
            args.arch,
            args.build_tools,
            args.java_version,
            (args.verbose or args.debug),
        )
    else:
        image_tag = (
            f"android-emulator:{args.android_version}-"
            f"{args.system_image}-{args.arch}-java{args.java_version}"
        )
        logging.info(f"Skipping Docker build. Using existing image: {image_tag}")

    apk_abs = str(Path(args.apk_path).resolve())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        str(args.user_id),
        "-v",
        f"{apk_abs}:/tmp/app.apk",
        image_tag,
        "aapt",
        "dump",
        "badging",
        "/tmp/app.apk",
    ]
    logging.debug(f"Analyzing APK with command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        badging_output = result.stdout
    except subprocess.CalledProcessError as e:
        logging.error("Failed to run 'aapt dump badging' inside container.")
        logging.error(e.stderr)
        sys.exit(1)

    metadata = _parse_badging_info(badging_output)

    print("===== APK Analysis =====")
    print(f"Package Name:          {metadata.package_name}")
    print(f"Version Code:          {metadata.version_code}")
    print(f"Version Name:          {metadata.version_name}")
    print(f"Compile SDK Version:   {metadata.compile_sdk_version}")
    print(f"Compile SDK Codename:  {metadata.compile_sdk_version_codename}")
    print(f"Min SDK Version:       {metadata.sdk_version}")
    print(f"Target SDK Version:    {metadata.target_sdk_version}")
    print(f"Application Label:     {metadata.application_label}")
    print(f"Launchable Activity:   {metadata.launchable_activity}")
    if metadata.permissions:
        print("Permissions:")
        for p in metadata.permissions:
            print(f"  - {p}")
    if metadata.features:
        print("Features:")
        for f in metadata.features:
            print(f"  - {f}")
    if metadata.activities:
        print("Activities:")
        for a in metadata.activities:
            print(f"  - Name: {a.name}, Label: {a.label}, Icon: {a.icon}")


def main() -> None:
    """
    Main entry point that parses arguments, configures logging,
    and dispatches to the appropriate sub-command.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    if args.subcommand == "build":
        handle_build(args)
    elif args.subcommand == "run":
        handle_run(args)
    elif args.subcommand == "analyze":
        handle_analyze(args)


if __name__ == "__main__":
    main()
