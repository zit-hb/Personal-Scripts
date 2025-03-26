#!/usr/bin/env python3

# -------------------------------------------------------
# Script: qemu-live.py
#
# Description:
# This script launches a temporary live system using QEMU.
#
# Usage:
#   ./qemu-live.py [iso] [options]
#
# Arguments:
#   - [iso]: Predefined keyword, a URL, or a path to a local ISO file.
#
# Options:
#   -m, --memory MEMORY       Memory to allocate to the VM in MB (default: 8096).
#   -c, --cpus CPUS           Number of CPU cores to allocate (default: 6).
#   -q, --qemu PATH           Path to the QEMU executable (default: qemu-system-x86_64).
#   -C, --cache DIR           Directory to cache downloaded ISOs
#                             (default: ~/.cache/buchwald/qemu/isos/).
#   -b, --boot BOOT           Boot order (default: "d" for boot from CD-ROM).
#   -k, --enable-kvm          Explicitly enable KVM acceleration.
#   -K, --disable-kvm         Explicitly disable KVM acceleration.
#                             If neither is specified, the script will auto-detect.
#   -g, --vga VGA             VGA adapter type (default: std).
#   -n, --net NET             Network backend to use (default: user; use "none" for no network).
#   -u, --usb                 Enable USB support.
#   -d, --display DISPLAY     Display type for QEMU (e.g., sdl, gtk).
#   -e, --extra EXTRA_ARGS    Additional QEMU arguments.
#   -L, --list-isos           List all available predefined ISOs.
#   -F, --foreground          Run QEMU in foreground (attached) instead of detached.
#   -v, --verbose             Enable verbose logging (INFO level).
#   -vv, --debug              Enable debug logging (DEBUG level).
#
# Requirements:
#   - Qemu/KVM (install via: apt-get install -y qemu-kvm)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import sys
import subprocess
import hashlib
import os
from pathlib import Path
from typing import List, Optional
import urllib.request

PREDEFINED_ISOS = {
    "haikuos": {
        "url": "https://mirrors.tnonline.net/haiku/haiku-release/r1beta5/haiku-r1beta5-x86_64-anyboot.iso",
        "sha256": "22ae312a38e98083718b6984186e753d15806bd6ea44542144fdcef42c4dcb69",
    },
    "kali": {
        "url": "https://cdimage.kali.org/kali-2024.4/kali-linux-2024.4-live-amd64.iso",
        "sha256": "f07c14ff6f5a89024b2d0d0427c3bc94de86b493a0598a2377286b87478da706",
    },
    "systemrescue": {
        "url": "https://fastly-cdn.system-rescue.org/releases/11.03/systemrescue-11.03-amd64.iso",
        "sha256": "efffa0ba2320a5661593a383a5099a6ac15905e297804a1150dd15a07488f0af",
    },
    "templeos": {
        "url": "https://templeos.org/Downloads/TempleOS.ISO",
        "sha256": "5d0fc944e5d89c155c0fc17c148646715bc1db6fa5750c0b913772cfec19ba26",
    },
    "ubuntu": {
        "url": "https://releases.ubuntu.com/24.04/ubuntu-24.04.1-desktop-amd64.iso",
        "sha256": "c2e6f4dc37ac944e2ed507f87c6188dd4d3179bf4a3f9e110d3c88d1f3294bdc",
    },
}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Launch a temporary live system using QEMU."
    )
    parser.add_argument(
        "iso",
        type=str,
        nargs="?",
        help="Predefined keyword (e.g., 'templeos'), a URL, or a local ISO file path.",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=int,
        default=8096,
        help="Memory to allocate to the VM in MB (default: 8096).",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        default=6,
        help="Number of CPU cores to allocate (default: 6).",
    )
    parser.add_argument(
        "-q",
        "--qemu",
        type=str,
        default="qemu-system-x86_64",
        help="Path to the QEMU executable (default: qemu-system-x86_64).",
    )
    parser.add_argument(
        "-C",
        "--cache",
        type=str,
        default="~/.cache/buchwald/qemu/isos/",
        help="Directory to cache downloaded ISOs (default: ~/.cache/buchwald/qemu/isos/).",
    )
    parser.add_argument(
        "-b",
        "--boot",
        type=str,
        default="d",
        help="Boot order (default: 'd' for boot from CD-ROM).",
    )
    kvm_group = parser.add_mutually_exclusive_group()
    kvm_group.add_argument(
        "-k",
        "--enable-kvm",
        action="store_true",
        help="Explicitly enable KVM acceleration.",
    )
    kvm_group.add_argument(
        "-K",
        "--disable-kvm",
        action="store_true",
        help="Explicitly disable KVM acceleration.",
    )
    parser.add_argument(
        "-g",
        "--vga",
        type=str,
        default="std",
        help="VGA adapter type (default: std).",
    )
    parser.add_argument(
        "-n",
        "--net",
        type=str,
        default="user",
        help="Network backend to use (default: user; use 'none' for no network).",
    )
    parser.add_argument(
        "-u",
        "--usb",
        action="store_true",
        help="Enable USB support.",
    )
    parser.add_argument(
        "-d",
        "--display",
        type=str,
        help="Display type for QEMU (e.g., sdl, gtk).",
    )
    parser.add_argument(
        "-e",
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional QEMU arguments.",
    )
    parser.add_argument(
        "-L",
        "--list-isos",
        action="store_true",
        help="List all available predefined ISOs.",
    )
    parser.add_argument(
        "-F",
        "--foreground",
        action="store_true",
        help="Run QEMU in foreground (attached) instead of detached.",
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

    if not args.iso and not args.list_isos:
        parser.error(
            "You must specify an ISO keyword, URL, or file path unless using --list-isos."
        )

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


def list_predefined_isos() -> None:
    """
    Lists all predefined ISOs with their URL and SHA256.
    """
    for name, info in PREDEFINED_ISOS.items():
        print(f"{name}")
        print(f"    URL: {info.get('url')}")
        sha256 = info.get("sha256")
        if sha256:
            print(f"    SHA256: {sha256}")
        print()
    sys.exit(0)


def download_iso(url: str, dest: Path, expected_sha256: Optional[str] = None) -> None:
    """
    Downloads the ISO from the given URL to the destination path.
    If expected_sha256 is provided, verifies the file's integrity.
    """
    if dest.exists():
        logging.info(f"ISO already cached at '{dest}'.")
        return

    logging.info(f"Downloading ISO from '{url}' to '{dest}'...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        logging.error(f"Failed to download ISO: {e}")
        sys.exit(1)

    if expected_sha256:
        logging.info("Verifying SHA256 checksum...")
        hash_func = hashlib.sha256()
        try:
            with dest.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            file_hash = hash_func.hexdigest()
            if file_hash != expected_sha256:
                logging.error(
                    f"SHA256 mismatch: expected {expected_sha256}, got {file_hash}"
                )
                dest.unlink(missing_ok=True)
                sys.exit(1)
            logging.info("SHA256 checksum verified.")
        except Exception as e:
            logging.error(f"Error verifying checksum: {e}")
            sys.exit(1)


def resolve_iso(iso_arg: str, cache: Path) -> Path:
    """
    Resolves the ISO argument to a local file path.
    For predefined keywords or remote URLs, downloads and caches the ISO.
    For local files, verifies the file exists.
    """
    if iso_arg in PREDEFINED_ISOS:
        iso_info = PREDEFINED_ISOS[iso_arg]
        url = iso_info.get("url")
        sha256 = iso_info.get("sha256")
        iso_filename = url.split("/")[-1]
        iso_path = cache / iso_filename
        download_iso(url, iso_path, sha256)
        return iso_path

    if iso_arg.startswith("http://") or iso_arg.startswith("https://"):
        iso_filename = iso_arg.split("/")[-1]
        iso_path = cache / iso_filename
        download_iso(iso_arg, iso_path, None)
        return iso_path

    # Treat as local file
    iso_path = Path(iso_arg).resolve()
    if not iso_path.exists():
        logging.error(f"ISO file '{iso_path}' does not exist.")
        sys.exit(1)
    return iso_path


def detect_kvm() -> bool:
    """
    Checks if KVM is available and usable.
    Returns True if KVM can be used, False otherwise.
    """
    kvm_path = Path("/dev/kvm")
    if kvm_path.exists() and os.access(str(kvm_path), os.R_OK | os.W_OK):
        logging.info("KVM detected and available.")
        return True
    logging.info("KVM not detected or not accessible.")
    return False


def run_qemu(
    iso_path: Path,
    qemu_executable: str,
    memory: int,
    cpus: int,
    enable_kvm: bool,
    vga: str,
    net: str,
    usb: bool,
    display: Optional[str],
    boot: str,
    extra_args: Optional[List[str]],
    foreground: bool,
) -> None:
    """
    Constructs and runs the QEMU command to start the live system.
    If 'foreground' is True, QEMU runs attached (blocking);
    otherwise it starts detached.
    """
    command = [
        qemu_executable,
        "-m",
        str(memory),
        "-smp",
        str(cpus),
    ]
    if enable_kvm:
        command.append("-enable-kvm")
    command.extend(["-cdrom", str(iso_path)])
    command.extend(["-boot", boot])
    command.append("-snapshot")
    command.extend(["-vga", vga])
    if net.lower() != "none":
        command.extend(["-net", "nic", "-net", net])
    if usb:
        command.append("-usb")
        command.extend(["-device", "usb-tablet"])
    if display:
        command.extend(["-display", display])
    if extra_args:
        command.extend(extra_args)

    if foreground:
        logging.info(f"Starting QEMU in foreground with command: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"QEMU exited with error: {e}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            logging.info("QEMU execution interrupted by user.")
            sys.exit(0)
    else:
        logging.info(f"Starting QEMU detached with command: {' '.join(command)}")
        try:
            proc = subprocess.Popen(
                command,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logging.info(f"QEMU started detached with PID {proc.pid}.")
        except Exception as e:
            logging.error(f"Error starting QEMU detached: {e}")
            sys.exit(1)


def main() -> None:
    """
    Main function to orchestrate the launching of the live system.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.list_isos:
        list_predefined_isos()

    cache = Path(args.cache).expanduser().resolve()
    iso_path = resolve_iso(args.iso, cache)

    if args.disable_kvm:
        kvm_enabled = False
        logging.info("KVM acceleration explicitly disabled.")
    elif args.enable_kvm:
        kvm_enabled = True
        logging.info("KVM acceleration explicitly enabled.")
    else:
        kvm_enabled = detect_kvm()

    run_qemu(
        iso_path,
        args.qemu,
        args.memory,
        args.cpus,
        kvm_enabled,
        args.vga,
        args.net,
        args.usb,
        args.display,
        args.boot,
        args.extra,
        args.foreground,
    )


if __name__ == "__main__":
    main()
