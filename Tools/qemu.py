#!/usr/bin/env python3

# -------------------------------------------------------
# Script: qemu.py
#
# Description:
# This script creates and manages persistent QEMU VMs with proper
# installations, persistent storage, and configurable settings.
# You can create, start, list, and delete VMs.
#
# Usage:
#   ./qemu.py [command] [options]
#
# Commands:
#   create       Create a new VM with a qcow2 disk image. If the VM exists,
#                update its configuration settings (e.g., CPU, memory).
#   start        Start an existing VM.
#   list         List all VMs.
#   delete       Delete an existing VM and its associated data.
#
# Options:
#   Global:
#     -V, --vms               Directory for storing VM configurations and disk images (default: ~/.cache/buchwald/qemu/vms).
#     -v, --verbose           Enable verbose logging (INFO level).
#     -vv, --debug            Enable debug logging (DEBUG level).
#
#   Create:
#     -s, --disk-size SIZE    Disk size for the VM (default: 50G).
#     -m, --memory MEMORY     Memory in MB (default: 8096).
#     -c, --cpus CPUS         Number of CPU cores (default: 6).
#     -q, --qemu PATH         Path to QEMU executable (default: qemu-system-x86_64).
#     -g, --vga VGA           VGA adapter type (default: std).
#     -n, --net NET           Network backend (default: user; use "none" for no network).
#     -u, --usb               Enable USB support.
#     -d, --display DISPLAY   Display type for QEMU (e.g., sdl, gtk).
#     -e, --extra EXTRA_ARGS  Additional QEMU arguments.
#     -I, --disk-type IF      Disk interface type (default, ide, scsi, ahci, virtio).
#
#   Start:
#     -i, --iso ISO           Attach an installation ISO (predefined keyword, URL, or local path).
#     -R, --redownload        Force redownload of cached ISO if it is corrupted.
#     -C, --cache             Directory for caching ISOs (default: ~/.cache/buchwald/qemu/isos)
#     -F, --foreground        Run QEMU in the foreground (attached)
#     -b, --boot BOOT         Boot order override. For example, 'c' for disk first, or 'd' for CD-ROM first.
#     -k, --enable-kvm        Explicitly enable KVM acceleration.
#     -K, --disable-kvm       Explicitly disable KVM acceleration.
#                             If neither is specified, the script will auto-detect.
#
#   Delete:
#     -y, --yes               Confirm deletion without prompting.
#
# Requirements:
#   - Qemu/KVM (install via: apt-get install -y qemu-kvm)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

PREDEFINED_ISOS = {
    "centos10": {
        "url": "https://mirror.dogado.de/centos-stream/10-stream/BaseOS/x86_64/iso/CentOS-Stream-10-latest-x86_64-dvd1.iso",
        "sha256": "23f4b6b6d41188ce1063e1bf5596f3bd5b697ac5c6acbbd4b6d2c18b401d176b",
    },
    "gentoo": {
        "url": "https://distfiles.gentoo.org/releases/amd64/autobuilds/20250216T164837Z/install-amd64-minimal-20250216T164837Z.iso",
        "sha256": "c7da771b38b7d564caadf6a3bc4334a19b2b9a3c95c46d3b7ab15fcac18e6e7f",
    },
    "kali2024.4": {
        "url": "https://cdimage.kali.org/kali-2024.4/kali-linux-2024.4-live-amd64.iso",
        "sha256": "f07c14ff6f5a89024b2d0d0427c3bc94de86b493a0598a2377286b87478da706",
    },
    "kubuntu24.04": {
        "url": "https://cdimage.ubuntu.com/kubuntu/releases/24.10/release/kubuntu-24.10-desktop-amd64.iso",
        "sha256": "3b3c6f4b14c6609bfc296f671eccf11fed4f17702e0e675e810fa79db8fc097c",
    },
    "templeos": {
        "url": "https://templeos.org/Downloads/TempleOS.ISO",
        "sha256": "5d0fc944e5d89c155c0fc17c148646715bc1db6fa5750c0b913772cfec19ba26",
    },
    "ubuntu24.04": {
        "url": "https://releases.ubuntu.com/24.04/ubuntu-24.04.1-desktop-amd64.iso",
        "sha256": "c2e6f4dc37ac944e2ed507f87c6188dd4d3179bf4a3f9e110d3c88d1f3294bdc",
    },
    "xubuntu24.04": {
        "url": "http://ftp.uni-kl.de/pub/linux/ubuntu-dvd/xubuntu/releases/24.04/release/xubuntu-24.04.1-desktop-amd64.iso",
        "sha256": "c333806173558ccc2a95f44c5c7b57437ee3d409b50a3a5a1367bcf7eaf3ef90",
    },
    "zealos": {
        "url": "https://github.com/Zeal-Operating-System/ZealOS/releases/download/latest/ZealOS-PublicDomain-BIOS-2025-02-21-08_16_57.iso",
        "sha256": "ff4cda8db3eeacce36ad774887e6e78e7691f9f0dfc6697b6860309ef6045650",
    },
}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns them.
    """
    parser = argparse.ArgumentParser(
        description="Manage persistent QEMU VMs with proper installations and storage."
    )
    parser.add_argument(
        "-V",
        "--vms",
        type=Path,
        default=Path("~/.cache/buchwald/qemu/vms"),
        help="Directory for storing VM configurations and disk images (default: ~/.cache/buchwald/qemu/vms)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)",
    )
    parser.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level)",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-commands"
    )

    # "create" command
    parser_create = subparsers.add_parser(
        "create", help="Create a new VM (or update an existing VM's configuration)"
    )
    parser_create.add_argument("name", help="Name of the VM")
    parser_create.add_argument(
        "-s",
        "--disk-size",
        type=str,
        default="50G",
        help="Disk size for the VM (default: 50G)",
    )
    parser_create.add_argument(
        "-m",
        "--memory",
        type=int,
        default=8096,
        help="Memory in MB (default: 8096)",
    )
    parser_create.add_argument(
        "-c",
        "--cpus",
        type=int,
        default=6,
        help="Number of CPU cores (default: 6)",
    )
    parser_create.add_argument(
        "-q",
        "--qemu",
        type=str,
        default="qemu-system-x86_64",
        help="Path to QEMU executable (default: qemu-system-x86_64)",
    )
    parser_create.add_argument(
        "-g",
        "--vga",
        type=str,
        default="std",
        help="VGA adapter type (default: std)",
    )
    parser_create.add_argument(
        "-n",
        "--net",
        type=str,
        default="user",
        help="Network backend (default: user; use 'none' for no network). ",
    )
    parser_create.add_argument(
        "-u",
        "--usb",
        action="store_true",
        help="Enable USB support",
    )
    parser_create.add_argument(
        "-d",
        "--display",
        type=str,
        help="Display type for QEMU (e.g., sdl, gtk)",
    )
    parser_create.add_argument(
        "-e",
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional QEMU arguments.",
    )
    parser_create.add_argument(
        "-I",
        "--disk-type",
        type=str,
        default="default",
        help="Disk interface type (default, ide, scsi, ahci, virtio) (default: default).",
    )

    # "start" command
    parser_start = subparsers.add_parser("start", help="Start a VM")
    parser_start.add_argument("name", help="Name of the VM")
    parser_start.add_argument(
        "-i",
        "--iso",
        help="Attach an installation ISO (predefined keyword, URL, or local path)",
    )
    parser_start.add_argument(
        "-C",
        "--cache",
        type=Path,
        default=Path("~/.cache/buchwald/qemu/isos"),
        help="Directory for caching ISOs (default: ~/.cache/buchwald/qemu/isos)",
    )
    parser_start.add_argument(
        "-R",
        "--redownload",
        action="store_true",
        help="Force redownload of cached ISO if it is corrupted.",
    )
    parser_start.add_argument(
        "-F",
        "--foreground",
        action="store_true",
        help="Run QEMU in the foreground (attached)",
    )
    parser_start.add_argument(
        "-b",
        "--boot",
        type=str,
        help="Boot order override. For example, 'c' for disk first, or 'd' for CD-ROM first. "
        "If not provided, the boot order is automatically determined: boot from CD-ROM if an ISO is attached, otherwise boot from disk.",
    )
    parser_start.add_argument(
        "-k",
        "--enable-kvm",
        action="store_true",
        help="Explicitly enable KVM acceleration.",
    )
    parser_start.add_argument(
        "-K",
        "--disable-kvm",
        action="store_true",
        help="Explicitly disable KVM acceleration.",
    )

    # "list" command
    subparsers.add_parser("list", help="List all VMs")

    # "delete" command
    parser_delete = subparsers.add_parser("delete", help="Delete a VM")
    parser_delete.add_argument("name", help="Name of the VM to delete")
    parser_delete.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm deletion without prompting.",
    )

    args = parser.parse_args()
    args.vms = args.vms.expanduser().resolve()
    if hasattr(args, "cache") and args.cache is not None:
        args.cache = args.cache.expanduser().resolve()
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


def download_iso(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
) -> None:
    """
    Downloads the ISO from the given URL to the destination path.
    If expected_sha256 is provided, verifies the file's integrity.
    If a cached ISO exists, its checksum is verified even if already cached.
    If the checksum fails and redownload is True, the file is redownloaded.
    Otherwise, a critical error is raised.
    """
    if dest.exists():
        if expected_sha256:
            logging.info(f"Verifying cached ISO at '{dest}'...")
            hash_func = hashlib.sha256()
            try:
                with dest.open("rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_func.update(chunk)
                file_hash = hash_func.hexdigest()
                if file_hash != expected_sha256:
                    if redownload:
                        logging.warning(
                            f"Cached ISO at '{dest}' is corrupted (SHA256 mismatch). Redownloading..."
                        )
                        dest.unlink(missing_ok=True)
                    else:
                        logging.error(
                            f"SHA256 mismatch for cached ISO: expected {expected_sha256}, got {file_hash}"
                        )
                        sys.exit(1)
                else:
                    logging.info("SHA256 checksum verified for cached ISO.")
                    return
            except Exception as e:
                logging.error(f"Error verifying cached ISO: {e}")
                sys.exit(1)
        else:
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
                    f"SHA256 mismatch after download: expected {expected_sha256}, got {file_hash}"
                )
                dest.unlink(missing_ok=True)
                sys.exit(1)
            logging.info("SHA256 checksum verified.")
        except Exception as e:
            logging.error(f"Error verifying checksum: {e}")
            sys.exit(1)


def resolve_iso(iso_arg: str, cache: Path, redownload: bool = False) -> Path:
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
        download_iso(url, iso_path, sha256, redownload)
        return iso_path

    if iso_arg.startswith("http://") or iso_arg.startswith("https://"):
        iso_filename = iso_arg.split("/")[-1]
        iso_path = cache / iso_filename
        download_iso(iso_arg, iso_path, None, redownload)
        return iso_path

    iso_path = Path(iso_arg).resolve()
    if not iso_path.exists():
        logging.error(f"ISO file '{iso_path}' does not exist.")
        sys.exit(1)
    return iso_path


def get_vm_path(name: str, vms_dir: Path) -> Path:
    """
    Return the directory path for the VM with the given name.
    """
    return vms_dir / name


def load_vm_config(name: str, vms_dir: Path) -> dict:
    """
    Load the configuration for the given VM.
    """
    config_file = get_vm_path(name, vms_dir) / "config.json"
    if not config_file.exists():
        logging.error(f"VM '{name}' configuration not found.")
        sys.exit(1)
    with config_file.open("r") as f:
        config = json.load(f)
    return config


def save_vm_config(name: str, config: dict, vms_dir: Path) -> None:
    """
    Save the VM configuration to its config.json file.
    """
    vm_path = get_vm_path(name, vms_dir)
    vm_path.mkdir(parents=True, exist_ok=True)
    config_file = vm_path / "config.json"
    with config_file.open("w") as f:
        json.dump(config, f, indent=4)


def create_vm(args) -> None:
    """
    Creates a new VM or updates an existing one:
      - Creates/updates a VM directory under <vms_dir>/<name>
      - Creates a qcow2 disk image (using qemu-img) if not already present
      - Saves/updates the VM configuration (excluding boot order and KVM settings,
        which are determined at startup)
    """
    vm_path = get_vm_path(args.name, args.vms)
    new_vm = not vm_path.exists()
    disk_image = vm_path / f"{args.name}.qcow2"

    if new_vm:
        vm_path.mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Creating disk image at '{disk_image}' with size {args.disk_size}..."
        )
        try:
            subprocess.run(
                ["qemu-img", "create", "-f", "qcow2", str(disk_image), args.disk_size],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create disk image: {e}")
            sys.exit(1)
    else:
        if not disk_image.exists():
            logging.info(
                f"Disk image '{disk_image}' not found, creating new disk image with size {args.disk_size}..."
            )
            try:
                subprocess.run(
                    [
                        "qemu-img",
                        "create",
                        "-f",
                        "qcow2",
                        str(disk_image),
                        args.disk_size,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to create disk image: {e}")
                sys.exit(1)
        logging.info(f"VM '{args.name}' already exists, updating configuration.")

    # Preserve original creation time if updating an existing VM.
    created_at = time.time()
    if not new_vm:
        try:
            existing_config = load_vm_config(args.name, args.vms)
            created_at = existing_config.get("created_at", created_at)
        except Exception:
            pass

    config = {
        "name": args.name,
        "disk_image": str(disk_image),
        "disk_size": args.disk_size,
        "memory": args.memory,
        "cpus": args.cpus,
        "qemu": args.qemu,
        "vga": args.vga,
        "net": args.net,
        "usb": args.usb,
        "display": args.display,
        "extra": args.extra if args.extra else [],
        "disk_interface": args.disk_type,
        "created_at": created_at,
    }
    save_vm_config(args.name, config, args.vms)
    logging.info(f"VM '{args.name}' created/updated successfully.")


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


def start_vm(args) -> None:
    """
    Starts an existing VM:
      - Loads the saved configuration.
      - Constructs and executes the QEMU command with persistent disk.
      - Optionally attaches the installation ISO if --iso is specified.
      - Determines boot order automatically unless overridden:
            * If an ISO is attached, defaults to boot from CD-ROM ('d').
            * Otherwise, defaults to boot from disk ('c').
      - Allows overriding boot order with the --boot option.
      - Configures KVM acceleration based on command-line flags:
            * Explicitly enabled/disabled via --enable-kvm/--disable-kvm.
            * Otherwise, auto-detected.
      - Allows specifying disk interface type (default, ide, scsi, ahci, virtio).
    """
    config = load_vm_config(args.name, args.vms)
    qemu_executable = config.get("qemu", "qemu-system-x86_64")
    command = [
        qemu_executable,
        "-m",
        str(config.get("memory", 8096)),
        "-smp",
        str(config.get("cpus", 6)),
    ]

    # KVM detection/flags
    if args.disable_kvm:
        kvm_enabled = False
        logging.info("KVM acceleration explicitly disabled.")
    elif args.enable_kvm:
        kvm_enabled = True
        logging.info("KVM acceleration explicitly enabled.")
    else:
        kvm_enabled = detect_kvm()

    if kvm_enabled:
        command.append("-enable-kvm")

    # ISO handling
    if args.iso:
        iso_path = resolve_iso(args.iso, args.cache, redownload=args.redownload)
        # Use if=none so QEMU doesn't auto-connect this drive.
        # Then explicitly attach it as an IDE CD device.
        command.extend(
            [
                "-drive",
                f"if=none,file={iso_path},id=cdrom0,media=cdrom",
                "-device",
                "ide-cd,drive=cdrom0",
            ]
        )
        boot_order = args.boot if args.boot else "d"
    else:
        boot_order = args.boot if args.boot else "c"

    command.extend(["-boot", boot_order])

    # Disk interface handling
    disk_image = config.get("disk_image")
    disk_interface = config.get("disk_interface", "default")

    if disk_image:
        if disk_interface == "ahci":
            command.extend(
                [
                    "-device",
                    "ich9-ahci,id=ahci",
                    "-drive",
                    f"if=none,file={disk_image},format=qcow2,id=drive0",
                    "-device",
                    "ide-hd,drive=drive0,bus=ahci.0",
                ]
            )
        elif disk_interface == "virtio":
            command.extend(
                [
                    "-drive",
                    f"file={disk_image},format=qcow2,if=virtio",
                ]
            )
        elif disk_interface == "scsi":
            command.extend(
                [
                    "-device",
                    "virtio-scsi-pci,id=scsi0",
                    "-drive",
                    f"if=none,file={disk_image},format=qcow2,id=drive0",
                    "-device",
                    "scsi-hd,drive=drive0",
                ]
            )
        elif disk_interface == "ide":
            command.extend(
                [
                    "-drive",
                    f"file={disk_image},format=qcow2,if=ide",
                ]
            )
        else:
            # default or unknown
            command.extend(["-drive", f"file={disk_image},format=qcow2"])

    # VGA
    command.extend(["-vga", config.get("vga", "std")])

    # Network
    net_config = config.get("net", "user")
    if net_config.lower() == "none":
        logging.info("Networking disabled for this VM.")
    elif net_config.lower() == "user":
        command.extend(["-netdev", "user,id=net0", "-device", "e1000,netdev=net0"])
    else:
        command.extend(["-net", "nic", "-net", net_config])

    # USB
    if config.get("usb", False):
        command.append("-usb")
        command.extend(["-device", "usb-tablet"])

    # Display
    if config.get("display"):
        command.extend(["-display", config["display"]])

    # Extra arguments
    if config.get("extra"):
        command.extend(config["extra"])

    logging.info(f"QEMU command: {' '.join(command)}")
    if args.foreground:
        logging.info("Starting VM in foreground...")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"QEMU exited with error: {e}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            logging.info("QEMU execution interrupted by user.")
            sys.exit(0)
    else:
        logging.info("Starting VM detached...")
        try:
            proc = subprocess.Popen(
                command,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logging.info(f"VM '{args.name}' started detached with PID {proc.pid}.")
        except Exception as e:
            logging.error(f"Error starting VM: {e}")
            sys.exit(1)


def list_vms(args) -> None:
    """
    Lists all managed VMs and displays their basic configuration details.
    """
    if not args.vms.exists():
        print("No VMs found.")
        return
    for vm_dir in sorted(args.vms.iterdir()):
        if vm_dir.is_dir():
            config_file = vm_dir / "config.json"
            if config_file.exists():
                with config_file.open("r") as f:
                    config = json.load(f)
                print(f"Name: {config.get('name')}")
                print(f"  Disk Image: {config.get('disk_image')}")
                print(f"  Memory: {config.get('memory')} MB")
                print(f"  CPUs: {config.get('cpus')}")
                print(f"  Networking: {config.get('net')}")
                disk_if = config.get("disk_interface", "default")
                print(f"  Disk Interface: {disk_if}")
                created_at = config.get("created_at")
                if created_at:
                    created_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(created_at)
                    )
                    print(f"  Created at: {created_time}")
                print()
            else:
                print(f"VM '{vm_dir.name}' has no configuration file.")


def delete_vm(args) -> None:
    """
    Deletes a VM and all its associated data.
    """
    vm_path = get_vm_path(args.name, args.vms)
    if not vm_path.exists():
        logging.error(f"VM '{args.name}' does not exist.")
        sys.exit(1)
    if not args.yes:
        confirm = input(
            f"Are you sure you want to delete VM '{args.name}'? This will remove all its data. [y/N]: "
        )
        if confirm.lower() != "y":
            print("Deletion cancelled.")
            sys.exit(0)
    try:
        shutil.rmtree(vm_path)
        logging.info(f"VM '{args.name}' deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting VM '{args.name}': {e}")
        sys.exit(1)


def main() -> None:
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.command == "create":
        create_vm(args)
    elif args.command == "start":
        start_vm(args)
    elif args.command == "list":
        list_vms(args)
    elif args.command == "delete":
        delete_vm(args)
    else:
        print("Invalid command specified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
