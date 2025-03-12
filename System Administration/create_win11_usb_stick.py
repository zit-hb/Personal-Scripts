#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_win11_usb_stick.py
#
# Description:
# This script automates the creation of a Windows 11 USB installation stick.
# It partitions, formats, and copies the necessary files from a specified
# Windows 11 ISO to a USB device. Everything is handled in temporary
# directories which are cleaned up after creation.
#
# Usage:
#   ./create_win11_usb_stick.py [options] win11_iso_file
#
# Arguments:
#   - win11_iso_file: Path to the Windows 11 ISO file.
#
# Options:
#   -d, --device DEVICE     Specify the target USB device (e.g. /dev/sdc).
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - parted (install via: apt-get install -y parted)
#   - rsync (install via: apt-get install -y rsync)
#   - ntfs-3g (install via: apt-get install -y ntfs-3g)
#   - udisks2 (install via: apt-get install -y udisks2)
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


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Automates the creation of a Windows 11 USB installation stick by "
            "formatting a USB device and copying the required files from an ISO."
        )
    )
    parser.add_argument(
        "win11_iso_file",
        type=str,
        help="Path to the Windows 11 ISO file.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=True,
        help="The target USB device (e.g. /dev/sdc).",
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


def create_windows_usb(iso_path: str, device: str) -> None:
    """
    Partitions, formats, and copies Windows 11 installation files onto a USB stick.
    """
    boot_partition = device + "1"
    install_partition = device + "2"

    try:
        logging.info("Wiping existing filesystem signatures.")
        subprocess.run(["wipefs", "-a", device], check=True)

        logging.info("Creating GPT partition table.")
        subprocess.run(["parted", "-s", device, "mklabel", "gpt"], check=True)

        logging.info("Creating BOOT partition.")
        subprocess.run(
            ["parted", "-s", device, "mkpart", "BOOT", "fat32", "0%", "1GiB"],
            check=True,
        )

        logging.info("Creating INSTALL partition.")
        subprocess.run(
            ["parted", "-s", device, "mkpart", "INSTALL", "ntfs", "1GiB", "10GiB"],
            check=True,
        )

        subprocess.run(["partprobe", device], check=True)

        logging.info("Attempting to unmount partitions if automounted.")
        subprocess.run(["umount", boot_partition], check=False)
        subprocess.run(["umount", install_partition], check=False)

        logging.info("Formatting BOOT partition (FAT32).")
        subprocess.run(["mkfs.vfat", "-n", "BOOT", boot_partition], check=True)

        logging.info("Formatting INSTALL partition (NTFS).")
        subprocess.run(
            ["mkfs.ntfs", "--quick", "-L", "INSTALL", install_partition],
            check=True,
        )

        logging.info("Attempting to unmount partitions if automounted.")
        subprocess.run(["umount", boot_partition], check=False)
        subprocess.run(["umount", install_partition], check=False)

        with (
            tempfile.TemporaryDirectory() as iso_mount_dir,
            tempfile.TemporaryDirectory() as vfat_mount_dir,
            tempfile.TemporaryDirectory() as ntfs_mount_dir,
        ):
            logging.info("Mounting ISO.")
            subprocess.run(["mount", iso_path, iso_mount_dir], check=True)

            logging.info("Mounting BOOT partition.")
            subprocess.run(["mount", boot_partition, vfat_mount_dir], check=True)

            logging.info("Copying boot files to BOOT partition.")
            subprocess.run(
                [
                    "rsync",
                    "-r",
                    "--progress",
                    "--exclude",
                    "sources",
                    "--delete-before",
                    f"{iso_mount_dir}/",
                    f"{vfat_mount_dir}/",
                ],
                check=True,
            )

            logging.info("Copying boot.wim to BOOT partition.")
            os.mkdir(os.path.join(vfat_mount_dir, "sources"))
            subprocess.run(
                [
                    "cp",
                    os.path.join(iso_mount_dir, "sources", "boot.wim"),
                    os.path.join(vfat_mount_dir, "sources"),
                ],
                check=True,
            )

            logging.info("Mounting INSTALL partition.")
            subprocess.run(["mount", install_partition, ntfs_mount_dir], check=True)

            logging.info("Copying installation files to INSTALL partition.")
            subprocess.run(
                [
                    "rsync",
                    "-r",
                    "--progress",
                    "--delete-before",
                    f"{iso_mount_dir}/",
                    f"{ntfs_mount_dir}/",
                ],
                check=True,
            )

            logging.info("Unmounting partitions and ISO.")
            subprocess.run(["umount", ntfs_mount_dir], check=True)
            subprocess.run(["umount", vfat_mount_dir], check=True)
            subprocess.run(["umount", iso_mount_dir], check=True)

        logging.info("Flushing filesystem buffers.")
        subprocess.run(["sync"], check=True)

        logging.info("Powering off device.")
        subprocess.run(["udisksctl", "power-off", "-b", device], check=True)

        logging.info("Windows USB stick created successfully.")

    except subprocess.CalledProcessError as e:
        logging.error("An error occurred while executing a system command.")
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main function to orchestrate the Windows USB creation process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if not os.path.isfile(args.win11_iso_file):
        logging.error(f"ISO file '{args.win11_iso_file}' not found.")
        sys.exit(1)

    create_windows_usb(args.win11_iso_file, args.device)


if __name__ == "__main__":
    main()
