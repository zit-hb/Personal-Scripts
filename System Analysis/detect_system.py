#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_system.py
#
# Description:
# This script detects the underlying operating system and identifies if it is running within a virtualized environment.
# It provides detailed information about the OS, including distribution and version for Linux systems, and identifies
# the type of virtualization (e.g., Docker, VirtualBox, VMware) if present, along with version details where possible.
#
# Usage:
# ./detect_system.py [options]
#
# Options:
# -v, --verbose               Enable verbose logging (INFO level).
# -vv, --debug                Enable debug logging (DEBUG level).
# -a, --all                   Perform all possible checks, including those not limited by OS type.
# -o, --output FILE           Output the detection results to a specified file (JSON format).
# -h, --help                  Show help message and exit.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import glob
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


# Base class for Operating Systems
class OSBase(ABC):
    @abstractmethod
    def matches(self) -> bool:
        """Determine if this class should handle the current OS."""
        pass

    @abstractmethod
    def get_os_info(self) -> Dict[str, Any]:
        """Gather OS-specific information."""
        pass


# Windows OS Class
class OSWindows(OSBase):
    def matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSWindows.matches() called. Current OS: {current_os}")
        return current_os == "Windows"

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Windows OS information.")
        os_info = {
            "OS": "Windows",
            "Version": platform.version(),
            "Release": platform.release(),
            "Architecture": platform.machine(),
            "Processor": platform.processor(),
        }
        return os_info


# macOS Class
class OSMacOS(OSBase):
    def matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSMacOS.matches() called. Current OS: {current_os}")
        return current_os == "Darwin"

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering macOS information.")
        mac_ver = platform.mac_ver()[0]
        os_info = {
            "OS": "macOS",
            "Version": mac_ver if mac_ver else "Unknown",
            "Release": platform.release(),
            "Architecture": platform.machine(),
            "Processor": platform.processor(),
        }
        return os_info


# Linux OS Class
class OSLinux(OSBase):
    def matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSLinux.matches() called. Current OS: {current_os}")
        return current_os == "Linux"

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Linux OS information.")
        os_name, os_version, os_codename = self._parse_os_release()
        os_info = {
            "OS": "Linux",
            "Distribution": os_name,
            "Version": os_version,
            "Codename": os_codename,
            "Architecture": platform.machine(),
            "Processor": platform.processor(),
        }
        return os_info

    def _parse_os_release(self) -> Tuple[str, str, str]:
        """
        Parses the /etc/os-release file to get distribution information.
        """
        os_release_path = "/etc/os-release"
        os_name = "Unknown"
        os_version = "Unknown"
        os_codename = "Unknown"

        try:
            with open(os_release_path, 'r') as f:
                for line in f:
                    if line.startswith("NAME="):
                        os_name = line.strip().split('=')[1].strip('"')
                    elif line.startswith("VERSION_ID="):
                        os_version = line.strip().split('=')[1].strip('"')
                    elif line.startswith("VERSION_CODENAME="):
                        os_codename = line.strip().split('=')[1].strip('"')
        except FileNotFoundError:
            logging.warning(f"'{os_release_path}' not found. Distribution information may be limited.")
        except Exception as e:
            logging.error(f"Error reading '{os_release_path}': {e}")

        return os_name, os_version, os_codename


# Generic OS Class for Unsupported Systems
class OSGeneric(OSBase):
    def matches(self) -> bool:
        # GenericOS matches if no other OS classes do
        logging.debug("OSGeneric.matches() called. Always returns True as fallback.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering generic OS information.")
        os_info = {
            "OS": platform.system(),
            "Version": platform.version(),
            "Release": platform.release(),
            "Architecture": platform.machine(),
            "Processor": platform.processor(),
        }
        return os_info


# Virtualization Base Class
class VMBase(ABC):
    @abstractmethod
    def matches(self) -> bool:
        """Determine if this class should handle the current virtualization environment."""
        pass

    @abstractmethod
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Gather virtualization-specific information."""
        pass


# Docker Detection Class
class VMDocker(VMBase):
    def matches(self) -> bool:
        logging.debug("VMDocker.matches() called.")
        docker_env = False
        if os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'):
            logging.debug("Detected presence of Docker environment files.")
            docker_env = True
        else:
            # Attempt to detect Docker via cgroup
            try:
                with open('/proc/1/cgroup', 'rt') as f:
                    for line in f:
                        if 'docker' in line or 'kubepods' in line:
                            logging.debug("Detected Docker indicators in cgroup.")
                            docker_env = True
                            break
            except Exception as e:
                logging.error(f"Error reading '/proc/1/cgroup': {e}")

        return docker_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Docker environment information.")
        version = self._get_docker_version()
        return {"Environment": "Docker", "Version": version}

    def _get_docker_version(self) -> str:
        """
        Attempts to retrieve the Docker version using the Docker CLI.
        """
        try:
            result = subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                # Example output: "Docker version 20.10.7, build f0df350"
                version_info = result.stdout.strip()
                logging.debug(f"Docker version output: {version_info}")
                return version_info
            else:
                logging.warning(f"Docker CLI returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("Docker CLI not found.")
        except subprocess.TimeoutExpired:
            logging.warning("Docker version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving Docker version: {e}")
        return "Unknown"


# VirtualBox Detection Class
class VMVirtualBox(VMBase):
    def matches(self) -> bool:
        logging.debug("VMVirtualBox.matches() called.")
        product_name_path = '/sys/class/dmi/id/product_name'
        try:
            if os.path.exists(product_name_path):
                with open(product_name_path, 'r') as f:
                    product_name = f.read().strip().lower()
                    logging.debug(f"Product Name: {product_name}")
                    if "virtualbox" in product_name:
                        logging.debug("VirtualBox environment detected via product name.")
                        return True
        except Exception as e:
            logging.error(f"Error reading '{product_name_path}': {e}")
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering VirtualBox environment information.")
        version = self._get_virtualbox_version()
        return {"Environment": "VirtualBox", "Version": version}

    def _get_virtualbox_version(self) -> str:
        """
        Attempts to retrieve the VirtualBox Guest Additions version.
        """
        version = "Unknown"
        possible_paths = [
            '/opt/VBoxGuestAdditions-*/lib/VBoxGuestAdditions.version',
            '/var/log/vboxadd-install.log'
        ]

        # Attempt to read version from known files
        for path in possible_paths:
            expanded_paths = glob.glob(path)
            for p in expanded_paths:
                try:
                    with open(p, 'r') as f:
                        content = f.read().strip()
                        logging.debug(f"VirtualBox version found in '{p}': {content}")
                        return content
                except Exception as e:
                    logging.debug(f"Failed to read VirtualBox version from '{p}': {e}")

        # Fallback: Attempt to use VBoxControl if available
        try:
            result = subprocess.run(['VBoxControl', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                logging.debug(f"VBoxControl version output: {version}")
                return version
            else:
                logging.warning(f"VBoxControl returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("VBoxControl not found.")
        except subprocess.TimeoutExpired:
            logging.warning("VBoxControl version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving VirtualBox version: {e}")

        return version


# VMware Detection Class
class VMMware(VMBase):
    def matches(self) -> bool:
        logging.debug("VMMware.matches() called.")
        product_name_path = '/sys/class/dmi/id/product_name'
        try:
            if os.path.exists(product_name_path):
                with open(product_name_path, 'r') as f:
                    product_name = f.read().strip().lower()
                    logging.debug(f"Product Name: {product_name}")
                    if "vmware" in product_name:
                        logging.debug("VMware environment detected via product name.")
                        return True
        except Exception as e:
            logging.error(f"Error reading '{product_name_path}': {e}")
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering VMware environment information.")
        version = self._get_vmware_version()
        return {"Environment": "VMware", "Version": version}

    def _get_vmware_version(self) -> str:
        """
        Attempts to retrieve the VMware Tools version.
        """
        version = "Unknown"
        # Attempt to use vmware-toolbox-cmd if available
        try:
            result = subprocess.run(['vmware-toolbox-cmd', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"VMware Tools version output: {version_info}")
                return version_info
            else:
                logging.warning(f"vmware-toolbox-cmd returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("vmware-toolbox-cmd not found.")
        except subprocess.TimeoutExpired:
            logging.warning("VMware version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving VMware version: {e}")

        # Fallback: Check for VMware Tools package
        try:
            if os.path.exists('/usr/bin/vmware-toolbox-cmd'):
                # Potentially parse binary for version, but not straightforward
                pass  # Not implemented
        except Exception as e:
            logging.debug(f"Error accessing VMware Tools binary: {e}")

        return version


# KVM Detection Class
class VMKVM(VMBase):
    def matches(self) -> bool:
        logging.debug("VMKVM.matches() called.")
        # Check for KVM in CPU flags
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'kvm' in cpuinfo:
                    logging.debug("KVM environment detected via CPU flags.")
                    return True
        except Exception as e:
            logging.error(f"Error reading '/proc/cpuinfo': {e}")
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering KVM environment information.")
        version = self._get_kvm_version()
        return {"Environment": "KVM", "Version": version}

    def _get_kvm_version(self) -> str:
        """
        Attempts to retrieve the KVM module version.
        """
        version = "Unknown"
        kvm_version_path = '/sys/module/kvm/version'
        try:
            if os.path.exists(kvm_version_path):
                with open(kvm_version_path, 'r') as f:
                    version = f.read().strip()
                    logging.debug(f"KVM version: {version}")
                    return version
        except Exception as e:
            logging.error(f"Error reading '{kvm_version_path}': {e}")

        # Fallback: Attempt to use modinfo
        try:
            result = subprocess.run(['modinfo', 'kvm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("version:"):
                        version = line.split(':', 1)[1].strip()
                        logging.debug(f"modinfo kvm version: {version}")
                        return version
            else:
                logging.warning(f"modinfo kvm returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("modinfo command not found.")
        except subprocess.TimeoutExpired:
            logging.warning("modinfo kvm command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving KVM version: {e}")

        return version


# Bare Metal Detection Class
class VMBareMetal(VMBase):
    def matches(self) -> bool:
        logging.debug("VMBareMetal.matches() called.")
        # Bare Metal is assumed if no other VM matches
        return False  # Always returns False; handled separately

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Bare Metal environment information.")
        return {"Environment": "Bare Metal"}


# System Detector Class
class SystemDetector:
    def __init__(self, perform_all_checks: bool = False) -> None:
        self.perform_all_checks = perform_all_checks
        self.os_detectors = self._initialize_os_detectors()
        self.vm_detectors = self._initialize_vm_detectors()

    def _initialize_os_detectors(self) -> List[OSBase]:
        logging.debug("Initializing OS detectors.")
        return [
            OSWindows(),
            OSMacOS(),
            OSLinux(),
            OSGeneric(),  # GenericOS as fallback
        ]

    def _initialize_vm_detectors(self) -> List[VMBase]:
        logging.debug("Initializing virtualization detectors.")
        return [
            VMDocker(),
            VMVirtualBox(),
            VMMware(),
            VMKVM(),
            # VMBareMetal() is handled separately
        ]

    def detect_system(self) -> Dict[str, Any]:
        logging.debug("Starting system detection.")
        os_info = self._detect_os()

        # Determine if virtualization checks should be performed
        perform_vm_checks = self.perform_all_checks or isinstance(
            self._get_active_os_detector(), OSLinux
        )

        if perform_vm_checks:
            vm_info = self._detect_virtualization()
            os_info["Virtualization"] = vm_info
        else:
            logging.info("Skipping virtualization checks based on OS type.")
            os_info["Virtualization"] = "Not Checked"

        logging.debug("System detection completed.")
        return os_info

    def _detect_os(self) -> Dict[str, Any]:
        for detector in self.os_detectors:
            if detector.matches():
                logging.debug(f"OS detector matched: {detector.__class__.__name__}")
                return detector.get_os_info()
        logging.warning("No OS detector matched. Using generic information.")
        return {"OS": "Unknown"}

    def _get_active_os_detector(self) -> Optional[OSBase]:
        for detector in self.os_detectors:
            if detector.matches():
                return detector
        return None

    def _detect_virtualization(self) -> Dict[str, Any]:
        logging.debug("Detecting virtualization environment.")
        for detector in self.vm_detectors:
            if detector.matches():
                info = detector.get_info()
                if info:
                    logging.debug(f"Virtualization detected: {info}")
                    return info
        # If no virtualization detected, assume Bare Metal
        logging.info("No virtualization environment detected. Assuming Bare Metal.")
        return {"Environment": "Bare Metal"}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect the operating system and virtualization environment.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Global options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    parser.add_argument(
        '-vv', '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )
    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help='Perform all possible checks, including those not limited by OS type.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output the detection results to a specified file (JSON format).'
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

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_output(data: Dict[str, Any], filepath: str) -> bool:
    """
    Saves the detection results to a JSON file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Detection results saved to '{filepath}'.")
        return True
    except Exception as e:
        logging.error(f"Error saving detection results: {e}")
        return False


def display_results(data: Dict[str, Any]) -> None:
    """
    Displays the detection results in a formatted manner.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    """
    Main function to orchestrate the system detection.
    """
    args = parse_arguments()
    setup_logging(
        verbose=args.verbose,
        debug=args.debug
    )

    detector = SystemDetector(perform_all_checks=args.all)
    system_info = detector.detect_system()

    display_results(system_info)

    if args.output:
        if not save_output(system_info, args.output):
            logging.error("Failed to save detection results.")
            sys.exit(1)

    logging.info("System detection completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
