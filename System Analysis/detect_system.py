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
# -p, --paranoid              Enable paranoid mode: perform untrusting OS checks.
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

try:
    import winreg
except ImportError:
    winreg = None  # Handle non-Windows environments gracefully


# Base class for Operating Systems
class OSBase(ABC):
    def __init__(self, paranoid: bool = False) -> None:
        self.paranoid = paranoid

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
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSWindows._trusting_matches() called. Current OS: {current_os}")
        return current_os == "Windows"

    def _paranoid_matches(self) -> bool:
        """
        Perform paranoid checks to verify the OS is genuine Windows.
        This includes verifying critical system files/directories and checking registry keys.
        """
        logging.debug("OSWindows._paranoid_matches() called. Performing paranoid OS verification.")

        # Verify essential Windows directories and files
        essential_paths = [
            r"C:\Windows\System32",
            r"C:\Windows\win.ini",
            r"C:\Windows\System32\cmd.exe",
            r"C:\Windows\System32\kernel32.dll",
            r"C:\Windows\System32\drivers\etc\hosts"
        ]

        for path in essential_paths:
            if not os.path.exists(path):
                logging.debug(f"Paranoid match failed: Essential path '{path}' does not exist.")
                return False
            else:
                logging.debug(f"Paranoid check passed: '{path}' exists.")

        # Verify critical registry keys
        registry_checks = {
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion": [
                "ProductName",
                "ReleaseId",
                "CurrentBuild",
                "CurrentBuildNumber",
                "EditionID"
            ]
        }

        try:
            for reg_path, values in registry_checks.items():
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    for value_name in values:
                        try:
                            value, regtype = winreg.QueryValueEx(key, value_name)
                            if not value:
                                logging.debug(f"Paranoid match failed: Registry value '{value_name}' is empty.")
                                return False
                            logging.debug(f"Paranoid check passed: Registry '{value_name}' = '{value}'.")
                        except FileNotFoundError:
                            logging.debug(f"Paranoid match failed: Registry value '{value_name}' not found.")
                            return False
        except Exception as e:
            logging.error(f"Error accessing registry for paranoid OS verification: {e}")
            return False

        # Verify critical system services are running
        required_services = [
            "Service Control Manager",  # services.exe
            "wuauserv",                 # Windows Update
            "MpsSvc"                    # Windows Firewall
        ]

        for service in required_services:
            if not self._is_service_running(service):
                logging.debug(f"Paranoid match failed: Required service '{service}' is not running.")
                return False
            else:
                logging.debug(f"Paranoid check passed: Service '{service}' is running.")

        logging.debug("Paranoid match succeeded: All paranoid checks passed.")
        return True

    def _is_service_running(self, service_name: str) -> bool:
        """
        Checks if a given Windows service is running.
        """
        try:
            # Use 'sc query' to check service status
            result = subprocess.run(['sc', 'query', service_name],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode != 0:
                logging.debug(f"Service '{service_name}' query failed: {result.stderr.strip()}")
                return False

            # Parse the output to check if the service is running
            for line in result.stdout.splitlines():
                if "STATE" in line:
                    if "RUNNING" in line:
                        logging.debug(f"Service '{service_name}' is running.")
                        return True
                    else:
                        logging.debug(f"Service '{service_name}' is not running. State: {line.strip()}")
                        return False
            logging.debug(f"Service '{service_name}' state not found in query output.")
            return False
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout expired while checking service '{service_name}'.")
            return False
        except Exception as e:
            logging.error(f"Error checking service '{service_name}': {e}")
            return False

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Windows OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            os_info = {
                "OS": "Windows",
                "Version": platform.version(),
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        """
        Gather OS information without relying on platform APIs.
        Reads directly from system files and registry.
        """
        logging.debug("OSWindows._paranoid_get_os_info() called. Gathering OS info in paranoid mode.")

        os_info = {
            "OS": "Windows",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }

        # Retrieve OS Version and Release from registry
        try:
            reg_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                os_info["ProductName"], _ = winreg.QueryValueEx(key, "ProductName")
                os_info["Release"] = winreg.QueryValueEx(key, "ReleaseId")[0]
                os_info["CurrentBuild"] = winreg.QueryValueEx(key, "CurrentBuild")[0]
                os_info["EditionID"], _ = winreg.QueryValueEx(key, "EditionID")
                logging.debug(f"Retrieved registry info: ProductName='{os_info['ProductName']}', ReleaseId='{os_info['Release']}', CurrentBuild='{os_info['CurrentBuild']}', EditionID='{os_info['EditionID']}'")
        except Exception as e:
            logging.error(f"Error retrieving OS version from registry: {e}")

        # Retrieve Architecture from system directories
        try:
            system_dir = r"C:\Windows\System32"
            if os.path.exists(system_dir):
                arch = platform.machine()
                os_info["Architecture"] = arch
                logging.debug(f"Retrieved architecture: {arch}")
            else:
                logging.debug(f"System directory '{system_dir}' does not exist.")
        except Exception as e:
            logging.error(f"Error retrieving architecture: {e}")

        # Retrieve Processor information
        try:
            # Using environment variables as a fallback
            processor = os.environ.get('PROCESSOR_IDENTIFIER', 'Unknown')
            os_info["Processor"] = processor
            logging.debug(f"Retrieved processor info: {processor}")
        except Exception as e:
            logging.error(f"Error retrieving processor information: {e}")

        return os_info


# macOS Class
class OSMacOS(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSMacOS._trusting_matches() called. Current OS: {current_os}")
        return current_os == "Darwin"

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, check for specific macOS files
        specific_files = [
            '/System/Library/CoreServices/SystemVersion.plist',
            '/Applications',
            '/System/Library/Frameworks'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"Paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("Paranoid match succeeded: All specific macOS files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering macOS OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            mac_ver = platform.mac_ver()[0]
            os_info = {
                "OS": "macOS",
                "Version": mac_ver if mac_ver else "Unknown",
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather OS info without relying on platform
        os_info = {
            "OS": "macOS",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Attempt to read SystemVersion.plist
        try:
            plist_path = '/System/Library/CoreServices/SystemVersion.plist'
            if os.path.exists(plist_path):
                import plistlib
                with open(plist_path, 'rb') as f:
                    plist = plistlib.load(f)
                    os_info["Version"] = plist.get('ProductVersion', 'Unknown')
                    os_info["Release"] = plist.get('ProductBuildVersion', 'Unknown')
        except Exception as e:
            logging.error(f"Error retrieving macOS version in paranoid mode: {e}")
        return os_info


# Linux OS Class
class OSLinux(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSLinux._trusting_matches() called. Current OS: {current_os}")
        return current_os == "Linux"

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, check for specific Linux files
        specific_files = [
            '/etc/os-release',
            '/bin/bash',
            '/usr/bin/python3'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"Paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("Paranoid match succeeded: All specific Linux files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Linux OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
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

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather OS info without relying on platform
        os_info = {
            "OS": "Linux",
            "Distribution": "Unknown",
            "Version": "Unknown",
            "Codename": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Parse /etc/os-release manually
        os_release_path = "/etc/os-release"
        try:
            with open(os_release_path, 'r') as f:
                for line in f:
                    if line.startswith("NAME="):
                        os_info["Distribution"] = line.strip().split('=')[1].strip('"')
                    elif line.startswith("VERSION_ID="):
                        os_info["Version"] = line.strip().split('=')[1].strip('"')
                    elif line.startswith("VERSION_CODENAME="):
                        os_info["Codename"] = line.strip().split('=')[1].strip('"')
        except FileNotFoundError:
            logging.warning(f"'{os_release_path}' not found. Distribution information may be limited.")
        except Exception as e:
            logging.error(f"Error reading '{os_release_path}': {e}")

        # Attempt to get architecture from /proc/cpuinfo
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith("Architecture") or line.startswith("model name"):
                        os_info["Architecture"] = platform.machine()
                        break
        except Exception as e:
            logging.error(f"Error reading '/proc/cpuinfo': {e}")

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


# FreeBSD OS Class
class OSFreeBSD(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSFreeBSD._trusting_matches() called. Current OS: {current_os}")
        return current_os == "FreeBSD"

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, check for specific FreeBSD files
        specific_files = [
            '/etc/freebsd-version',
            '/bin/freebsd',
            '/sbin/init'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"Paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("Paranoid match succeeded: All specific FreeBSD files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering FreeBSD OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            os_info = {
                "OS": "FreeBSD",
                "Version": self._get_freebsd_version(),
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather OS info without relying on platform
        os_info = {
            "OS": "FreeBSD",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Attempt to read /etc/freebsd-version
        version_path = '/etc/freebsd-version'
        try:
            if os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    os_info["Version"] = f.read().strip()
        except Exception as e:
            logging.error(f"Error retrieving FreeBSD version in paranoid mode: {e}")
        # Attempt to read uname information
        try:
            release = subprocess.check_output(['uname', '-r'], text=True).strip()
            os_info["Release"] = release
        except Exception as e:
            logging.error(f"Error retrieving FreeBSD release in paranoid mode: {e}")
        return os_info

    def _get_freebsd_version(self) -> str:
        try:
            version = subprocess.check_output(['freebsd-version'], text=True).strip()
        except Exception as e:
            logging.error(f"Error retrieving FreeBSD version: {e}")
            version = "Unknown"
        return version


# OpenBSD OS Class
class OSOpenBSD(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSOpenBSD._trusting_matches() called. Current OS: {current_os}")
        return current_os == "OpenBSD"

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, check for specific OpenBSD files
        specific_files = [
            '/etc/openbsd-version',
            '/bin/ksh',
            '/sbin/init'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"Paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("Paranoid match succeeded: All specific OpenBSD files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering OpenBSD OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            try:
                version = subprocess.check_output(['uname', '-r'], text=True).strip()
            except Exception as e:
                logging.error(f"Error retrieving OpenBSD version: {e}")
                version = "Unknown"
            os_info = {
                "OS": "OpenBSD",
                "Version": version,
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather OS info without relying on platform
        os_info = {
            "OS": "OpenBSD",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Attempt to read /etc/openbsd-version
        version_path = '/etc/openbsd-version'
        try:
            if os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    os_info["Version"] = f.read().strip()
        except Exception as e:
            logging.error(f"Error retrieving OpenBSD version in paranoid mode: {e}")
        # Attempt to read uname information
        try:
            release = subprocess.check_output(['uname', '-r'], text=True).strip()
            os_info["Release"] = release
        except Exception as e:
            logging.error(f"Error retrieving OpenBSD release in paranoid mode: {e}")
        return os_info


# NetBSD OS Class
class OSNetBSD(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSNetBSD._trusting_matches() called. Current OS: {current_os}")
        return current_os == "NetBSD"

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, check for specific NetBSD files
        specific_files = [
            '/etc/NetBSD-version',
            '/bin/sh',
            '/sbin/init'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"Paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("Paranoid match succeeded: All specific NetBSD files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering NetBSD OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            try:
                version = subprocess.check_output(['uname', '-r'], text=True).strip()
            except Exception as e:
                logging.error(f"Error retrieving NetBSD version: {e}")
                version = "Unknown"
            os_info = {
                "OS": "NetBSD",
                "Version": version,
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather OS info without relying on platform
        os_info = {
            "OS": "NetBSD",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Attempt to read /etc/NetBSD-version
        version_path = '/etc/NetBSD-version'
        try:
            if os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    os_info["Version"] = f.read().strip()
        except Exception as e:
            logging.error(f"Error retrieving NetBSD version in paranoid mode: {e}")
        # Attempt to read uname information
        try:
            release = subprocess.check_output(['uname', '-r'], text=True).strip()
            os_info["Release"] = release
        except Exception as e:
            logging.error(f"Error retrieving NetBSD release in paranoid mode: {e}")
        return os_info


# Solaris OS Class
class OSSolaris(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        current_os = platform.system()
        logging.debug(f"OSSolaris._trusting_matches() called. Current OS: {current_os}")
        return current_os in ["SunOS", "Solaris"]

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, check for specific Solaris files
        specific_files = [
            '/etc/release',
            '/usr/bin/zonename',
            '/sbin/init'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"Paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("Paranoid match succeeded: All specific Solaris files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Solaris OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            try:
                version = subprocess.check_output(['uname', '-r'], text=True).strip()
            except Exception as e:
                logging.error(f"Error retrieving Solaris version: {e}")
                version = "Unknown"
            os_info = {
                "OS": "Solaris",
                "Version": version,
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather OS info without relying on platform
        os_info = {
            "OS": "Solaris",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Attempt to read /etc/release
        release_path = '/etc/release'
        try:
            if os.path.exists(release_path):
                with open(release_path, 'r') as f:
                    os_info["Version"] = f.read().strip().replace('\n', ' ')
        except Exception as e:
            logging.error(f"Error retrieving Solaris version in paranoid mode: {e}")
        # Attempt to read uname information
        try:
            release = subprocess.check_output(['uname', '-r'], text=True).strip()
            os_info["Release"] = release
        except Exception as e:
            logging.error(f"Error retrieving Solaris release in paranoid mode: {e}")
        return os_info


# Generic OS Class for Unsupported Systems
class OSGeneric(OSBase):
    def matches(self) -> bool:
        if self.paranoid:
            return self._paranoid_matches()
        else:
            return self._trusting_matches()

    def _trusting_matches(self) -> bool:
        # GenericOS matches if no other OS classes do
        logging.debug("OSGeneric._trusting_matches() called. Always returns True as fallback.")
        return True

    def _paranoid_matches(self) -> bool:
        # In paranoid mode, perform minimal checks
        # For example, check for the existence of /bin/sh and /etc/passwd
        specific_files = [
            '/bin/sh',
            '/etc/passwd'
        ]
        for file in specific_files:
            if not os.path.exists(file):
                logging.debug(f"OSGeneric paranoid match failed: '{file}' does not exist.")
                return False
        logging.debug("OSGeneric paranoid match succeeded: Required generic files found.")
        return True

    def get_os_info(self) -> Dict[str, Any]:
        logging.debug("Gathering generic OS information.")
        if self.paranoid:
            return self._paranoid_get_os_info()
        else:
            os_info = {
                "OS": platform.system(),
                "Version": platform.version(),
                "Release": platform.release(),
                "Architecture": platform.machine(),
                "Processor": platform.processor(),
            }
            return os_info

    def _paranoid_get_os_info(self) -> Dict[str, Any]:
        # In paranoid mode, gather generic OS info without relying on platform
        os_info = {
            "OS": "Unknown",
            "Version": "Unknown",
            "Release": "Unknown",
            "Architecture": "Unknown",
            "Processor": "Unknown",
        }
        # Attempt to read /etc/os-release if exists
        os_release_path = "/etc/os-release"
        try:
            if os.path.exists(os_release_path):
                with open(os_release_path, 'r') as f:
                    for line in f:
                        if line.startswith("NAME="):
                            os_info["OS"] = line.strip().split('=')[1].strip('"')
                        elif line.startswith("VERSION_ID="):
                            os_info["Version"] = line.strip().split('=')[1].strip('"')
                        elif line.startswith("VERSION_CODENAME="):
                            os_info["Release"] = line.strip().split('=')[1].strip('"')
        except Exception as e:
            logging.error(f"Error reading '{os_release_path}' in paranoid mode: {e}")
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
                        if 'docker' in line or 'kubepods' in line or 'containerd' in line:
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


# LXC/LXD Detection Class
class VMLXC(VMBase):
    def matches(self) -> bool:
        logging.debug("VMLXC.matches() called.")
        lxc_env = False
        try:
            with open('/proc/1/cgroup', 'rt') as f:
                for line in f:
                    if 'lxc' in line or 'lxd' in line:
                        logging.debug("Detected LXC/LXD indicators in cgroup.")
                        lxc_env = True
                        break
        except Exception as e:
            logging.error(f"Error reading '/proc/1/cgroup': {e}")

        # Check for LXC-specific files
        if not lxc_env:
            if os.path.exists('/var/lib/lxc/') or os.path.exists('/etc/lxc/'):
                logging.debug("Detected presence of LXC/LXD configuration directories.")
                lxc_env = True

        return lxc_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering LXC/LXD environment information.")
        version = self._get_lxc_version()
        return {"Environment": "LXC/LXD", "Version": version}

    def _get_lxc_version(self) -> str:
        """
        Attempts to retrieve the LXC version using the LXC CLI.
        """
        try:
            result = subprocess.run(['lxc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"LXC version output: {version_info}")
                return version_info
            else:
                logging.warning(f"LXC CLI returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("LXC CLI not found.")
        except subprocess.TimeoutExpired:
            logging.warning("LXC version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving LXC version: {e}")
        return "Unknown"


# systemd-nspawn Detection Class
class VMSYSTEMDNSPAWN(VMBase):
    def matches(self) -> bool:
        logging.debug("VMSYSTEMDNSPAWN.matches() called.")
        systemd_spawn = False
        try:
            with open('/proc/1/cgroup', 'rt') as f:
                for line in f:
                    if 'systemd-nspawn' in line:
                        logging.debug("Detected systemd-nspawn indicators in cgroup.")
                        systemd_spawn = True
                        break
        except Exception as e:
            logging.error(f"Error reading '/proc/1/cgroup': {e}")
        return systemd_spawn

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering systemd-nspawn environment information.")
        version = self._get_systemd_nspawn_version()
        return {"Environment": "systemd-nspawn", "Version": version}

    def _get_systemd_nspawn_version(self) -> str:
        """
        Attempts to retrieve the systemd-nspawn version.
        """
        try:
            result = subprocess.run(['systemd-nspawn', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"systemd-nspawn version output: {version_info}")
                return version_info
            else:
                logging.warning(f"systemd-nspawn returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("systemd-nspawn not found.")
        except subprocess.TimeoutExpired:
            logging.warning("systemd-nspawn version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving systemd-nspawn version: {e}")
        return "Unknown"


# Chroot Detection Class
class VMChroot(VMBase):
    def matches(self) -> bool:
        logging.debug("VMChroot.matches() called.")
        # Detecting chroot is non-trivial; one heuristic is checking if /proc/1 is not the same as the current process
        try:
            with open('/proc/1/comm', 'rt') as f:
                init_process = f.read().strip()
            current_comm = subprocess.check_output(['ps', '-p', '1', '-o', 'comm='], text=True).strip()
            if init_process != current_comm:
                logging.debug("Chroot environment detected based on init process mismatch.")
                return True
        except Exception as e:
            logging.error(f"Error detecting chroot environment: {e}")
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Chroot environment information.")
        return {"Environment": "Chroot"}


# Podman Detection Class
class VMPodman(VMBase):
    def matches(self) -> bool:
        logging.debug("VMPodman.matches() called.")
        podman_env = False
        if os.path.exists('/run/.containerenv') or os.path.exists('/.containerenv'):
            logging.debug("Detected presence of Podman environment files.")
            podman_env = True
        else:
            # Attempt to detect Podman via cgroup
            try:
                with open('/proc/1/cgroup', 'rt') as f:
                    for line in f:
                        if 'podman' in line:
                            logging.debug("Detected Podman indicators in cgroup.")
                            podman_env = True
                            break
            except Exception as e:
                logging.error(f"Error reading '/proc/1/cgroup': {e}")

        return podman_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Podman environment information.")
        version = self._get_podman_version()
        return {"Environment": "Podman", "Version": version}

    def _get_podman_version(self) -> str:
        """
        Attempts to retrieve the Podman version using the Podman CLI.
        """
        try:
            result = subprocess.run(['podman', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                # Example output: "podman version 3.3.1"
                version_info = result.stdout.strip()
                logging.debug(f"Podman version output: {version_info}")
                return version_info
            else:
                logging.warning(f"Podman CLI returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("Podman CLI not found.")
        except subprocess.TimeoutExpired:
            logging.warning("Podman version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving Podman version: {e}")
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
                    if "vmware" in product_name or "qemu" in product_name:
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


# Xen Detection Class
class VMXen(VMBase):
    def matches(self) -> bool:
        logging.debug("VMXen.matches() called.")
        xen_env = False
        # Check for Xen hypervisor in CPU flags
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'xen' in cpuinfo:
                    logging.debug("Xen environment detected via CPU flags.")
                    xen_env = True
        except Exception as e:
            logging.error(f"Error reading '/proc/cpuinfo': {e}")

        # Check for Xen-specific files
        if not xen_env and os.path.exists('/proc/xen'):
            logging.debug("Xen environment detected via '/proc/xen'.")
            xen_env = True

        return xen_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Xen environment information.")
        version = self._get_xen_version()
        return {"Environment": "Xen", "Version": version}

    def _get_xen_version(self) -> str:
        """
        Attempts to retrieve the Xen hypervisor version.
        """
        version = "Unknown"
        try:
            result = subprocess.run(['xen', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"Xen version output: {version_info}")
                return version_info
            else:
                logging.warning(f"Xen CLI returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("Xen CLI not found.")
        except subprocess.TimeoutExpired:
            logging.warning("Xen version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving Xen version: {e}")
        return version


# Hyper-V Detection Class
class VMHyperV(VMBase):
    def matches(self) -> bool:
        logging.debug("VMHyperV.matches() called.")
        hyperv_env = False
        # Check for Hyper-V specific CPU flags
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'hyperv' in cpuinfo:
                    logging.debug("Hyper-V environment detected via CPU flags.")
                    hyperv_env = True
        except Exception as e:
            logging.error(f"Error reading '/proc/cpuinfo': {e}")

        # Check for Hyper-V specific files
        if not hyperv_env and os.path.exists('/sys/devices/virtual/dmi/id/product_name'):
            try:
                with open('/sys/devices/virtual/dmi/id/product_name', 'r') as f:
                    product_name = f.read().strip().lower()
                    if "microsoft" in product_name or "virtual machine" in product_name:
                        logging.debug("Hyper-V environment detected via product name.")
                        hyperv_env = True
            except Exception as e:
                logging.error(f"Error reading product name: {e}")

        return hyperv_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Hyper-V environment information.")
        version = self._get_hyperv_version()
        return {"Environment": "Hyper-V", "Version": version}

    def _get_hyperv_version(self) -> str:
        """
        Attempts to retrieve the Hyper-V version.
        """
        version = "Unknown"
        # Hyper-V does not have a standard CLI tool for version retrieval on Linux
        # Attempt to parse system logs or use specific tools if available
        try:
            result = subprocess.run(['dmesg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'Hyper-V' in line:
                        version_info = line.strip()
                        logging.debug(f"Hyper-V detected in dmesg: {version_info}")
                        return version_info
            else:
                logging.warning(f"dmesg returned non-zero exit code: {result.stderr.strip()}")
        except Exception as e:
            logging.error(f"Error retrieving Hyper-V version from dmesg: {e}")
        return version


# Parallels Detection Class
class VMParallels(VMBase):
    def matches(self) -> bool:
        logging.debug("VMParallels.matches() called.")
        parallels_env = False
        # Check for Parallels Tools specific files
        if os.path.exists('/usr/lib/parallels-tools'):
            logging.debug("Parallels environment detected via '/usr/lib/parallels-tools'.")
            parallels_env = True

        # Check product name
        try:
            product_name_path = '/sys/class/dmi/id/product_name'
            if os.path.exists(product_name_path):
                with open(product_name_path, 'r') as f:
                    product_name = f.read().strip().lower()
                    if "parallels" in product_name:
                        logging.debug("Parallels environment detected via product name.")
                        parallels_env = True
        except Exception as e:
            logging.error(f"Error reading '{product_name_path}': {e}")

        return parallels_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Parallels environment information.")
        version = self._get_parallels_version()
        return {"Environment": "Parallels", "Version": version}

    def _get_parallels_version(self) -> str:
        """
        Attempts to retrieve the Parallels Tools version.
        """
        version = "Unknown"
        try:
            result = subprocess.run(['prlsrvctl', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"Parallels Tools version output: {version_info}")
                return version_info
            else:
                logging.warning(f"prlsrvctl returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("prlsrvctl not found.")
        except subprocess.TimeoutExpired:
            logging.warning("prlsrvctl version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving Parallels version: {e}")
        return version


# QEMU Detection Class
class VMQEMU(VMBase):
    def matches(self) -> bool:
        logging.debug("VMQEMU.matches() called.")
        qemu_env = False
        # Check for QEMU specific CPU flags
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'qemu' in cpuinfo or 'tcg' in cpuinfo:
                    logging.debug("QEMU environment detected via CPU flags.")
                    qemu_env = True
        except Exception as e:
            logging.error(f"Error reading '/proc/cpuinfo': {e}")

        # Check for QEMU-specific files or processes
        if not qemu_env:
            try:
                result = subprocess.run(['pgrep', '-f', 'qemu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                if result.stdout.strip():
                    logging.debug("QEMU environment detected via running processes.")
                    qemu_env = True
            except Exception as e:
                logging.error(f"Error detecting QEMU processes: {e}")

        return qemu_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering QEMU environment information.")
        version = self._get_qemu_version()
        return {"Environment": "QEMU", "Version": version}

    def _get_qemu_version(self) -> str:
        """
        Attempts to retrieve the QEMU version.
        """
        version = "Unknown"
        try:
            result = subprocess.run(['qemu-system-x86_64', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"QEMU version output: {version_info}")
                return version_info
            else:
                logging.warning(f"QEMU CLI returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("QEMU CLI not found.")
        except subprocess.TimeoutExpired:
            logging.warning("QEMU version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving QEMU version: {e}")
        return version


# Singularity Detection Class
class VMSingularity(VMBase):
    def matches(self) -> bool:
        logging.debug("VMSingularity.matches() called.")
        singularity_env = False
        # Check for Singularity environment variables
        if 'SINGULARITY_NAME' in os.environ:
            logging.debug("Singularity environment detected via environment variables.")
            singularity_env = True

        # Check for Singularity specific files
        if not singularity_env and os.path.exists('/.singularity.d'):
            logging.debug("Singularity environment detected via '/.singularity.d' directory.")
            singularity_env = True

        return singularity_env

    def get_info(self) -> Optional[Dict[str, Any]]:
        logging.debug("Gathering Singularity environment information.")
        version = self._get_singularity_version()
        return {"Environment": "Singularity", "Version": version}

    def _get_singularity_version(self) -> str:
        """
        Attempts to retrieve the Singularity version.
        """
        version = "Unknown"
        try:
            result = subprocess.run(['singularity', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logging.debug(f"Singularity version output: {version_info}")
                return version_info
            else:
                logging.warning(f"Singularity CLI returned non-zero exit code: {result.stderr.strip()}")
        except FileNotFoundError:
            logging.warning("Singularity CLI not found.")
        except subprocess.TimeoutExpired:
            logging.warning("Singularity version command timed out.")
        except Exception as e:
            logging.error(f"Error retrieving Singularity version: {e}")
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
    def __init__(self, perform_all_checks: bool = False, paranoid: bool = False) -> None:
        self.perform_all_checks = perform_all_checks
        self.paranoid = paranoid
        self.os_detectors = self._initialize_os_detectors()
        self.vm_detectors = self._initialize_vm_detectors()

    def _initialize_os_detectors(self) -> List[OSBase]:
        logging.debug("Initializing OS detectors.")
        return [
            OSWindows(paranoid=self.paranoid),
            OSMacOS(paranoid=self.paranoid),
            OSLinux(paranoid=self.paranoid),
            OSFreeBSD(paranoid=self.paranoid),
            OSOpenBSD(paranoid=self.paranoid),
            OSNetBSD(paranoid=self.paranoid),
            OSSolaris(paranoid=self.paranoid),
            OSGeneric(paranoid=self.paranoid),  # GenericOS as fallback
        ]

    def _initialize_vm_detectors(self) -> List[VMBase]:
        logging.debug("Initializing virtualization detectors.")
        return [
            VMDocker(),
            VMLXC(),
            VMSYSTEMDNSPAWN(),
            VMChroot(),
            VMPodman(),
            VMVirtualBox(),
            VMMware(),
            VMKVM(),
            VMXen(),
            VMHyperV(),
            VMParallels(),
            VMQEMU(),
            VMSingularity(),
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
        '-p', '--paranoid',
        action='store_true',
        help='Enable paranoid mode: perform untrusting OS checks.'
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

    detector = SystemDetector(perform_all_checks=args.all, paranoid=args.paranoid)
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
