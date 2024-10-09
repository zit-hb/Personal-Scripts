#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_system.py
#
# Description:
# This script detects the underlying operating system and identifies if it is running within a virtualized environment.
# It provides detailed information about the OS, including distribution and version for Linux systems, and identifies
# the type of virtualization (e.g., Docker, VirtualBox, VMWare) if present, along with version details where possible.
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
import logging
import subprocess
import sys
import json
import platform
import os
import ctypes
import typing
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

# Constants for file paths and registry keys
LINUX_OS_RELEASE_PATH = '/etc/os-release'
PROC_SCSI_SCSI = '/proc/scsi/scsi'


class OperatingSystemType(Enum):
    """Enum representing supported operating system types."""
    WINDOWS = 'Windows'
    LINUX = 'Linux'
    MACOS = 'MacOS'
    FREEBSD = 'FreeBSD'
    UNKNOWN = 'Unknown'


class VirtualMachineType(Enum):
    """Enum representing supported virtual machine types."""
    VMWARE = 'VMware'
    VIRTUALBOX = 'VirtualBox'
    HYPERV = 'Hyper-V'
    KVM = 'KVM'
    XEN = 'Xen'
    UNKNOWN = 'Unknown'


class SandboxType(Enum):
    """Enum representing supported sandbox types."""
    DOCKER = 'Docker'
    KUBERNETES = 'Kubernetes'
    UNKNOWN = 'Unknown'


@dataclass
class OperatingSystemInfo:
    """Data class for storing operating system information."""
    type: OperatingSystemType = OperatingSystemType.UNKNOWN
    version: str = ''
    architecture: str = ''
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VirtualMachineInfo:
    """Data class for storing virtual machine information."""
    type: VirtualMachineType = VirtualMachineType.UNKNOWN
    version: str = ''
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxInfo:
    """Data class for storing sandbox environment information."""
    type: SandboxType = SandboxType.UNKNOWN
    version: str = ''
    additional_info: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract base class for detectors."""

    @abstractmethod
    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[Any]:
        pass


class LinuxOperatingSystemDetector(BaseDetector):
    """Detector for Linux operating systems."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[OperatingSystemInfo]:
        """Detects if the operating system is Linux and identifies its distribution."""
        if not self._is_running(paranoid):
            return None

        additional_info = {
            'Distribution': self._get_distribution()
        }

        if not paranoid:
            return OperatingSystemInfo(
                type=OperatingSystemType.LINUX,
                architecture=platform.machine(),
                version=platform.release(),
                additional_info=additional_info
            )

        return OperatingSystemInfo(
            type=OperatingSystemType.LINUX,
            architecture=self._get_architecture(),
            version=self._get_version(),
            additional_info=additional_info
        )

    def _is_running(self, paranoid: bool = False) -> bool:
        """Detects if the system is running Linux."""
        if not paranoid:
            return platform.system().lower() == OperatingSystemType.LINUX.value.lower()

        # Paranoid detection: Check for Linux-specific files
        linux_indicators = [
            "/proc/version",
            "/etc/os-release",
            "/bin/bash",
            "/usr/bin/ls",
            "/usr/bin/grep",
            "/usr/bin/awk",
        ]

        missing_files = [f for f in linux_indicators if not os.path.exists(f)]
        if missing_files:
            logging.debug(f"Paranoid detection failed. Missing files: {missing_files}")
            return False

        return True

    def _get_architecture(self) -> str:
        """Gets the machine architecture by executing specific commands."""
        try:
            arch = subprocess.check_output(["uname", "-m"], text=True).strip()
            logging.debug(f"Architecture from uname: {arch}")
            return arch
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing uname for architecture: {e}")
        except FileNotFoundError:
            logging.error("uname command not found.")
        return "Unknown"

    def _get_version(self) -> str:
        """Gets the kernel version by executing specific commands."""
        try:
            version = subprocess.check_output(["uname", "-r"], text=True).strip()
            logging.debug(f"Version from uname: {version}")
            return version
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing uname for version: {e}")
        except FileNotFoundError:
            logging.error("uname command not found.")
        return "Unknown"

    def _get_distribution(self) -> Optional[str]:
        """Gets the Linux distribution by reading /etc/os-release."""
        try:
            with open(LINUX_OS_RELEASE_PATH, "r") as f:
                os_release = self._parse_os_release(f)
                distribution = os_release.get("NAME") + " " + os_release.get("VERSION")
                logging.debug(f"Distribution details from {LINUX_OS_RELEASE_PATH}: {distribution}")
                return distribution
        except Exception as e:
            logging.error(f"Error reading {LINUX_OS_RELEASE_PATH} for distribution: {e}")
        return None

    def _parse_os_release(self, file) -> Dict[str, str]:
        """Parses the /etc/os-release file into a dictionary."""
        os_release = {}
        for line in file:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                os_release[key.upper()] = value
        logging.debug(f"Parsed {LINUX_OS_RELEASE_PATH}: {os_release}")
        return os_release

    def _parse_os_release_field(self, field: str) -> Optional[str]:
        """Parses a specific field from /etc/os-release."""
        try:
            with open(LINUX_OS_RELEASE_PATH, "r") as f:
                for line in f:
                    if line.startswith(f"{field}="):
                        _, value = line.strip().split("=", 1)
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        logging.debug(f"Field {field} from {LINUX_OS_RELEASE_PATH}: {value}")
                        return value
        except Exception as e:
            logging.error(f"Error parsing {field} from {LINUX_OS_RELEASE_PATH}: {e}")
        return None


class WindowsOperatingSystemDetector(BaseDetector):
    """Detector for Windows operating systems."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[OperatingSystemInfo]:
        """Detects if the operating system is Windows."""
        if not self._is_running(paranoid):
            return None

        if not paranoid:
            return OperatingSystemInfo(
                type=OperatingSystemType.WINDOWS,
                architecture=platform.machine(),
                version=self._get_version_from_platform()
            )

        return OperatingSystemInfo(
            type=OperatingSystemType.WINDOWS,
            architecture=self._get_architecture(),
            version=self._get_version(),
        )

    def _is_running(self, paranoid: bool = False) -> bool:
        """Detects if the system is running Windows."""
        if not paranoid:
            return platform.system().lower() == OperatingSystemType.WINDOWS.value.lower()

        # Paranoid detection: Check for Windows-specific files
        system_root = os.environ.get("SystemRoot", "C:\\Windows")
        windows_indicators = [
            os.path.join(system_root, "System32", "kernel32.dll"),
            os.path.join(system_root, "System32", "cmd.exe"),
            os.path.join(system_root, "System32", "notepad.exe"),
            os.path.join(system_root, "explorer.exe"),
            os.path.join(system_root, "System32", "drivers", "etc", "hosts"),
        ]

        missing_files = [f for f in windows_indicators if not os.path.exists(f)]
        if missing_files:
            logging.debug(f"Paranoid detection failed. Missing files: {missing_files}")
            return False

        return True

    def _get_architecture(self) -> str:
        """Gets the machine architecture by executing specific commands."""
        try:
            arch_output = subprocess.check_output(
                ["wmic", "os", "get", "OSArchitecture"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip().split('\n')
            if len(arch_output) < 2:
                logging.debug("wmic output is insufficient to determine OS architecture.")
                return "Unknown"
            architecture = arch_output[1].strip()
            logging.debug(f"Architecture from wmic: {architecture}")
            return architecture
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing wmic for architecture: {e}")
        except FileNotFoundError:
            logging.error("wmic command not found.")
        return "Unknown"

    def _get_version(self) -> str:
        """Gets the OS version by executing specific commands."""
        try:
            version_output = subprocess.check_output(
                ["wmic", "os", "get", "Version"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip().split('\n')
            if len(version_output) < 2:
                logging.debug("wmic output is insufficient to determine OS version.")
                return "Unknown"
            version = version_output[1].strip()
            logging.debug(f"OS version from wmic: {version}")
            return version
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing wmic for version: {e}")
        except FileNotFoundError:
            logging.error("wmic command not found.")
        return "Unknown"

    def _get_version_from_platform(self) -> str:
        """Gets the OS version using the platform module."""
        try:
            version = platform.version()
            logging.debug(f"OS version from platform: {version}")
            return version
        except Exception as e:
            logging.error(f"Error getting version from platform: {e}")
        return "Unknown"


class MacOSOperatingSystemDetector(BaseDetector):
    """Detector for macOS operating systems."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[OperatingSystemInfo]:
        """Detects if the operating system is macOS."""
        if not self._is_running(paranoid):
            return None

        if not paranoid:
            return OperatingSystemInfo(
                type=OperatingSystemType.MACOS,
                architecture=platform.machine(),
                version=platform.mac_ver()[0]
            )

        return OperatingSystemInfo(
            type=OperatingSystemType.MACOS,
            architecture=self._get_architecture(),
            version=self._get_version(),
        )

    def _is_running(self, paranoid: bool = False) -> bool:
        """Detects if the system is running MacOS."""
        if not paranoid:
            return platform.system().lower() == OperatingSystemType.MACOS.value.lower()

        # Paranoid detection: Check for macOS-specific files
        macos_indicators = [
            "/System/Library/CoreServices/SystemVersion.plist",
            "/usr/bin/sw_vers",
            "/Applications",
            "/System",
            "/usr/bin/osascript",
        ]

        missing_files = [f for f in macos_indicators if not os.path.exists(f)]
        if missing_files:
            logging.debug(f"Paranoid detection failed. Missing files: {missing_files}")
            return False

        return True

    def _get_architecture(self) -> str:
        """Gets the machine architecture by reading specific system files or executing commands."""
        try:
            arch = subprocess.check_output(["sysctl", "-n", "hw.machine"], text=True).strip()
            logging.debug(f"Architecture from sysctl: {arch}")
            return arch
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing sysctl for architecture: {e}")
        except FileNotFoundError:
            logging.error("sysctl command not found.")
        return "Unknown"

    def _get_version(self) -> str:
        """Gets the OS version by executing sw_vers."""
        try:
            version = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
            logging.debug(f"OS version from sw_vers: {version}")
            return version
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing sw_vers for version: {e}")
        except FileNotFoundError:
            logging.error("sw_vers command not found.")
        return "Unknown"


class FreeBSDOperatingSystemDetector(BaseDetector):
    """Detector for FreeBSD operating systems."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[OperatingSystemInfo]:
        """Detects if the operating system is FreeBSD."""
        if not self._is_running(paranoid):
            return None

        if not paranoid:
            return OperatingSystemInfo(
                type=OperatingSystemType.FREEBSD,
                architecture=platform.machine(),
                version=platform.release()
            )

        return OperatingSystemInfo(
            type=OperatingSystemType.FREEBSD,
            architecture=self._get_architecture(),
            version=self._get_version(),
        )

    def _is_running(self, paranoid: bool = False) -> bool:
        """Detects if the system is running FreeBSD."""
        if not paranoid:
            return platform.system().lower() == OperatingSystemType.FREEBSD.value.lower()

        # Paranoid detection: Check for the existence of FreeBSD-specific files
        freebsd_indicators = [
            "/etc/freebsd_version",
            "/usr/bin/sysctl",
            "/sbin/init",
            "/bin/sh",
            "/var/db/ports",  # Ports Collection directory
        ]

        missing_files = [f for f in freebsd_indicators if not os.path.exists(f)]
        if missing_files:
            logging.debug(f"Paranoid detection failed. Missing files: {missing_files}")
            return False

        return True

    def _get_architecture(self) -> str:
        """Gets the machine architecture by reading specific system files or executing commands."""
        try:
            arch = subprocess.check_output(["sysctl", "-n", "hw.machine"], text=True).strip()
            logging.debug(f"Architecture from sysctl: {arch}")
            return arch
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing sysctl for architecture: {e}")
        except FileNotFoundError:
            logging.error("sysctl command not found.")
        return "Unknown"

    def _get_version(self) -> str:
        """Gets the OS version by reading /etc/freebsd_version."""
        try:
            with open("/etc/freebsd_version", "r") as f:
                version_info = f.read().strip()
                logging.debug(f"OS version from /etc/freebsd_version: {version_info}")
                return version_info
        except Exception as e:
            logging.error(f"Error reading /etc/freebsd_version for version: {e}")
        return "Unknown"


class VMWareVirtualMachineDetector(BaseDetector):
    """Detector for VMWare virtual machines."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[VirtualMachineInfo]:
        """Detects if the system is running on VMWare."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        elif os_info.type == OperatingSystemType.WINDOWS:
            detected = self._detect_windows(paranoid)
        else:
            return None

        if detected:
            return VirtualMachineInfo(
                type=VirtualMachineType.VMWARE,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Detects VMWare on Linux systems."""
        try:
            with open('/sys/class/dmi/id/product_name', 'r', encoding='utf-8') as f:
                if 'vmware' in f.read().lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading product_name: {e}")

        try:
            with open('/sys/class/dmi/id/sys_vendor', 'r', encoding='utf-8') as f:
                if 'vmware' in f.read().lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading sys_vendor: {e}")

        if paranoid:
            try:
                with open(PROC_SCSI_SCSI, 'r', encoding='utf-8') as f:
                    if 'vmware' in f.read().lower():
                        return True
            except Exception as e:
                logging.debug(f"Error reading {PROC_SCSI_SCSI}: {e}")
        return False

    def _detect_windows(self, paranoid: bool) -> bool:
        """Detects VMWare on Windows systems."""
        if os.name.lower() != 'nt':
            return False
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            buffer_size = kernel32.GetSystemFirmwareTable(0x52534D42, 0, None, 0)
            if buffer_size == 0:
                return False
            buffer = ctypes.create_string_buffer(buffer_size)
            if kernel32.GetSystemFirmwareTable(0x52534D42, 0, buffer, buffer_size) == 0:
                return False
            if b'vmware' in buffer.raw.lower():
                return True
        except Exception as e:
            logging.debug(f"Error detecting VMWare on Windows: {e}")

        if paranoid:
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\Disk\Enum') as key:
                    device0, _ = winreg.QueryValueEx(key, '0')
                    if 'vmware' in device0.lower():
                        return True
            except Exception as e:
                logging.debug(f"Error reading Windows registry for VMWare detection: {e}")
        return False


class VirtualBoxVirtualMachineDetector(BaseDetector):
    """Detector for VirtualBox virtual machines."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[VirtualMachineInfo]:
        """Detects if the system is running on VirtualBox."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        elif os_info.type == OperatingSystemType.WINDOWS:
            detected = self._detect_windows(paranoid)
        elif os_info.type == OperatingSystemType.MACOS:
            detected = self._detect_macos(paranoid)
        else:
            return None

        if detected:
            return VirtualMachineInfo(
                type=VirtualMachineType.VIRTUALBOX,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Detects VirtualBox on Linux systems."""
        try:
            with open('/sys/class/dmi/id/product_name', 'r', encoding='utf-8') as f:
                if 'virtualbox' in f.read().lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading product_name: {e}")

        if paranoid:
            try:
                with open('/proc/modules', 'r', encoding='utf-8') as f:
                    if 'vboxguest' in f.read().lower():
                        return True
            except Exception as e:
                logging.debug(f"Error reading /proc/modules: {e}")
        return False

    def _detect_windows(self, paranoid: bool) -> bool:
        """Detects VirtualBox on Windows systems."""
        if os.name.lower() != 'nt':
            return False
        try:
            import wmi
            c = wmi.WMI()
            for system in c.Win32_ComputerSystem():
                if 'virtualbox' in system.Manufacturer.lower() or 'virtualbox' in system.Model.lower():
                    return True
        except ImportError:
            logging.error("wmi module not available.")
        except Exception as e:
            logging.debug(f"Error detecting VirtualBox on Windows: {e}")
        return False

    def _detect_macos(self, paranoid: bool) -> bool:
        """Detects VirtualBox on macOS systems."""
        try:
            import subprocess
            output = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], encoding='utf-8')
            if 'virtualbox' in output.lower():
                return True
        except Exception as e:
            logging.debug(f"Error running system_profiler: {e}")
        return False


class HyperVVirtualMachineDetector(BaseDetector):
    """Detector for Microsoft Hyper-V virtual machines."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[VirtualMachineInfo]:
        """Detects if the system is running on Hyper-V."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.WINDOWS:
            detected = self._detect_windows(paranoid)
        else:
            return None

        if detected:
            return VirtualMachineInfo(
                type=VirtualMachineType.HYPERV,
            )
        return None

    def _detect_windows(self, paranoid: bool) -> bool:
        """Detects Hyper-V on Windows systems."""
        try:
            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [("wProcessorArchitecture", ctypes.c_uint16),
                            ("wReserved", ctypes.c_uint16),
                            ("dwPageSize", ctypes.c_uint32),
                            ("lpMinimumApplicationAddress", ctypes.c_void_p),
                            ("lpMaximumApplicationAddress", ctypes.c_void_p),
                            ("dwActiveProcessorMask", ctypes.c_void_p),
                            ("dwNumberOfProcessors", ctypes.c_uint32),
                            ("dwProcessorType", ctypes.c_uint32),
                            ("dwAllocationGranularity", ctypes.c_uint32),
                            ("wProcessorLevel", ctypes.c_uint16),
                            ("wProcessorRevision", ctypes.c_uint16)]

            sys_info = SYSTEM_INFO()
            ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(sys_info))
            if sys_info.wProcessorArchitecture == 9:  # PROCESSOR_ARCHITECTURE_AMD64
                return True
        except Exception as e:
            logging.debug(f"Error detecting Hyper-V on Windows: {e}")
        return False


class KVMVirtualMachineDetector(BaseDetector):
    """Detector for KVM virtual machines."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[VirtualMachineInfo]:
        """Detects if the system is running on KVM."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        else:
            return None

        if detected:
            return VirtualMachineInfo(
                type=VirtualMachineType.KVM,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Detects KVM on Linux systems."""
        try:
            with open('/sys/class/dmi/id/product_name', 'r', encoding='utf-8') as f:
                if 'kvm' in f.read().lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading product_name: {e}")

        if paranoid:
            try:
                with open('/proc/cpuinfo', 'r', encoding='utf-8') as f:
                    if 'qemu' in f.read().lower():
                        return True
            except Exception as e:
                logging.debug(f"Error reading /proc/cpuinfo: {e}")
        return False


class XenVirtualMachineDetector(BaseDetector):
    """Detector for Xen virtual machines."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[VirtualMachineInfo]:
        """Detects if the system is running on Xen."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        else:
            return None

        if detected:
            return VirtualMachineInfo(
                type=VirtualMachineType.XEN,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Detects Xen on Linux systems."""
        try:
            with open('/sys/hypervisor/type', 'r', encoding='utf-8') as f:
                if 'xen' in f.read().lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading /sys/hypervisor/type: {e}")

        if paranoid:
            try:
                with open('/proc/xen/capabilities', 'r', encoding='utf-8') as f:
                    if 'control_d' in f.read().lower():
                        return True
            except Exception as e:
                logging.debug(f"Error reading /proc/xen/capabilities: {e}")
        return False


class DockerSandboxDetector(BaseDetector):
    """Detector for Docker sandbox environments."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[SandboxInfo]:
        """Detects if the system is running inside a Docker container."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        else:
            return None

        if detected:
            return SandboxInfo(
                type=SandboxType.DOCKER,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Detects Docker environment."""
        if os.path.exists('/.dockerenv'):
            return True
        try:
            with open('/proc/1/cgroup', 'r', encoding='utf-8') as f:
                content = f.read()
                if 'docker' in content.lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading /proc/1/cgroup: {e}")

        if paranoid:
            try:
                with open('/proc/self/cgroup', 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'docker' in content.lower():
                        return True
            except Exception as e:
                logging.debug(f"Error reading /proc/self/cgroup: {e}")
            if os.environ.get('container', '') == 'docker':
                return True
        return False


class KubernetesSandboxDetector(BaseDetector):
    """Detector for Kubernetes sandbox environments."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[SandboxInfo]:
        """Detects if the system is running inside a Kubernetes container."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        else:
            return None

        if detected:
            return SandboxInfo(
                type=SandboxType.KUBERNETES,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Detects Kubernetes environment."""
        try:
            with open('/proc/1/cgroup', 'r', encoding='utf-8') as f:
                content = f.read()
                if 'kubepods' in content.lower():
                    return True
        except Exception as e:
            logging.debug(f"Error reading /proc/1/cgroup: {e}")

        if paranoid and os.environ.get('KUBERNETES_SERVICE_HOST'):
            return True
        return False


class GenericSandboxDetector(BaseDetector):
    """Generic detector for sandbox environments."""

    def detect(self, os_info: Optional[OperatingSystemInfo] = None, paranoid: bool = False) -> Optional[SandboxInfo]:
        """Attempts to detect if the system is running inside any sandbox environment."""
        if os_info is None:
            return None

        if os_info.type == OperatingSystemType.LINUX:
            detected = self._detect_linux(paranoid)
        elif os_info.type == OperatingSystemType.WINDOWS:
            detected = self._detect_windows(paranoid)
        elif os_info.type == OperatingSystemType.MACOS:
            detected = self._detect_macos(paranoid)
        else:
            return None

        if detected:
            return SandboxInfo(
                type=SandboxType.UNKNOWN,
            )
        return None

    def _detect_linux(self, paranoid: bool) -> bool:
        """Generic sandbox detection on Linux systems."""
        try:
            # Check if we are in a chroot by comparing device and inode numbers
            root_stat = os.stat("/")
            parent_stat = os.stat("/..")
            if root_stat.st_ino != parent_stat.st_ino or root_stat.st_dev != parent_stat.st_dev:
                return True
        except Exception as e:
            logging.debug(f"Error checking chroot: {e}")

        try:
            # Check for restricted mount namespaces
            with open("/proc/1/mountinfo", "r", encoding="utf-8") as f:
                if len(f.readlines()) == 0:
                    return True
        except Exception as e:
            logging.debug(f"Error reading /proc/1/mountinfo: {e}")

        if paranoid:
            try:
                # Check for unprivileged user namespaces
                with open("/proc/self/status", "r", encoding="utf-8") as f:
                    content = f.read()
                    if "CapEff:\t00000000" in content:
                        return True
            except Exception as e:
                logging.debug(f"Error reading /proc/self/status: {e}")

        return False

    def _detect_windows(self, paranoid: bool) -> bool:
        """Generic sandbox detection on Windows systems."""
        try:
            # Check for common sandbox artifacts
            import os

            sandbox_files = [
                "C:\\Sandbox",
                "C:\\shadow",
                "C:\\virtual",
                "C:\\hwcv.exe",
                "C:\\crowdstrike",
            ]
            for path in sandbox_files:
                if os.path.exists(path):
                    return True

            # Check for sandbox environment variables
            sandbox_env_vars = ["SANDBOX", "VIRTUAL_ENV"]
            for var in sandbox_env_vars:
                if var in os.environ:
                    return True

            if paranoid:
                # Check for low integrity level
                class SID_AND_ATTRIBUTES(ctypes.Structure):
                    _fields_ = [
                        ("Sid", ctypes.POINTER(ctypes.c_void_p)),
                        ("Attributes", ctypes.c_uint32)
                    ]

                class TOKEN_MANDATORY_LABEL(ctypes.Structure):
                    _fields_ = [("Label", SID_AND_ATTRIBUTES)]

                h_token = ctypes.wintypes.HANDLE()
                TOKEN_QUERY = 0x0008
                token_integrity_level = 25
                if ctypes.windll.advapi32.OpenProcessToken(
                        ctypes.windll.kernel32.GetCurrentProcess(),
                        TOKEN_QUERY,
                        ctypes.byref(h_token),
                ):
                    info = TOKEN_MANDATORY_LABEL()
                    ret_len = ctypes.wintypes.DWORD()
                    ctypes.windll.advapi32.GetTokenInformation(
                        h_token,
                        token_integrity_level,
                        ctypes.byref(info),
                        ctypes.sizeof(info),
                        ctypes.byref(ret_len),
                    )
                    sub_auth = ctypes.cast(
                        info.Label[0], ctypes.POINTER(ctypes.wintypes.DWORD)
                    )[2]
                    LOW_INTEGRITY = 0x1000
                    if sub_auth == LOW_INTEGRITY:
                        return True
        except Exception as e:
            logging.debug(f"Error detecting sandbox on Windows: {e}")
        return False

    def _detect_macos(self, paranoid: bool) -> bool:
        """Generic sandbox detection on macOS systems."""
        try:
            # Check for sandbox environment variables
            if "APP_SANDBOX_CONTAINER_ID" in os.environ:
                return True

            # Check for sandboxed file system paths
            if os.path.exists("/System/Volumes/Data"):
                return True

            if paranoid:
                # Check for DYLD_INSERT_LIBRARIES
                if "DYLD_INSERT_LIBRARIES" in os.environ:
                    return True
        except Exception as e:
            logging.debug(f"Error detecting sandbox on macOS: {e}")
        return False


class DetectorManager:
    """Manager for registering and executing detectors."""

    def __init__(self):
        self.detectors: List[BaseDetector] = []

    def register_detector(self, detector: BaseDetector):
        """Registers a detector."""
        self.detectors.append(detector)

    def detect_all(self, os_info: Optional[OperatingSystemInfo], paranoid: bool) -> List[Any]:
        """Runs all registered detectors."""
        results = []
        for detector in self.detectors:
            result = detector.detect(os_info, paranoid)
            if result:
                results.append(result)
        return results


class EnvironmentDetector:
    """Main class for detecting the environment."""

    def __init__(self, perform_all_checks: bool = False, paranoid: bool = False):
        self.perform_all_checks = perform_all_checks
        self.paranoid = paranoid
        self.os_info: Optional[OperatingSystemInfo] = None
        self.vm_info: Optional[VirtualMachineInfo] = None
        self.sandbox_info: Optional[SandboxInfo] = None
        self.detector_manager = DetectorManager()
        self._register_detectors()
        self.os_detectors: List[BaseDetector] = []
        self.other_detectors: List[BaseDetector] = []
        self._group_detectors_by_return_type()

    def _register_detectors(self):
        """Registers all available detectors."""
        # Operating System Detectors
        self.detector_manager.register_detector(WindowsOperatingSystemDetector())
        self.detector_manager.register_detector(LinuxOperatingSystemDetector())
        self.detector_manager.register_detector(MacOSOperatingSystemDetector())
        self.detector_manager.register_detector(FreeBSDOperatingSystemDetector())

        # Virtual Machine Detectors
        self.detector_manager.register_detector(VMWareVirtualMachineDetector())
        self.detector_manager.register_detector(VirtualBoxVirtualMachineDetector())
        self.detector_manager.register_detector(HyperVVirtualMachineDetector())
        self.detector_manager.register_detector(KVMVirtualMachineDetector())
        self.detector_manager.register_detector(XenVirtualMachineDetector())

        # Sandbox Detectors
        self.detector_manager.register_detector(DockerSandboxDetector())
        self.detector_manager.register_detector(KubernetesSandboxDetector())
        self.detector_manager.register_detector(GenericSandboxDetector())

    def _group_detectors_by_return_type(self):
        """Groups detectors based on the return type of their detect method."""
        for detector in self.detector_manager.detectors:
            # Get the type hints of the detect method
            try:
                type_hints = typing.get_type_hints(detector.detect)
            except Exception as e:
                logging.warning(f"Failed to get type hints for {detector.__class__.__name__}: {e}")
                continue

            return_type = type_hints.get('return', None)
            if return_type is None:
                logging.warning(f"Detector {detector.__class__.__name__} has no return type annotation.")
                continue

            # Handle Optional[...] which is Union[..., NoneType]
            origin = typing.get_origin(return_type)
            args = typing.get_args(return_type)

            if origin is typing.Union and type(None) in args:
                # Extract the first argument that is not NoneType
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return_type_inner = non_none_args[0]
                else:
                    # More than one non-NoneType argument, ambiguous
                    return_type_inner = return_type
            else:
                return_type_inner = return_type

            # Ensure return_type_inner is a class before using issubclass
            if isinstance(return_type_inner, type):
                if issubclass(return_type_inner, OperatingSystemInfo):
                    self.os_detectors.append(detector)
                elif issubclass(return_type_inner, VirtualMachineInfo):
                    self.other_detectors.append(detector)
                elif issubclass(return_type_inner, SandboxInfo):
                    self.other_detectors.append(detector)
                else:
                    logging.warning(f"Detector {detector.__class__.__name__} has unknown return type {return_type_inner}.")
                    self.other_detectors.append(detector)
            else:
                logging.warning(f"Detector {detector.__class__.__name__} has non-class return type {return_type_inner}.")
                self.other_detectors.append(detector)

    def detect(self):
        """Performs the detection process."""
        # First, detect OS using all OS detectors
        for detector in self.os_detectors:
            try:
                os_info = detector.detect(paranoid=self.paranoid)
            except Exception as e:
                logging.error(f"Error during OS detection with {detector.__class__.__name__}: {e}")
                continue

            if os_info:
                self.os_info = os_info
                logging.info(f"Detected OS: {os_info.type.value}")
                break

        if self.os_info:
            # Detect VM and Sandbox using other detectors
            for detector in self.other_detectors:
                try:
                    result = detector.detect(self.os_info, paranoid=self.paranoid)
                except Exception as e:
                    logging.error(f"Error during detection with {detector.__class__.__name__}: {e}")
                    continue

                if isinstance(result, VirtualMachineInfo):
                    if self.vm_info is None or self.vm_info.type == VirtualMachineType.UNKNOWN:
                        self.vm_info = result
                        logging.info(f"Detected Virtual Machine: {result.type.value}")
                        if not self.perform_all_checks:
                            break
                elif isinstance(result, SandboxInfo):
                    if self.sandbox_info is None or self.sandbox_info.type == SandboxType.UNKNOWN:
                        self.sandbox_info = result
                        logging.info(f"Detected Sandbox: {result.type.value}")
                        if not self.perform_all_checks:
                            break

        # If perform_all_checks is False, stop after first detection
        if not self.perform_all_checks:
            return


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
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
    """Sets up the logging configuration."""
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
    """Saves the detection results to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Detection results saved to '{filepath}'.")
        return True
    except Exception as e:
        logging.error(f"Error saving detection results: {e}")
        return False


def display_results(data: Dict[str, Any]) -> None:
    """Displays the detection results in a formatted manner."""
    for key, value in data.items():
        if not value:
            continue

        print(f"{key}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not sub_value:
                    continue

                if isinstance(sub_value, dict):
                    print(f"  {sub_key}:")
                    for k, v in sub_value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {sub_key}: {sub_value}")
        else:
            print(f"  {value}")
        print()


def collect_results(detector: EnvironmentDetector) -> Dict[str, Any]:
    """Collects detection results into a dictionary."""
    system_info: Dict[str, Any] = {
        'Operating System': {},
        'Virtual Machine': {},
        'Sandbox': {}
    }

    if detector.os_info:
        system_info['Operating System']['Type'] = detector.os_info.type.value
        system_info['Operating System']['Version'] = detector.os_info.version
        system_info['Operating System']['Architecture'] = detector.os_info.architecture
        system_info['Operating System']['Additional Info'] = detector.os_info.additional_info

    if detector.vm_info:
        system_info['Virtual Machine']['Type'] = detector.vm_info.type.value
        system_info['Virtual Machine']['Version'] = detector.vm_info.version
        system_info['Virtual Machine']['Additional Info'] = detector.vm_info.additional_info

    if detector.sandbox_info:
        system_info['Sandbox']['Type'] = detector.sandbox_info.type.value
        system_info['Sandbox']['Version'] = detector.sandbox_info.version
        system_info['Sandbox']['Additional Info'] = detector.sandbox_info.additional_info

    return system_info


def run_detection(args: argparse.Namespace) -> EnvironmentDetector:
    """Runs the detection process."""
    detector = EnvironmentDetector(perform_all_checks=args.all, paranoid=args.paranoid)
    detector.detect()
    return detector


def main():
    """Main function to orchestrate the system detection."""
    args = parse_arguments()
    setup_logging(
        verbose=args.verbose,
        debug=args.debug
    )
    detector = run_detection(args)
    system_info = collect_results(detector)
    display_results(system_info)
    if args.output and not save_output(system_info, args.output):
            logging.error("Failed to save detection results.")
            sys.exit(1)
    logging.info("System detection completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
