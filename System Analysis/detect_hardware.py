#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_hardware.py
#
# Description:
# This script detects detailed hardware information about the system.
# It collects data about built-in hardware components such as CPU, GPU,
# memory, storage, audio devices, and also external devices like USB devices.
#
# Usage:
# ./detect_hardware.py [options]
#
# Options:
# -v, --verbose               Enable verbose logging (INFO level).
# -vv, --debug                Enable debug logging (DEBUG level).
# -o, --output FILE           Output the detection results to a specified file (JSON format).
# -h, --help                  Show help message and exit.
#
# Template: ubuntu22.04
#
# Requirements:
# - Linux:
#   - pciutils (install via: apt install pciutils)
#   - usbutils (install via: apt install usbutils)
#   - dmidecode (install via: apt install dmidecode)
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
import json
import platform
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Constants for file paths and commands
LSPCI_CMD = 'lspci'
LSUSB_CMD = 'lsusb'
DMIDECODE_CMD = 'dmidecode'


class OperatingSystemType(Enum):
    """Enum representing supported operating system types."""
    WINDOWS = 'Windows'
    LINUX = 'Linux'
    UNKNOWN = 'Unknown'


@dataclass
class CPUInfo:
    """Data class for storing CPU information."""
    model: str = ''
    cores: int = 0
    threads: int = 0
    architecture: str = ''
    frequency: str = ''
    flags: List[str] = field(default_factory=list)
    cache_size: str = ''


@dataclass
class MemoryInfo:
    """Data class for storing Memory information."""
    total: str = ''
    slots: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StorageDeviceInfo:
    """Data class for storing Storage Device information."""
    model: str = ''
    vendor: str = ''
    size: str = ''
    serial: str = ''


@dataclass
class USBDeviceInfo:
    """Data class for storing USB Device information."""
    bus: str = ''
    device_id: str = ''
    vendor_id: str = ''
    vendor: str = ''
    product_id: str = ''
    product: str = ''


@dataclass
class PciDeviceInfo:
    """Data class for storing PCI Device information."""
    class_name: str
    vendor: str
    device: str


@dataclass
class HardwareInfo:
    """Data class for storing all hardware information."""
    cpu: Optional[CPUInfo] = None
    memory: Optional[MemoryInfo] = None
    storage: List[StorageDeviceInfo] = field(default_factory=list)
    usb_devices: Dict[str, List[USBDeviceInfo]] = field(default_factory=dict)
    pci_devices: Dict[str, List[PciDeviceInfo]] = field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract base class for hardware detectors."""

    @abstractmethod
    def detect(self) -> Optional[Any]:
        pass


class LinuxCPUDetector(BaseDetector):
    """Detector for CPU information on Linux."""

    def detect(self) -> Optional[CPUInfo]:
        """Detect CPU information on Linux."""
        cpu_info = CPUInfo()
        try:
            with open('/proc/cpuinfo', 'r', encoding='utf-8') as f:
                cpuinfo = f.read()
            lines = cpuinfo.split('\n')
            model_name = ''
            cpu_cores = 0
            siblings = 0
            architecture = platform.machine()
            frequency = ''
            flags = []
            cache_size = ''
            for line in lines:
                if 'model name' in line:
                    model_name = line.split(':', 1)[1].strip()
                elif 'cpu cores' in line:
                    cpu_cores = int(line.split(':', 1)[1].strip())
                elif 'siblings' in line:
                    siblings = int(line.split(':', 1)[1].strip())
                elif 'cpu MHz' in line:
                    frequency = line.split(':', 1)[1].strip() + ' MHz'
                elif 'flags' in line:
                    flags = line.split(':', 1)[1].strip().split()
                elif 'cache size' in line:
                    cache_size = line.split(':', 1)[1].strip()
            cpu_info.model = model_name
            cpu_info.cores = cpu_cores
            cpu_info.threads = siblings
            cpu_info.architecture = architecture
            cpu_info.frequency = frequency
            cpu_info.flags = flags
            cpu_info.cache_size = cache_size
            logging.debug(f"Detected CPU info: {cpu_info}")
            return cpu_info
        except Exception as e:
            logging.error(f"Error detecting CPU info: {e}")
            return None


class LinuxMemoryDetector(BaseDetector):
    """Detector for Memory information on Linux."""

    def detect(self) -> Optional[MemoryInfo]:
        """Detect Memory information on Linux."""
        memory_info = MemoryInfo()
        try:
            with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                meminfo_output = f.read()
            lines = meminfo_output.strip().split('\n')
            total_mem = ''
            for line in lines:
                if 'MemTotal' in line:
                    total_mem = line.split(':', 1)[1].strip()
                    break
            memory_info.total = total_mem

            # For detailed slot information, root privileges and dmidecode are required
            if os.geteuid() != 0:
                logging.info("Root privileges required to detect detailed memory slot info.")
                return memory_info

            try:
                dmidecode_output = subprocess.check_output([DMIDECODE_CMD, '--type', '17'], text=True)
                slot_info = []
                slot = {}
                for line in dmidecode_output.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('Handle'):
                        if slot:
                            slot_info.append(slot)
                            slot = {}
                    elif line == '':
                        continue
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        slot[key.strip()] = value.strip()
                if slot:
                    slot_info.append(slot)
                memory_info.slots = slot_info
            except Exception as e:
                logging.error(f"Error detecting detailed memory slot info: {e}")
            logging.debug(f"Detected Memory info: {memory_info}")
            return memory_info
        except Exception as e:
            logging.error(f"Error detecting Memory info: {e}")
            return None


class LinuxStorageDetector(BaseDetector):
    """Detector for Storage Device information on Linux."""

    def detect(self) -> Optional[List[StorageDeviceInfo]]:
        """Detect Storage Device information on Linux using JSON output from lsblk."""
        try:
            # Execute lsblk with JSON output
            lsblk_output = subprocess.check_output(
                ['lsblk', '-J', '-o', 'NAME,MODEL,VENDOR,SIZE,TYPE,SERIAL'],
                text=True
            )
            lsblk_json = json.loads(lsblk_output)
            storage_devices = []

            # Traverse the JSON structure to find devices of type 'disk'
            for device in lsblk_json.get('blockdevices', []):
                if device.get('type') == 'disk':
                    storage_device = StorageDeviceInfo(
                        model=(device.get('model') or '').strip(),
                        vendor=(device.get('vendor') or '').strip(),
                        size=(device.get('size') or '').strip(),
                        serial=(device.get('serial') or '').strip(),
                    )
                    storage_devices.append(storage_device)

            # Sort storage devices alphabetically by vendor then model
            storage_devices.sort(key=lambda x: (x.vendor.lower(), x.model.lower()))
            logging.debug(f"Detected Storage Devices: {storage_devices}")
            return storage_devices
        except subprocess.CalledProcessError as e:
            logging.error(f"lsblk command failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing lsblk JSON output: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error detecting Storage Devices: {e}")
            return None


class LinuxUSBDetector(BaseDetector):
    """Detector for USB Device information on Linux."""

    def detect(self) -> Optional[Dict[str, List[USBDeviceInfo]]]:
        """Detect USB Device information on Linux grouped by vendor."""
        try:
            lsusb_output = subprocess.check_output([LSUSB_CMD], text=True)
            lines = lsusb_output.strip().split('\n')
            usb_devices = []
            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    continue
                bus = parts[1]
                device = parts[3].strip(':')
                ids = parts[5]
                if ':' in ids:
                    vendor_id, product_id = ids.split(':', 1)
                else:
                    vendor_id, product_id = '', ''
                vendor_product = ' '.join(parts[6:])
                # Split vendor and product if possible
                vendor_product_split = vendor_product.split(' ', 1)
                vendor = vendor_product_split[0].strip()
                product = vendor_product_split[1].strip() if len(vendor_product_split) > 1 else ''
                usb_device = USBDeviceInfo(
                    bus=bus,
                    device_id=device,
                    vendor_id=vendor_id,
                    product_id=product_id,
                    vendor=vendor,
                    product=product,
                )
                usb_devices.append(usb_device)
            # Group USB devices by vendor
            grouped_usb = {}
            for device in usb_devices:
                vendor_key = device.vendor if device.vendor else 'Unknown'
                if vendor_key not in grouped_usb:
                    grouped_usb[vendor_key] = []
                grouped_usb[vendor_key].append(device)
            # Sort the grouped_usb by vendor name
            sorted_grouped_usb = dict(sorted(grouped_usb.items(), key=lambda x: x[0].lower()))
            # Sort each device list alphabetically by product
            for devices in sorted_grouped_usb.values():
                devices.sort(key=lambda x: (x.product.lower()))
            logging.debug(f"Detected USB Devices: {sorted_grouped_usb}")
            return sorted_grouped_usb if sorted_grouped_usb else None
        except subprocess.CalledProcessError as e:
            logging.error(f"lsusb command failed: {e}")
            return None
        except Exception as e:
            logging.error(f"Error detecting USB Devices: {e}")
            return None


class LinuxPciDevicesDetector(BaseDetector):
    """Detector for PCI devices on Linux using descriptive class names."""

    CLASS_ALIAS_MAP: Dict[str, str] = {
        # Display Controllers
        "VGA compatible controller": "GPU",
        "3D controller": "GPU",
        "Display controller": "GPU",

        # Audio Controllers
        "Audio device": "Audio",
        "Multimedia audio controller": "Audio",

        # Network Controllers
        "Ethernet controller": "Network",
        "Network controller": "Network",
        "Wireless controller": "Network",
        "Other network controller": "Network",
        "Bluetooth controller": "Network",

        # Bridge Devices
        "Host bridge": "Bridge",
        "ISA bridge": "Bridge",
        "IDE interface": "Storage",
        "Bridge": "Bridge",
        "PCI bridge": "Bridge",

        # Storage Controllers
        "SATA controller": "Storage",
        "Mass storage controller": "Storage",
        "Non-volatile memory controller": "Storage",
        "SCSI storage controller": "Storage",
        "RAID bus controller": "Storage",
        "Other mass storage controller": "Storage",

        # Communication Controllers
        "Serial controller": "Serial",
        "USB controller": "USB",
        "FireWire controller": "FireWire",

        # Input Device Controllers
        "Input device controller": "Input",
        "Mouse controller": "Input",
        "Keyboard controller": "Input",
        "Touchpad controller": "Input",

        # Processors and Memory
        "Processor": "Processor",
        "Memory controller": "Memory",
        "RAM memory": "Memory",

        # Security
        "Encryption controller": "Security",
    }

    DEFAULT_CATEGORY = "Other"
    LSPCI_CMD = "lspci"

    def __init__(self):
        """Initialize the detector with a case-insensitive mapping."""
        self.CLASS_ALIAS_MAP_LOWER = {k.lower(): v for k, v in self.CLASS_ALIAS_MAP.items()}

    def detect(self) -> Optional[Dict[str, List[PciDeviceInfo]]]:
        """Detect PCI devices on Linux and group them by class."""
        try:
            lspci_data = self._get_lspci_data()
            if not lspci_data:
                logging.info("No PCI devices detected.")
                return None

            grouped_devices: Dict[str, List[PciDeviceInfo]] = {}
            for device in lspci_data:
                original_class = device.class_name.strip()
                category = self.CLASS_ALIAS_MAP_LOWER.get(original_class.lower(), self.DEFAULT_CATEGORY)

                if category not in grouped_devices:
                    grouped_devices[category] = []
                grouped_devices[category].append(device)

            # Sort the groups alphabetically
            sorted_grouped_devices = dict(sorted(grouped_devices.items(), key=lambda x: x[0].lower()))

            # Sort devices within each group by vendor and device name
            for devices in sorted_grouped_devices.values():
                devices.sort(key=lambda x: (x.vendor.lower(), x.device.lower()))

            logging.debug(f"Grouped PCI Devices: {sorted_grouped_devices}")
            return sorted_grouped_devices if sorted_grouped_devices else None

        except Exception as e:
            logging.error(f"Error detecting PCI Devices: {e}")
            return None

    def _get_lspci_data(self) -> Optional[List[PciDeviceInfo]]:
        """Parses lspci -mm output."""
        try:
            lspci_output = subprocess.check_output([self.LSPCI_CMD, '-mm'], text=True)
            lines = lspci_output.strip().split('\n')
            regex = re.compile(r'"([^"]*)"')
            parsed_data = []
            for line in lines:
                fields = regex.findall(line)
                if len(fields) >= 3:
                    class_name = fields[0]
                    vendor = fields[1]
                    device = fields[2]
                    device_info = PciDeviceInfo(
                        class_name=class_name,
                        vendor=vendor,
                        device=device,
                    )
                    parsed_data.append(device_info)
                else:
                    logging.warning(f"Unexpected lspci line format: {line}")
            return parsed_data
        except subprocess.CalledProcessError as e:
            logging.error(f"lspci command failed: {e}")
            return None
        except Exception as e:
            logging.error(f"Error parsing lspci output: {e}")
            return None


class DetectorManager:
    """Manager for registering and executing hardware detectors."""

    def __init__(self):
        self.detectors: List[BaseDetector] = []

    def register_detector(self, detector: BaseDetector):
        """Registers a detector."""
        self.detectors.append(detector)

    def detect_all(self) -> List[Any]:
        """Runs all registered detectors and collects their results."""
        results = []
        for detector in self.detectors:
            try:
                result = detector.detect()
                if result:
                    results.append((detector.__class__.__name__, result))
            except Exception as e:
                logging.error(f"Error running detector {detector.__class__.__name__}: {e}")
        return results


class HardwareDetector:
    """Main class for detecting hardware information."""

    def __init__(self):
        self.hardware_info = HardwareInfo()
        self.detector_manager = DetectorManager()
        self.os_type = self._detect_os_type()
        self._register_detectors()

    def _detect_os_type(self) -> OperatingSystemType:
        """Detects the operating system type."""
        system = platform.system().lower()
        if 'linux' in system:
            return OperatingSystemType.LINUX
        else:
            return OperatingSystemType.UNKNOWN

    def _register_detectors(self):
        """Registers detectors based on the operating system."""
        if self.os_type == OperatingSystemType.LINUX:
            self.detector_manager.register_detector(LinuxCPUDetector())
            self.detector_manager.register_detector(LinuxMemoryDetector())
            self.detector_manager.register_detector(LinuxStorageDetector())
            self.detector_manager.register_detector(LinuxUSBDetector())
            self.detector_manager.register_detector(LinuxPciDevicesDetector())
        else:
            logging.error("Unsupported operating system.")

    def detect(self):
        """Performs the hardware detection."""
        logging.info(f"Starting hardware detection on {self.os_type.value}")
        results = self.detector_manager.detect_all()
        for detector_name, result in results:
            if isinstance(result, CPUInfo):
                self.hardware_info.cpu = result
            elif isinstance(result, MemoryInfo):
                self.hardware_info.memory = result
            elif isinstance(result, list):
                if result and isinstance(result[0], StorageDeviceInfo):
                    self.hardware_info.storage.extend(result)
            elif isinstance(result, dict):
                if result and all(isinstance(v, list) and v and isinstance(v[0], PciDeviceInfo) for v in result.values()):
                    self.hardware_info.pci_devices.update(result)
                elif result and all(isinstance(v, list) and v and isinstance(v[0], USBDeviceInfo) for v in result.values()):
                    self.hardware_info.usb_devices.update(result)
        logging.info("Hardware detection completed.")


class HardwareInfoDisplay:
    def __init__(self, hardware_info: HardwareInfo) -> None:
        """Initialize the display class with hardware information."""
        self.hardware_info = hardware_info

    def display(self) -> None:
        """Displays the detection results in a formatted manner."""
        sections: List[tuple[str, Any]] = [
            ("CPU Information", self.hardware_info.cpu),
            ("Memory Information", self.hardware_info.memory),
            ("PCI Devices", self.hardware_info.pci_devices),
            ("Storage Devices", self.hardware_info.storage),
            ("USB Devices", self.hardware_info.usb_devices),
        ]

        for title, data in sections:
            if data:
                self._print_section(title, data)

    def _join_with_line_breaks(self, items: List[str], max_length: int, padding: str = "") -> str:
        """Join a list of strings with commas, inserting line breaks when the line length exceeds max_length."""
        if not items:
            return ""

        wrapped_lines = []
        current_line = padding
        for item in items:
            separator = ", " if current_line.strip() else ""
            potential_line = f"{current_line}{separator}{item}"

            if len(potential_line) > max_length:
                if current_line.strip():
                    wrapped_lines.append(current_line.rstrip())
                    current_line = f"{padding}{item}"
                else:
                    # If the single item is longer than max_length, add it as is
                    wrapped_lines.append(f"{padding}{item}")
                    current_line = padding
            else:
                current_line = potential_line

        if current_line.strip():
            wrapped_lines.append(current_line)

        return "\n".join(wrapped_lines)

    def _print_sorted_attributes(self, obj: Any, indent: str = "  ") -> None:
        """Print sorted non-empty attributes of an object with optional indentation."""
        for key, value in sorted(obj.__dict__.items()):
            if value:
                if isinstance(value, list):
                    print(f"{indent}{key}:")
                    formatted_list = self._join_with_line_breaks(
                        [str(item) for item in value],
                        max_length=100,
                        padding=indent + "    "
                    )
                    print(formatted_list)
                else:
                    print(f"{indent}{key}: {value}")
        print()

    def _print_section(self, title: str, data: Any) -> None:
        """Print a section title and its associated data."""
        print(f"{title}:")
        if isinstance(data, list):
            for item in data:
                self._print_sorted_attributes(item, indent="  ")
        elif isinstance(data, dict):
            for class_name in sorted(data.keys(), key=lambda x: x.lower()):
                print(f"  {class_name}:")
                for device in data[class_name]:
                    self._print_sorted_attributes(device, indent="    ")
        elif hasattr(data, '__dict__'):
            self._print_sorted_attributes(data, indent="  ")
        print()


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect hardware information about the system.",
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
            json.dump(data, f, indent=4, sort_keys=True)
        logging.info(f"Detection results saved to '{filepath}'.")
        return True
    except Exception as e:
        logging.error(f"Error saving detection results: {e}")
        return False


def collect_results(hardware_info: HardwareInfo) -> Dict[str, Any]:
    """Collects hardware information into a dictionary."""
    # Sort storage devices by vendor then model
    sorted_storage = sorted(
        [storage.__dict__ for storage in hardware_info.storage],
        key=lambda x: (x.get('vendor', '').lower(), x.get('model', '').lower())
    )

    # Sort USB devices by vendor then product
    sorted_usb = {}
    for vendor in sorted(hardware_info.usb_devices.keys(), key=lambda x: x.lower()):
        devices = hardware_info.usb_devices[vendor]
        sorted_devices = sorted(
            [usb.__dict__ for usb in devices],
            key=lambda x: x.get('product', '').lower()
        )
        sorted_usb[vendor] = sorted_devices

    # Sort PCI devices by class name and then by vendor and device
    sorted_pci = {}
    for class_name in sorted(hardware_info.pci_devices.keys(), key=lambda x: x.lower()):
        sorted_devices = sorted(
            [device.__dict__ for device in hardware_info.pci_devices[class_name]],
            key=lambda x: (x.get('vendor', '').lower(), x.get('device', '').lower())
        )
        sorted_pci[class_name] = sorted_devices

    return {
        'CPU': dict(sorted(hardware_info.cpu.__dict__.items())) if hardware_info.cpu else {},
        'Memory': dict(sorted(hardware_info.memory.__dict__.items())) if hardware_info.memory else {},
        'Storage': sorted_storage,
        'USBDevices': sorted_usb,
        'PciDevices': sorted_pci,
    }


def main():
    """Main function to orchestrate the hardware detection."""
    args = parse_arguments()
    setup_logging(
        verbose=args.verbose,
        debug=args.debug
    )
    detector = HardwareDetector()
    detector.detect()
    HardwareInfoDisplay(detector.hardware_info).display()
    if args.output:
        hardware_data = collect_results(detector.hardware_info)
        if not save_output(hardware_data, args.output):
            logging.error("Failed to save detection results.")
            sys.exit(1)
    logging.info("Hardware detection completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
