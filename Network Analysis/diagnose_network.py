#!/usr/bin/env python3

# -------------------------------------------------------
# Script: diagnose_network.py
#
# Description:
# This script provides comprehensive network diagnostics to
# automatically troubleshoot network issues.
# It includes functionalities such as system information gathering,
# device discovery, automated diagnostics, rogue device detection,
# and advanced network traffic monitoring.
#
# Usage:
# ./diagnose_network.py [command] [options]
#
# Commands:
# - system-info (si)          Display detailed network information about the system.
# - diagnose (dg)             Perform automated diagnostics on the network.
# - rogue-detection (rd)      Detect unauthorized or rogue devices on the network.
# - traffic-monitor (tm)      Monitor network traffic to detect anomalies using Scapy.
# - wifi (wf)                 Perform WiFi diagnostics and analyze available networks.
#
# Global Options:
# -v, --verbose               Enable verbose logging (INFO level).
# -vv, --debug                Enable debug logging (DEBUG level).
# -d, --detailed              Show detailed output where applicable.
#
# Traffic Monitor Command Options:
#   --interface, -i           Specify the network interface to monitor (e.g., wlan0, eth0).
#
# WiFi Command Options:
#   --ssid, -s                Specify the SSID to perform targeted diagnostics.
#                             If not specified, performs generic WiFi checks.
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
# - scapy (install via: pip install scapy)
# - requests (install via: pip install requests)
# - nmap (install via: apt install nmap)
# - rich (optional, for enhanced terminal outputs) (install via: pip install rich)
# - python-dotenv (optional, for loading .env files) (install via: pip install python-dotenv)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import subprocess
import socket
import json
import time
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Attempt to import rich for enhanced terminal outputs
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.logging import RichHandler
except ImportError:
    Console = None

# Attempt to import scapy for advanced network monitoring
try:
    from scapy.all import sniff, ARP, Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Attempt to import requests for HTTP requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Initialize Rich console if available
console = Console() if Console else None


# Base class for all commands
class BaseCommand(ABC):
    def __init__(self, args, logger, detailed: bool):
        """
        Initialize the BaseCommand with arguments, logger, and detailed flag.
        """
        self.args = args
        self.logger = logger
        self.detailed = detailed

    @abstractmethod
    def execute(self):
        """
        Execute the command.
        """
        pass

    def print_table(self, title: str, columns: List[str], rows: List[List[str]]):
        """
        Print a table with the given title, columns, and rows.
        """
        if console:
            table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
            for col in columns:
                table.add_column(col, style="cyan", no_wrap=True)
            for row in rows:
                table.add_row(*row)
            console.print(table)
        else:
            print(f"\n{title}")
            print("-" * len(title))
            print("\t".join(columns))
            for row in rows:
                print("\t".join(row))
            print()


# MAC Vendor Lookup Class
class MacVendorLookup:
    """
    Lookup MAC address vendors using OUI data.
    """
    OUI_URL = "http://standards-oui.ieee.org/oui/oui.json"

    def __init__(self, logger: logging.Logger):
        """
        Initialize the MacVendorLookup with a logger.
        """
        self.logger = logger
        self.oui_dict = self.load_oui_data()

    def load_oui_data(self) -> Dict[str, str]:
        """
        Load OUI data from a local file or download it.
        """
        if os.path.exists("oui.json"):
            self.logger.debug("Loading OUI data from local 'oui.json' file.")
            try:
                with open("oui.json", "r") as f:
                    data = json.load(f)
                return data
            except Exception as e:
                self.logger.error(f"Failed to load local OUI data: {e}")
        # Download OUI data
        self.logger.debug("Downloading OUI data from IEEE.")
        try:
            response = requests.get(self.OUI_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                with open("oui.json", "w") as f:
                    json.dump(data, f)
                self.logger.debug("OUI data downloaded and saved locally.")
                return data
            else:
                self.logger.error(f"Failed to download OUI data: HTTP {response.status_code}")
        except Exception as e:
            self.logger.error(f"Exception while downloading OUI data: {e}")
        return {}

    def get_vendor(self, mac: str) -> str:
        """
        Get the vendor name for a given MAC address.
        """
        mac_prefix = mac.upper().replace(":", "").replace("-", "").replace(".", "")[:6]
        oui_entry = self.oui_dict.get(mac_prefix, {})
        return oui_entry.get("company", "Unknown")


# Web Interface Analyzer Class
class WebInterfaceAnalyzer:
    """
    Analyze web interfaces of devices.
    """
    def __init__(self, logger: logging.Logger):
        """
            Initialize the WebInterfaceAnalyzer with a logger.
        """
        self.logger = logger

    def analyze(self, ip: str, port: int) -> Optional[str]:
        """
        Analyze the web interface at the given IP and port.
        """
        if not REQUESTS_AVAILABLE:
            self.logger.warning("Requests library not available. Skipping web interface analysis.")
            return None
        url = f"http://{ip}:{port}" if port == 80 else f"https://{ip}:{port}"
        try:
            response = requests.get(url, timeout=5, verify=False)
            server = response.headers.get("Server", "Unknown")
            self.logger.debug(f"Web interface at {url} responded with Server: {server}")
            return server
        except requests.RequestException as e:
            self.logger.debug(f"Failed to connect to web interface at {url}: {e}")
            return None


# Device Diagnostics Base Class
class DeviceDiagnostics(ABC):
    """
    Abstract base class for device diagnostics.
    """
    def __init__(self, device: Dict[str, str], logger: logging.Logger, detailed: bool):
        """
            Initialize the DeviceDiagnostics with device info, logger, and detailed flag.
        """
        self.device = device
        self.logger = logger
        self.detailed = detailed

    @abstractmethod
    def diagnose(self) -> Optional[List[str]]:
        """
        Perform diagnostics on the device.
        """
        pass

    def check_port(self, ip: str, port: int) -> bool:
        """
        Check if a specific port is open on the given IP.
        """
        try:
            with socket.create_connection((ip, port), timeout=2):
                return True
        except Exception:
            return False


# Router Diagnostics
class RouterDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to routers.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the router for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug("Router IP is N/A. Skipping diagnostics.")
            return issues

        # Check if SSH port 22 is open
        if not self.check_port(ip, 22):
            issues.append(f"Router {hostname} ({ip}): SSH Port 22 Unreachable")

        # Check if web interface port 80 or 443 is open
        web_ports = [80, 443]
        if not any(self.check_port(ip, port) for port in web_ports):
            issues.append(f"Router {hostname} ({ip}): Web Interface Ports 80/443 Unreachable")

        # Verify web interface details
        analyzer = WebInterfaceAnalyzer(self.logger)
        for port in web_ports:
            if self.check_port(ip, port):
                server_info = analyzer.analyze(ip, port)
                if server_info and "router" not in server_info.lower():
                    issues.append(f"Router {hostname} ({ip}): Unexpected Server Info '{server_info}' on Port {port}")

        return issues


# Printer Diagnostics
class PrinterDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to printers.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the printer for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug("Printer IP is N/A. Skipping diagnostics.")
            return issues

        # Check if port 9100 is open (common for network printers)
        if not self.check_port(ip, 9100):
            issues.append(f"Printer {hostname} ({ip}): Port 9100 Unreachable")

        # Optionally, verify printer status via SNMP or other protocols

        return issues


# Payment Terminal Diagnostics
class PaymentTerminalDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to payment terminals.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the payment terminal for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug("Payment Terminal IP is N/A. Skipping diagnostics.")
            return issues

        # Check if port 8443 is open (assuming API is exposed on this port)
        if not self.check_port(ip, 8443):
            issues.append(f"Payment Terminal {hostname} ({ip}): Port 8443 Unreachable")

        # Optionally, verify payment processing services

        return issues


# Web-Enabled Device Diagnostics
class WebEnabledDeviceDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to web-enabled devices.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the web-enabled device for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug("Web-Enabled Device IP is N/A. Skipping diagnostics.")
            return issues

        # Check if HTTP (80) is open
        http_open = self.check_port(ip, 80)
        # Check if HTTPS (443) is open
        https_open = self.check_port(ip, 443)

        if not http_open and not https_open:
            issues.append(f"Web-Enabled Device {hostname} ({ip}): Neither HTTP nor HTTPS Ports are Open")

        # Verify web server details
        analyzer = WebInterfaceAnalyzer(self.logger)
        for port in [80, 443]:
            if self.check_port(ip, port):
                server_info = analyzer.analyze(ip, port)
                if server_info and "apache" not in server_info.lower() and "nginx" not in server_info.lower():
                    issues.append(f"Web-Enabled Device {hostname} ({ip}): Unexpected Server Info '{server_info}' on Port {port}")

        return issues


# Laptop Diagnostics
class LaptopDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to laptops.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the laptop for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")
        os_info = self.device.get("OS", "Unknown").lower()

        if ip == "N/A":
            self.logger.debug("Laptop IP is N/A. Skipping diagnostics.")
            return issues

        # Check if SSH port 22 is open (common for remote access)
        if "linux" in os_info or "unix" in os_info:
            if not self.check_port(ip, 22):
                issues.append(f"Laptop {hostname} ({ip}): SSH Port 22 Unreachable")

        # Check for common open ports like RDP (3389) for Windows laptops
        if "windows" in os_info:
            if not self.check_port(ip, 3389):
                issues.append(f"Laptop {hostname} ({ip}): RDP Port 3389 Unreachable")

        # Verify if antivirus is running (placeholder for actual implementation)
        # This would require more advanced checks, potentially via SNMP or other protocols

        return issues


# Phone Diagnostics
class PhoneDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to phones.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the phone for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")
        os_info = self.device.get("OS", "Unknown").lower()

        if ip == "N/A":
            self.logger.debug("Phone IP is N/A. Skipping diagnostics.")
            return issues

        # Check for ports commonly used by mobile devices
        # Example: Port 5555 for Android Debug Bridge (ADB)
        if "android" in os_info:
            if not self.check_port(ip, 5555):
                issues.append(f"Phone {hostname} ({ip}): ADB Port 5555 Unreachable")

        # For iOS devices, check for ports related to Apple services if applicable
        # Placeholder for actual implementation

        # Verify if device is in expected state (placeholder for actual implementation)
        # This would require more advanced checks

        return issues


# Computer Diagnostics
class ComputerDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics specific to desktop computers.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose the desktop computer for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")
        os_info = self.device.get("OS", "Unknown").lower()

        if ip == "N/A":
            self.logger.debug("Computer IP is N/A. Skipping diagnostics.")
            return issues

        # Check if SSH port 22 is open (common for remote management)
        if "linux" in os_info or "unix" in os_info:
            if not self.check_port(ip, 22):
                issues.append(f"Computer {hostname} ({ip}): SSH Port 22 Unreachable")

        # Check if RDP port 3389 is open for Windows computers
        if "windows" in os_info:
            if not self.check_port(ip, 3389):
                issues.append(f"Computer {hostname} ({ip}): RDP Port 3389 Unreachable")

        # Check for file sharing ports (e.g., SMB on port 445)
        if not self.check_port(ip, 445):
            issues.append(f"Computer {hostname} ({ip}): SMB Port 445 Unreachable")

        # Verify if antivirus is running (placeholder for actual implementation)
        # This would require more advanced checks, potentially via SNMP or other protocols

        return issues


# Other Device Diagnostics
class OtherDeviceDiagnostics(DeviceDiagnostics):
    """
    Perform diagnostics for other types of devices.
    """
    def diagnose(self) -> Optional[List[str]]:
        """
        Diagnose other devices for reachability.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug("Other Device IP is N/A. Skipping diagnostics.")
            return issues

        # Generic diagnostic: Check if device is reachable
        status = self.ping_device(ip)
        if status != "Reachable":
            issues.append(f"Device {hostname} ({ip}): {status}")

        return issues

    def ping_device(self, ip: str) -> str:
        """
        Ping the device to check its reachability.
        """
        try:
            # Ping once with a timeout of 2 seconds
            result = subprocess.run(['ping', '-c', '1', '-W', '2', ip],
                                    stdout=subprocess.DEVNULL)
            if result.returncode == 0:
                return "Reachable"
            else:
                return "Unreachable"
        except Exception as e:
            self.logger.error(f"Error pinging {ip}: {e}")
            return "Error"


# Network Scanner and Classifier
class NetworkScannerCommand(BaseCommand):
    """
    Scan the network and classify connected devices.
    """
    def execute(self):
        """
        Execute the network scanning and classification.
        """
        self.logger.info("Scanning the network for connected devices...")
        devices = self.scan_network()
        if not devices:
            self.logger.error("No devices found on the network.")
            sys.exit(1)

        classified_devices = self.classify_devices(devices)
        self.display_devices(classified_devices)

        # Save devices to a JSON file if detailed output is requested
        if self.detailed:
            self.save_devices_to_file(classified_devices, "discovered_devices.json")

    def scan_network(self) -> List[Dict[str, str]]:
        """
        Scan the active subnet using nmap to discover devices.
        """
        try:
            # Dynamically determine the active subnet
            subnet = self.get_active_subnet()
            self.logger.debug(f"Active subnet detected: {subnet}")

            # Use nmap for comprehensive scanning with OS detection and service enumeration
            # -O: Enable OS detection
            # -sV: Probe open ports to determine service/version info
            # -T4: Faster execution
            # -oJ - : Output in JSON format to stdout
            self.logger.debug("Initiating nmap scan...")
            scan_command = ['nmap', '-O', '-sV', '-T4', '-oJ', '-', subnet]
            result = subprocess.run(scan_command, capture_output=True, text=True, check=True)
            devices = self.parse_nmap_output(result.stdout)
            self.logger.debug(f"Found {len(devices)} devices on the network.")
            return devices
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to scan network with nmap: {e}")
            return []
        except FileNotFoundError:
            self.logger.error("nmap is not installed. Install it using your package manager.")
            return []

    def get_active_subnet(self) -> str:
        """
        Determine the active subnet based on the system's local IP.
        """
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            subnet = '.'.join(local_ip.split('.')[:3]) + '.0/24'
            return subnet
        except Exception as e:
            self.logger.error(f"Failed to determine active subnet: {e}")
            sys.exit(1)

    def parse_nmap_output(self, output: str) -> List[Dict[str, str]]:
        """
        Parse the JSON output from nmap to extract device information.
        """
        devices = []
        try:
            nmap_data = json.loads(output)
            for host in nmap_data.get('hosts', []):
                if host.get('status', {}).get('state') != 'up':
                    continue
                device = {}
                device['IP'] = host.get('addresses', {}).get('ipv4', 'N/A')
                device['MAC'] = host.get('addresses', {}).get('mac', 'N/A')
                device['Vendor'] = host.get('vendor', {}).get(device['MAC'], 'Unknown') if device['MAC'] != 'N/A' else 'Unknown'
                hostnames = host.get('hostnames', [])
                device['Hostname'] = hostnames[0].get('name') if hostnames else "N/A"
                os_matches = host.get('os', {}).get('osmatches', [])
                device['OS'] = os_matches[0].get('name') if os_matches else "Unknown"
                ports = []
                for port in host.get('ports', []):
                    if port.get('state') == 'open':
                        service = port.get('service', {}).get('name', 'unknown')
                        port_info = f"{port.get('portid')}/{port.get('protocol')} {service}"
                        ports.append(port_info)
                device['Ports'] = ports
                devices.append(device)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse nmap JSON output: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during nmap parsing: {e}")
        return devices

    def classify_devices(self, devices: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Classify devices into different categories.
        """
        classifier = DeviceClassifier(self.logger)
        classified = classifier.classify(devices)
        return classified

    def display_devices(self, classified_devices: Dict[str, List[Dict[str, str]]]):
        """
        Display the classified devices in a tabular format.
        """
        for device_type, devices in classified_devices.items():
            title = f"{device_type}s"
            columns = ["Hostname", "IP Address", "MAC Address", "Vendor", "OS", "Open Ports"]
            rows = []
            for device in devices:
                hostname = device.get("Hostname", "N/A")
                ip = device.get("IP", "N/A")
                mac = device.get("MAC", "N/A")
                vendor = device.get("Vendor", "Unknown")
                os_info = device.get("OS", "Unknown")
                open_ports = ", ".join(device.get("Ports", []))
                rows.append([hostname, ip, mac, vendor, os_info, open_ports])
            self.print_table(title, columns, rows)

    def save_devices_to_file(self, classified_devices: Dict[str, List[Dict[str, str]]], filename: str):
        """
        Save the classified devices to a JSON file.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(classified_devices, f, indent=4)
            self.logger.info(f"Discovered devices saved to '{filename}'.")
        except Exception as e:
            self.logger.error(f"Failed to save devices to file: {e}")


# Device Classifier Class
class DeviceClassifier:
    """
    Classify devices based on their attributes.
    """
    def __init__(self, logger: logging.Logger):
        """
            Initialize the DeviceClassifier with a logger.
        """
        self.logger = logger
        self.mac_lookup = MacVendorLookup(logger)
        self.web_analyzer = WebInterfaceAnalyzer(logger)

    def classify(self, devices: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Classify the list of devices into categories.
        """
        classified = {}
        for device in devices:
            device_type = self.infer_device_type(device)
            if device_type not in classified:
                classified[device_type] = []
            classified[device_type].append(device)
        return classified

    def infer_device_type(self, device: Dict[str, str]) -> str:
        """
        Infer the device type based on its attributes.
        """
        # Infer device type based on open ports, OS information, and MAC vendor
        ports = device.get("Ports", [])
        os_info = device.get("OS", "").lower()
        mac = device.get("MAC", "N/A")
        vendor = self.mac_lookup.get_vendor(mac) if mac != 'N/A' else "Unknown"

        # Heuristic rules for classification
        if any("9100/tcp" in port for port in ports):
            return "Printer"
        elif any("8443/tcp" in port for port in ports):
            return "Payment Terminal"
        elif "router" in os_info or any("routing" in service.lower() for service in [p.split()[2] for p in ports]):
            return "Router"
        elif any(port.startswith("80/tcp") or port.startswith("443/tcp") for port in ports):
            return "Web-Enabled Device"
        elif "linux" in os_info and any(port.startswith("22/tcp") for port in ports):
            return "Linux Server"
        elif "windows" in os_info and any(port.startswith("3389/tcp") for port in ports):
            return "Windows Server"
        elif vendor.lower() in ['linksys', 'netgear', 'tp-link', 'd-link', 'asus']:
            return "Router"  # Additional vendor-based classification
        elif "laptop" in os_info or any(port.startswith("22/tcp") for port in ports):
            return "Laptop"
        elif "android" in os_info or "ios" in os_info:
            return "Phone"
        elif "windows" in os_info or "macos" in os_info:
            return "Desktop Computer"
        else:
            return "Other"


# Diagnostics Command
class DiagnosticsCommand(BaseCommand):
    """
    Perform automated network diagnostics.
    """
    def execute(self):
        """
        Execute the network diagnostics.
        """
        self.logger.info("Starting automated network diagnostics...")
        # Scan the network
        scanner = NetworkScannerCommand(self.args, self.logger, self.detailed)
        scanner.execute()

        # Load discovered devices
        try:
            with open("discovered_devices.json", 'r') as f:
                classified_devices = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load discovered devices: {e}")
            sys.exit(1)

        # Perform diagnostics based on device type
        self.perform_diagnostics(classified_devices)

    def perform_diagnostics(self, classified_devices: Dict[str, List[Dict[str, str]]]):
        """
        Perform diagnostics on the classified devices and collect issues.
        """
        issues_found = []

        for device_type, devices in classified_devices.items():
            for device in devices:
                diagnostic = self.get_diagnostic_class(device_type, device)
                if diagnostic:
                    issues = diagnostic.diagnose()
                    if issues:
                        for issue in issues:
                            # Split issue message to extract details
                            parts = issue.split(": ", 1)
                            if len(parts) == 2:
                                device_info, issue_desc = parts
                                device_info_parts = device_info.split(" ", 1)
                                if len(device_info_parts) == 2:
                                    dtype, details = device_info_parts
                                    hostname_ip = details.split(" (")
                                    if len(hostname_ip) == 2:
                                        hostname = hostname_ip[0]
                                        ip = hostname_ip[1].rstrip(")")
                                        issues_found.append([dtype, hostname, ip, issue_desc])

        # Display issues found
        if issues_found:
            columns = ["Device Type", "Hostname", "IP Address", "Issue"]
            self.print_table("Diagnostics Issues", columns, issues_found)
        else:
            self.logger.info("No issues detected during diagnostics.")

    def get_diagnostic_class(self, device_type: str, device: Dict[str, str]):
        """
        Get the appropriate diagnostic class based on device type.
        """
        if device_type == "Router":
            return RouterDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Printer":
            return PrinterDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Payment Terminal":
            return PaymentTerminalDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Web-Enabled Device":
            return WebEnabledDeviceDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Laptop":
            return LaptopDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Phone":
            return PhoneDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Desktop Computer":
            return ComputerDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Linux Server":
            return OtherDeviceDiagnostics(device, self.logger, self.detailed)
        elif device_type == "Windows Server":
            return OtherDeviceDiagnostics(device, self.logger, self.detailed)
        else:
            return OtherDeviceDiagnostics(device, self.logger, self.detailed)


# Rogue Device Detection Command
class RogueDeviceDetectionCommand(BaseCommand):
    """
    Detect unauthorized or rogue devices on the network.
    """
    def execute(self):
        """
        Execute rogue device detection.
        """
        self.logger.info("Starting rogue device detection...")
        # Scan the network
        scanner = NetworkScannerCommand(self.args, self.logger, self.detailed)
        scanner.execute()

        # Load discovered devices
        try:
            with open("discovered_devices.json", 'r') as f:
                classified_devices = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load discovered devices: {e}")
            sys.exit(1)

        # Detect rogue devices
        rogue_devices = self.detect_rogue_devices(classified_devices)

        # Display rogue devices
        if rogue_devices:
            columns = ["Hostname", "IP Address", "MAC Address", "Vendor", "OS", "Open Ports"]
            rows = []
            for device in rogue_devices:
                hostname = device.get("Hostname", "N/A")
                ip = device.get("IP", "N/A")
                mac = device.get("MAC", "N/A")
                vendor = device.get("Vendor", "Unknown")
                os_info = device.get("OS", "Unknown")
                open_ports = ", ".join(device.get("Ports", []))
                rows.append([hostname, ip, mac, vendor, os_info, open_ports])
            self.print_table("Rogue Devices Detected", columns, rows)
        else:
            self.logger.info("No rogue devices detected on the network.")

    def detect_rogue_devices(self, classified_devices: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Identify rogue devices based on classification.
        """
        rogue = []
        known_device_types = [
            "Router",
            "Printer",
            "Payment Terminal",
            "Web-Enabled Device",
            "Linux Server",
            "Windows Server",
            "Laptop",
            "Phone",
            "Desktop Computer"
        ]

        for device_type, devices in classified_devices.items():
            if device_type not in known_device_types:
                rogue.extend(devices)
            else:
                # Further logic can be implemented here to refine rogue detection
                # For example, based on unusual open ports or unexpected OS
                for device in devices:
                    if device_type == "Web-Enabled Device" and not self.validate_web_device(device):
                        rogue.append(device)
        return rogue

    def validate_web_device(self, device: Dict[str, str]) -> bool:
        """
        Validate web-enabled devices to ensure they meet expected criteria.
        """
        # Implement validation logic for web-enabled devices
        # For example, ensure that only expected services are running
        open_ports = device.get("Ports", [])
        expected_services = ["http", "https"]
        for port in open_ports:
            service = port.split(" ")[-1].lower()
            if service not in expected_services:
                return False
        return True


# Traffic Monitor Command
class TrafficMonitorCommand(BaseCommand):
    """
    Monitor network traffic to detect anomalies.
    """
    def execute(self):
        """
        Execute traffic monitoring on the specified interface.
        """
        if not SCAPY_AVAILABLE:
            self.logger.error("Scapy is not installed. Install it using 'pip install scapy'.")
            sys.exit(1)

        interface = self.args.interface
        if not interface:
            self.logger.error("Network interface not specified. Use --interface to specify one.")
            sys.exit(1)

        self.logger.info(f"Starting traffic monitoring on interface {interface}... (Press Ctrl+C to stop)")

        try:
            sniff(iface=interface, prn=self.process_packet, store=False)
        except PermissionError:
            self.logger.error("Permission denied. Run the script with elevated privileges.")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.info("Traffic monitoring stopped by user.")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Error during traffic monitoring: {e}")
            sys.exit(1)

    def process_packet(self, packet):
        """
        Process each captured packet to detect anomalies like ARP spoofing.
        """
        # Example: Detect ARP spoofing by monitoring ARP replies
        if packet.haslayer(ARP):
            arp = packet.getlayer(ARP)
            if arp.op == 2:  # is-at (response)
                info = f"ARP Reply: {arp.psrc} is-at {arp.hwsrc}"
                if self.detailed:
                    if console:
                        console.print(info, style="bold green")
                    else:
                        print(info)
                else:
                    self.logger.debug(info)


# System Information Command
class SystemInfoCommand(BaseCommand):
    """
    Gather and display system network information.
    """
    def execute(self):
        """
        Execute the system information gathering.
        """
        self.logger.info("Gathering system network information...")
        ip_info = self.get_ip_info()
        routing_info = self.get_routing_info()
        dns_info = self.get_dns_info()
        self.display_system_info(ip_info, routing_info, dns_info)

    def get_ip_info(self) -> str:
        """
        Retrieve IP configuration using the 'ip addr' command.
        """
        self.logger.debug("Retrieving IP configuration...")
        try:
            result = subprocess.run(['ip', 'addr'], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get IP information: {e}")
            return ""

    def get_routing_info(self) -> str:
        """
        Retrieve routing table using the 'ip route' command.
        """
        self.logger.debug("Retrieving routing table...")
        try:
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get routing information: {e}")
            return ""

    def get_dns_info(self) -> List[str]:
        """
        Retrieve DNS server information from /etc/resolv.conf.
        """
        self.logger.debug("Retrieving DNS servers...")
        dns_servers = []
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        dns_servers.append(line.split()[1])
        except Exception as e:
            self.logger.error(f"Failed to read DNS information: {e}")
        return dns_servers

    def display_system_info(self, ip_info: str, routing_info: str, dns_info: List[str]):
        """
        Display the gathered system information.
        """
        if self.detailed and console:
            console.print(Panel("[bold underline]IP Configuration[/bold underline]", style="cyan"))
            console.print(ip_info)
            console.print(Panel("[bold underline]Routing Table[/bold underline]", style="cyan"))
            console.print(routing_info)
            console.print(Panel("[bold underline]DNS Servers[/bold underline]", style="cyan"))
            for dns in dns_info:
                console.print(f"- {dns}")
        elif self.detailed:
            # Detailed output without rich
            print("\n=== IP Configuration ===")
            print(ip_info)
            print("\n=== Routing Table ===")
            print(routing_info)
            print("\n=== DNS Servers ===")
            for dns in dns_info:
                print(f"- {dns}")
        else:
            # Summary information
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except socket.error:
                local_ip = "N/A"

            print("\nSystem Information:")
            print(f"Hostname       : {hostname}")
            print(f"Local IP       : {local_ip}")
            print(f"DNS Servers    : {', '.join(dns_info) if dns_info else 'N/A'}")


# Wifi Diagnostics Command
class WifiDiagnosticsCommand(BaseCommand):
    """
    Perform WiFi diagnostics and analyze available networks.
    """
    def execute(self):
        """
        Execute WiFi diagnostics.
        """
        self.logger.info("Starting WiFi diagnostics...")
        wifi_networks = self.scan_wifi_networks()
        if not wifi_networks:
            self.logger.error("No WiFi networks found.")
            sys.exit(1)

        if self.args.ssid:
            target_ssid = self.args.ssid
            self.logger.info(f"Performing diagnostics for SSID: {target_ssid}")
            target_network = self.get_network_by_ssid(wifi_networks, target_ssid)
            if target_network:
                issues = self.diagnose_specific_network(target_network, wifi_networks)
                self.display_issues([issues] if issues else [])
            else:
                self.logger.error(f"SSID '{target_ssid}' not found among available networks.")
        else:
            self.logger.info("Performing generic WiFi diagnostics.")
            issues = self.diagnose_generic_wifi(wifi_networks)
            self.display_issues(issues)

    def scan_wifi_networks(self) -> List[Dict[str, str]]:
        """
        Scan available WiFi networks using nmcli.
        """
        self.logger.debug("Scanning available WiFi networks using nmcli...")
        try:
            scan_command = ['nmcli', '-f', 'SSID,SIGNAL,CHAN,SECURITY,BARS', 'device', 'wifi', 'list']
            result = subprocess.run(scan_command, capture_output=True, text=True, check=True)
            wifi_networks = self.parse_nmcli_output(result.stdout)
            self.logger.debug(f"Found {len(wifi_networks)} WiFi networks.")
            return wifi_networks
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to scan WiFi networks: {e}")
            return []
        except FileNotFoundError:
            self.logger.error("nmcli is not installed or not found in PATH.")
            return []

    def parse_nmcli_output(self, output: str) -> List[Dict[str, str]]:
        """
        Parse the output from nmcli to extract WiFi network details.
        """
        networks = []
        lines = output.strip().split('\n')
        if len(lines) < 2:
            return networks  # No networks found
        headers = lines[0].split()
        for line in lines[1:]:
            # Handle SSIDs with spaces by limiting splits
            parts = line.split(None, 4)  # Split into at most 5 parts
            if len(parts) < 5:
                # Incomplete information
                continue
            ssid, signal, channel, security, bars = parts
            networks.append({
                'SSID': ssid,
                'Signal': signal,
                'Channel': channel,
                'Security': security,
                'Bars': bars
            })
        return networks

    def get_network_by_ssid(self, networks: List[Dict[str, str]], ssid: str) -> Optional[Dict[str, str]]:
        """
        Retrieve a specific network's details by its SSID.
        """
        for network in networks:
            if network['SSID'] == ssid:
                return network
        return None

    def diagnose_specific_network(self, network: Dict[str, str], all_networks: List[Dict[str, str]]) -> List[str]:
        """
        Perform diagnostics on a specific WiFi network.
        """
        issues = []
        ssid = network['SSID']
        try:
            signal = int(network['Signal'])
        except ValueError:
            signal = 0
        try:
            channel = int(network['Channel'])
        except ValueError:
            channel = 0
        security = network['Security']

        # Check signal strength
        if signal < 40:
            issues.append(f"Low signal strength for SSID '{ssid}': {signal}%")

        # Check security protocols
        if security in ['OPEN', '--']:
            issues.append(f"SSID '{ssid}' is open and unsecured.")
        elif 'WPA3' not in security and 'WPA2' not in security:
            issues.append(f"SSID '{ssid}' is using weak security protocols: {security}")

        # Analyze channel interference
        channel_issues = self.analyze_channel_interference(channel, all_networks)
        if channel_issues:
            issues.extend(channel_issues)

        # Additional SSID-specific diagnostics can be added here

        return issues

    def diagnose_generic_wifi(self, networks: List[Dict[str, str]]) -> List[List[str]]:
        """
        Perform generic diagnostics across all available WiFi networks.
        """
        issues = []

        # Analyze channel interference across all networks
        channel_issues = self.analyze_channel_interference_generic(networks)
        issues.extend(channel_issues)

        # Check for open networks
        open_networks = [net for net in networks if net['Security'] in ['OPEN', '--']]
        for net in open_networks:
            issues.append([
                "WiFi Network",
                net['SSID'],
                "N/A",
                f"Open and unsecured network on channel {net['Channel']}."
            ])

        # Check for networks with weak signals
        weak_networks = [net for net in networks if self.safe_int(net['Signal']) < 40]
        for net in weak_networks:
            issues.append([
                "WiFi Network",
                net['SSID'],
                "N/A",
                f"Low signal strength: {net['Signal']}% on channel {net['Channel']}."
            ])

        return issues

    def analyze_channel_interference(self, channel: int, networks: List[Dict[str, str]]) -> List[str]:
        """
        Analyze channel interference for a specific channel.
        """
        overlapping_channels = self.get_overlapping_channels(channel)
        count = 0
        for net in networks:
            try:
                net_channel = int(net['Channel'])
            except ValueError:
                continue
            if net_channel in overlapping_channels:
                count += 1
        if count > 3:  # Threshold for interference
            return [f"High channel interference on channel {channel}: {count} networks overlapping."]
        return []

    def analyze_channel_interference_generic(self, networks: List[Dict[str, str]]) -> List[List[str]]:
        """
        Analyze channel interference across all channels.
        """
        issues = []
        channel_counts = {}
        for net in networks:
            try:
                channel = int(net['Channel'])
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
            except ValueError:
                continue

        for channel, count in channel_counts.items():
            if count > 4:  # Threshold for interference
                issues.append([
                    "WiFi Channel",
                    f"Channel {channel}",
                    "N/A",
                    f"High number of networks ({count}) on this channel causing interference."
                ])

        return issues

    def get_overlapping_channels(self, channel: int) -> List[int]:
        """
        Get overlapping channels for the 2.4GHz WiFi band.
        """
        # Define overlapping channels for 2.4GHz band
        overlapping = []
        if channel == 1:
            overlapping = [1, 2, 3]
        elif channel == 2:
            overlapping = [1, 2, 3, 4]
        elif channel == 3:
            overlapping = [1, 2, 3, 4, 5]
        elif channel == 4:
            overlapping = [2, 3, 4, 5, 6]
        elif channel == 5:
            overlapping = [3, 4, 5, 6, 7]
        elif channel == 6:
            overlapping = [4, 5, 6, 7, 8]
        elif channel == 7:
            overlapping = [5, 6, 7, 8, 9]
        elif channel == 8:
            overlapping = [6, 7, 8, 9, 10]
        elif channel == 9:
            overlapping = [7, 8, 9, 10, 11]
        elif channel == 10:
            overlapping = [8, 9, 10, 11, 12]
        elif channel == 11:
            overlapping = [9, 10, 11, 12, 13]
        elif channel == 12:
            overlapping = [10, 11, 12, 13, 14]
        elif channel == 13:
            overlapping = [11, 12, 13, 14]
        elif channel == 14:
            overlapping = [12, 13, 14]
        else:
            overlapping = [channel]
        return overlapping

    def display_issues(self, issues: List[List[str]]):
        """
        Display the identified WiFi diagnostic issues.
        """
        if not issues:
            self.logger.info("No WiFi issues detected.")
            return

        columns = ["Issue Type", "SSID", "IP Address", "Description"]
        self.print_table("WiFi Diagnostics Issues", columns, issues)

    def safe_int(self, value: str) -> int:
        """
        Safely convert a string to an integer, returning 0 on failure.
        """
        try:
            return int(value)
        except ValueError:
            return 0


# Argument Parser Setup
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Advanced Network Diagnostic Tool",
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
        '-d', '--detailed',
        action='store_true',
        help='Show detailed output where applicable.'
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # Subparser for system-info
    sys_info_parser = subparsers.add_parser(
        'system-info',
        aliases=['si'],
        help='Display detailed network information about the system.'
    )

    # Subparser for diagnose
    diagnose_parser = subparsers.add_parser(
        'diagnose',
        aliases=['dg'],
        help='Perform automated diagnostics on the network.'
    )

    # Subparser for rogue-detection
    rogue_parser = subparsers.add_parser(
        'rogue-detection',
        aliases=['rd'],
        help='Detect unauthorized or rogue devices on the network.'
    )

    # Subparser for traffic-monitor
    traffic_monitor_parser = subparsers.add_parser(
        'traffic-monitor',
        aliases=['tm'],
        help='Monitor network traffic to detect anomalies using Scapy.'
    )
    traffic_monitor_parser.add_argument(
        '--interface', '-i',
        type=str,
        required=True,
        help='Network interface to monitor (e.g., wlan0, eth0).'
    )

    # Subparser for wifi diagnostics
    wifi_parser = subparsers.add_parser(
        'wifi',
        aliases=['wf'],
        help='Perform WiFi diagnostics and analyze available networks.'
    )
    wifi_parser.add_argument(
        '--ssid', '-s',
        type=str,
        required=False,
        help='Specify the SSID to perform targeted diagnostics.'
    )

    return parser.parse_args()


# Logging Setup
def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """
    Set up the logging configuration.
    """
    logger = logging.getLogger("diagnose_network")
    logger.setLevel(logging.DEBUG)  # Set to lowest level; handlers will filter

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    if console:
        handler = RichHandler(rich_tracebacks=True)
        if debug:
            handler.setLevel(logging.DEBUG)
        elif verbose:
            handler.setLevel(logging.INFO)
        else:
            handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Fallback to standard StreamHandler
        ch = logging.StreamHandler()
        if debug:
            ch.setLevel(logging.DEBUG)
        elif verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# Command Handler Mapping
COMMAND_CLASSES = {
    'system-info': SystemInfoCommand,
    'si': SystemInfoCommand,
    'diagnose': DiagnosticsCommand,
    'dg': DiagnosticsCommand,
    'rogue-detection': RogueDeviceDetectionCommand,
    'rd': RogueDeviceDetectionCommand,
    'traffic-monitor': TrafficMonitorCommand,
    'tm': TrafficMonitorCommand,
    'wifi': WifiDiagnosticsCommand,
    'wf': WifiDiagnosticsCommand,
}


# Main Function
def main():
    """
    Main function to orchestrate the network diagnostic process.
    """
    args = parse_arguments()
    logger = setup_logging(verbose=args.verbose, debug=args.debug)

    # Instantiate and execute the appropriate command
    command_class = COMMAND_CLASSES.get(args.command)
    if not command_class:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

    command = command_class(args, logger, detailed=args.detailed)
    command.execute()

    logger.info("Network diagnostics completed successfully.")


if __name__ == "__main__":
    main()
