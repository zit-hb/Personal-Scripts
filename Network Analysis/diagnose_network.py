#!/usr/bin/env python3

# -------------------------------------------------------
# Script: diagnose_network.py
#
# Description:
# This script provides comprehensive network diagnostics to
# automatically troubleshoot network issues.
# It includes functionalities such as system information gathering, device discovery,
# automated diagnostics, advanced network traffic monitoring, and wifi analysis.
#
# Usage:
# ./diagnose_network.py [command] [options]
#
# Commands:
# - system-info (si)          Display detailed network information about the system.
# - diagnose (dg)             Perform automated diagnostics on the network.
# - traffic-monitor (tm)      Monitor network traffic to detect anomalies using Scapy.
# - wifi (wf)                 Perform WiFi diagnostics and analyze available networks.
#
# Global Options:
# -v, --verbose               Enable verbose logging (INFO level).
# -vv, --debug                Enable debug logging (DEBUG level).
#
# System Info Command Options:
# -t, --traceroute            Perform a traceroute to a specified address (default: 8.8.8.8, 2001:4860:4860::8888).
#
# Traffic Monitor Command Options:
# -i, --interface             Specify the network interface to monitor (e.g., wlan0, eth0).
# --dhcp-threshold            Set DHCP flood threshold (default: 100).
# --port-scan-threshold       Set port scan threshold (default: 50).
# --dns-exfil-threshold       Set DNS exfiltration threshold (default: 1000).
# --bandwidth-threshold       Set bandwidth abuse threshold in bytes per minute (default: 1000000).
# --icmp-threshold            Set ICMP flood threshold (default: 500).
# --syn-threshold             Set SYN flood threshold (default: 1000).
# --http-threshold            Set HTTP abuse threshold (default: 1000).
# --malformed-threshold       Set malformed packets threshold (default: 50).
# --rogue-dhcp-threshold      Set rogue DHCP server threshold (default: 1).
#
# Diagnostics Command Options:
# -V, --include-virtual       Include virtual interfaces in network scanning.
# -N, --nikto                 Enable Nikto scanning for discovered devices.
# -C, --credentials           Enable default credentials check for discovered devices.
# -d, --discovery             Perform device discovery only.
# -6, --ipv6                  Enable IPv6 scanning.
# -o, --output-file           Specify a file to save discovered devices.
#
# WiFi Command Options:
# -s, --ssid                  Specify the SSID to perform targeted diagnostics.
#                             If not specified, performs generic WiFi checks.
# -i, --interface             Specify the network interface to scan (e.g., wlan0, wlp3s0).
# -m, --signal-threshold      Set the minimum signal strength threshold (default: 50).
#
# Returns:
# Exit code 0 on success, non-zero on failure.
#
# Requirements:
#  System-Info Command:
#  - resolvectl (install via: apt install systemd-resolv)
#  - traceroute (install via: apt install traceroute)
#  Diagnose Command:
#  - requests (install via: pip install requests)
#  - nmap (install via: apt install nmap)
#  - nikto (install via: apt install nikto)
#  Traffic-Monitor Command:
#  - scapy (install via: pip install scapy)
#  Wifi Command:
#  - nmcli (install via: apt install network-manager)
#  Optional:
#  - rich (install via: pip install rich)
#  - python-dotenv (install via: pip install python-dotenv)
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
import struct
import ssl
import json
import time
import threading
import queue
import shutil
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

# Ignore unnecessary warnings
import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

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
    RICH_AVAILABLE = True
except ImportError:
    Console = None
    RICH_AVAILABLE = False

# Attempt to import scapy for advanced network monitoring
try:
    from scapy.all import sniff, ARP, DHCP, IP, TCP, UDP, ICMP, DNS, DNSQR, Ether, Raw
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


# Issue data classes
@dataclass
class DiagnoseIssue:
    device_type: str
    hostname: str
    ip: str
    description: str


@dataclass
class WifiIssue:
    issue_type: str
    location: str
    ip_address: str
    description: str


# Configuration data classes
@dataclass
class CredentialsConfig:
    """
    Configuration for default credentials, organized by vendor and generic credentials.
    """
    vendor_credentials: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: {
        "cisco": [
            {"username": "admin", "password": "admin"},
            {"username": "cisco", "password": "cisco"},
            {"username": "support", "password": "support"},
        ],
        "dlink": [
            {"username": "admin", "password": "admin"},
            {"username": "admin", "password": "password"},
            {"username": "user", "password": "user"},
        ],
        "netgear": [
            {"username": "admin", "password": "password"},
            {"username": "admin", "password": "netgear"},
            {"username": "admin", "password": "1234"},
        ],
        "tplink": [
            {"username": "admin", "password": "admin"},
            {"username": "user", "password": "user"},
            {"username": "admin", "password": "password"},
        ],
        "huawei": [
            {"username": "admin", "password": "admin"},
            {"username": "root", "password": "root"},
            {"username": "huawei", "password": "huawei"},
        ],
        "asus": [
            {"username": "admin", "password": "admin"},
            {"username": "admin", "password": "asus"},
        ],
        "linksys": [
            {"username": "admin", "password": "admin"},
            {"username": "admin", "password": "linksys"},
        ],
        "zyxel": [
            {"username": "admin", "password": "1234"},
            {"username": "admin", "password": "admin"},
        ],
        "mikrotik": [
            {"username": "admin", "password": ""},
            {"username": "admin", "password": "admin"},
        ],
        "belkin": [
            {"username": "admin", "password": "password"},
            {"username": "admin", "password": "admin"},
        ],
        # Add more vendors as needed
    })
    generic_credentials: List[Dict[str, str]] = field(default_factory=lambda: [
        {"username": "admin", "password": "admin"},
        {"username": "admin", "password": "password"},
        {"username": "root", "password": "root"},
        {"username": "user", "password": "user"},
        {"username": "guest", "password": "guest"},
        {"username": "admin", "password": "1234"},
        {"username": "admin", "password": "password123"},
        {"username": "admin", "password": "default"},
        {"username": "admin", "password": "letmein"},
        {"username": "admin", "password": "admin123"},
        # Add more generic credentials as needed
    ])

    def get_vendor_credentials(self, vendor: str) -> List[Dict[str, str]]:
        """
        Retrieve vendor-specific credentials based on the vendor string.
        Combines with generic credentials, avoiding duplicates.
        """
        credentials = []
        for vendor_substr, creds in self.vendor_credentials.items():
            if vendor_substr.lower() in vendor.lower():
                credentials.extend(creds)

        if credentials:
            # Combine with generic_credentials, avoiding duplicates
            seen = set()
            unique_credentials = []
            for cred in credentials:
                key = (cred['username'], cred['password'])
                if key not in seen:
                    unique_credentials.append(cred)
                    seen.add(key)
            for cred in self.generic_credentials:
                key = (cred['username'], cred['password'])
                if key not in seen:
                    unique_credentials.append(cred)
                    seen.add(key)
            return unique_credentials
        else:
            return self.generic_credentials


@dataclass
class EndpointsConfig:
    common_sensitive_endpoints: Set[str] = field(default_factory=lambda: {
        "/backup",
        "/diag.html",
        "/status",
        "/advanced",
        "/system",
        "/tools",
        "/filemanager",
        "/download",
        "/logs",
        "/debug",
    })
    vendor_additional_sensitive_endpoints: Dict[str, Set[str]] = field(default_factory=lambda: {
        'fritz!box': {
            "/admin/config.php",
            "/diag_wps.html",
        },
        'asus': {
            "/admin/config.php",
        },
        'netgear': {
            "/cgi-bin/fwupdate",
            "/status/wps",
        },
        'tp-link': set(),
        'd-link': {
            "/status.html",
        },
        'linksys': {
            "/status.html",
        },
        'belkin': {
            "/cgi-bin/admin/config",
            "/status.cgi",
        },
        'synology': {
            "/webman/index.cgi",
            "/status.cgi",
        },
        'ubiquiti': {
            "/cgi-bin/status.cgi",
        },
        'mikrotik': {
            "/login",
        },
        'zyxel': {
            "/cgi-bin/admin/config",
        },
        'huawei': {
            "/cgi-bin/hwcfg.cgi",
        },
        'apple': {
            "/airport/admin",
        },
    })

    def get_vendor_config(self, vendor: str) -> Dict[str, Set[str]]:
        # Collect all additional sensitive endpoints where the key is a substring of the vendor
        additional_sensitive = set()
        for key, endpoints in self.vendor_additional_sensitive_endpoints.items():
            if key.lower() in vendor.lower():
                additional_sensitive.update(endpoints)

        return {
            'sensitive_endpoints': self.common_sensitive_endpoints.union(additional_sensitive)
        }


@dataclass
class HttpSecurityConfig:
    """
    Configuration for HTTP security headers.
    """
    security_headers: Set[str] = field(default_factory=lambda: {
        'Content-Security-Policy',
        'Strict-Transport-Security',
        'X-Frame-Options',
        'X-Content-Type-Options',
        'Referrer-Policy',
        'Feature-Policy',
        'Permissions-Policy'
    })


@dataclass
class GamingServicesConfig:
    """
    Configuration for gaming-specific services per vendor.
    Maps vendor names to their associated gaming services and ports.
    """
    gaming_services: Dict[str, Dict[int, str]] = field(default_factory=lambda: {
        'sony': {
            3075: "PlayStation Network",
            3076: "PlayStation Network"
        },
        'microsoft': {
            3074: "Xbox Live",
            # Add more ports and services as needed
        },
        'nintendo': {
            6667: "Nintendo Switch",
            12400: "Nintendo Switch"
        },
        # Add additional vendors and their gaming services here
    })


@dataclass
class DeviceType:
    """
    Represents a type of network device with specific identification criteria.
    """
    name: str
    vendors: Set[str] = field(default_factory=set)
    ports: Set[str] = field(default_factory=set)
    os_keywords: Set[str] = field(default_factory=set)
    priority: int = 0  # Lower number means higher priority

    def matches(self, device: Dict[str, str], mac_lookup) -> bool:
        """
        Determines if a given device matches the criteria of this device type.
        """
        ports = set(device.get("Ports", []))
        os_info = device.get("OS", "").lower()
        mac = device.get("MAC", "N/A")
        vendor = mac_lookup.get_vendor(mac).lower() if mac != 'N/A' else "unknown"

        # Check vendor match
        vendor_match = any(v in vendor for v in self.vendors)

        # Check ports match
        ports_match = bool(self.ports.intersection(ports))

        # Check OS keywords
        os_match = any(keyword in os_info for keyword in self.os_keywords) if self.os_keywords else False

        # Combine conditions based on device type requirements
        if self.name == "Phone":
            return vendor_match or ports_match
        if self.name == "Smart":
            return vendor_match or os_match
        if self.name == "Game":
            return (vendor_match or os_match) and ports_match
        if self.name == "Computer":
            return os_match or ports

        # Default: require both vendor and ports to match
        return vendor_match and ports_match


@dataclass
class DeviceTypeConfig:
    """
    Configuration for various device types used in the network.
    Contains a list of DeviceType instances, each defining criteria for a specific type of device.
    """
    device_types: List[DeviceType] = field(default_factory=lambda: [
        DeviceType(
            name="Router",
            vendors={'fritz!box', 'asus', 'netgear', 'tp-link', 'd-link',
                     'linksys', 'belkin', 'synology', 'ubiquiti', 'mikrotik', 'zyxel'},
            ports={'80/tcp http', '443/tcp https', '23/tcp telnet', '22/tcp ssh'},
            priority=1
        ),
        DeviceType(
            name="Switch",
            vendors={'cisco', 'hp', 'd-link', 'netgear', 'ubiquiti', 'juniper', 'huawei'},
            ports={'22/tcp ssh', '23/tcp telnet', '161/udp snmp', '161/tcp snmp'},
            priority=2
        ),
        DeviceType(
            name="Printer",
            vendors={'hp', 'canon', 'epson', 'brother', 'lexmark', 'samsung', 'xerox'},
            ports={'9100/tcp jetdirect', '515/tcp lpd', '631/tcp ipp'},
            priority=3
        ),
        DeviceType(
            name="Phone",
            vendors={'cisco', 'yealink', 'polycom', 'avaya', 'grandstream'},
            ports={'5060/tcp sip', '5060/udp sip'},
            priority=4
        ),
        DeviceType(
            name="Smart",
            vendors={'google', 'amazon', 'ring', 'nest', 'philips', 'samsung', 'lg', 'lifi labs', 'roborock'},
            os_keywords={'smart', 'iot', 'camera', 'thermostat', 'light', 'sensor', 'hub'},
            priority=5
        ),
        DeviceType(
            name="Game",
            vendors={'sony', 'microsoft', 'nintendo'},
            ports={'3074/tcp xbox', '3074/udp xbox', '3075/tcp playstation',
                   '3075/udp playstation', '3076/tcp nintendo', '3076/udp nintendo'},
            priority=6
        ),
        DeviceType(
            name="Computer",
            ports={'22/tcp ssh', '139/tcp netbios-ssn', '445/tcp microsoft-ds',
                   '3389/tcp rdp', '5900/tcp vnc'},
            os_keywords={'windows', 'macos', 'linux'},
            priority=7
        ),
    ])


@dataclass
class AppConfig:
    """
    Comprehensive application configuration encompassing credentials, endpoints, device types,
    HTTP security settings, gaming services, SNMP communities, and more.
    """
    credentials: CredentialsConfig = CredentialsConfig()
    endpoints: EndpointsConfig = EndpointsConfig()
    device_types: DeviceTypeConfig = DeviceTypeConfig()
    http_security: HttpSecurityConfig = HttpSecurityConfig()
    gaming_services: GamingServicesConfig = GamingServicesConfig()
    snmp_communities: Set[str] = field(default_factory=lambda: {"public", "private", "admin"})


# MAC Vendor Lookup Class
class MacVendorLookup:
    """
    Lookup MAC address vendors using OUI data.
    """
    DEFAULT_OUI_URL = "https://standards-oui.ieee.org/oui/oui.txt"
    DEFAULT_OUI_JSON_PATH = "oui.json"

    def __init__(self, logger: logging.Logger, oui_url: str = None, oui_json_path: str = None):
        """
        Initialize the MacVendorLookup with a logger, and optionally customize the OUI URL and JSON path.
        """
        self.logger = logger
        self.oui_url = oui_url if oui_url is not None else self.DEFAULT_OUI_URL
        self.oui_json_path = oui_json_path if oui_json_path is not None else self.DEFAULT_OUI_JSON_PATH
        self.oui_dict = self.load_oui_data()

    def load_oui_data(self) -> Dict[str, str]:
        """
        Load OUI data from a local JSON file or download and parse it from the IEEE website.
        """
        if os.path.exists(self.oui_json_path):
            self.logger.debug(f"Loading OUI data from local '{self.oui_json_path}' file.")
            try:
                with open(self.oui_json_path, "r") as f:
                    data = json.load(f)
                # Check if data has 'timestamp' and 'data'
                if isinstance(data, dict) and 'data' in data:
                    self.logger.debug("OUI data loaded successfully from local JSON file.")
                    return data['data']
                elif isinstance(data, dict):
                    self.logger.debug("OUI data format from local JSON file is unexpected. Proceeding to download.")
            except Exception as e:
                self.logger.error(f"Failed to load local OUI JSON data: {e}")

        # Download and parse OUI data
        self.logger.debug(f"Downloading OUI data from {self.oui_url}.")
        try:
            response = requests.get(self.oui_url, timeout=10)
            if response.status_code == 200:
                oui_text = response.text
                self.logger.debug("OUI data downloaded successfully.")
                data = self.parse_oui_txt(oui_text)
                with open(self.oui_json_path, "w") as f:
                    json.dump({"timestamp": time.time(), "data": data}, f, indent=2)
                self.logger.debug(f"OUI data parsed and saved to '{self.oui_json_path}'.")
                return data
            else:
                self.logger.error(f"Failed to download OUI data: HTTP {response.status_code}")
        except Exception as e:
            self.logger.error(f"Exception while downloading OUI data: {e}")
        return {}

    def parse_oui_txt(self, text: str) -> Dict[str, str]:
        """
        Parse the OUI text data and return a dictionary mapping MAC prefixes to vendor names.
        """
        oui_dict: Dict[str, str] = {}
        lines = text.splitlines()
        for line in lines:
            if "(hex)" in line:
                parts = line.split("(hex)")
                if len(parts) >= 2:
                    mac_prefix = parts[0].strip().replace("-", "").upper()
                    company = parts[1].strip()
                    if len(mac_prefix) == 6:
                        oui_dict[mac_prefix] = company
        return oui_dict

    def get_vendor(self, mac: str) -> str:
        """
        Get the vendor name for a given MAC address.
        """
        if not mac:
            self.logger.warning("Empty MAC address provided.")
            return "Unknown"

        # Normalize MAC address
        mac_clean = mac.upper().replace(":", "").replace("-", "").replace(".", "")
        if len(mac_clean) < 6:
            self.logger.warning(f"Invalid MAC address format: {mac}")
            return "Unknown"

        mac_prefix = mac_clean[:6]
        oui_entry = self.oui_dict.get(mac_prefix, "Unknown")
        self.logger.debug(f"MAC Prefix: {mac_prefix}, Vendor: {oui_entry}")
        return oui_entry


# Base class for all commands
class BaseCommand(ABC):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger, config: AppConfig, mac_lookup: Optional[MacVendorLookup] = None):
        """
        Initialize the BaseCommand with arguments and logger.
        """
        self.args = args
        self.logger = logger
        self.config = config
        self.mac_lookup = mac_lookup

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the command.
        """
        pass

    def print_table(self, title: str, columns: List[str], rows: List[List[str]]) -> None:
        """
        Print a table with the given title, columns, and rows.
        """
        if console:
            table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
            for col in columns:
                table.add_column(col, style="cyan", overflow="fold", no_wrap=False)
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


# General Device Diagnostics Class
class GeneralDeviceDiagnostics(ABC):
    """
    Abstract base class for device diagnostics.
    """
    def __init__(self, device_type: str, device: Dict[str, str], logger: logging.Logger, args: argparse.Namespace, config: AppConfig, mac_lookup: Optional[MacVendorLookup] = None):
        """
        Initialize with device information and a logger.
        """
        self.device_type = device_type
        self.device = device
        self.logger = logger
        self.args = args
        self.config = config
        self.mac_lookup = mac_lookup

    @abstractmethod
    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        """
        Perform diagnostics on the device.
        """
        pass

    def check_port(self, ip: str, port: int) -> bool:
        """
        Check if a specific port is open on the given IP.
        """
        try:
            self.logger.debug(f"Checking if port {port} is open on {ip}.")
            with socket.create_connection((ip, port), timeout=2):
                self.logger.debug(f"Port {port} on {ip} is open.")
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            self.logger.debug(f"Port {port} on {ip} is closed or unreachable.")
            return False
        except Exception as e:
            self.logger.warning(f"Error while checking port {port} on {ip}: {e}")
            return False

    def ping_device(self, ip: str) -> bool:
        """
        Ping the specified IP address to check its reachability.
        """
        try:
            if ':' in ip:
                # Likely IPv6
                cmd = ['ping6', '-c', '1', '-W', '2', ip]
            else:
                # IPv4
                cmd = ['ping', '-c', '1', '-W', '2', ip]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                self.logger.debug(f"Ping to {ip} successful.")
                return True
            else:
                self.logger.debug(f"Ping to {ip} failed.")
                return False
        except Exception as e:
            self.logger.error(f"Error while pinging {ip}: {e}")
            return False


# Common Diagnostics Class
class CommonDiagnostics(GeneralDeviceDiagnostics):
    """
    Encapsulates shared diagnostic functionalities for various devices.
    Enhancements include caching of port and SSL checks, optimized protocol handling,
    and integration with Nikto for web server vulnerability scanning.
    """
    def __init__(self, device_type: str, device: Dict[str, str], logger: logging.Logger, args: argparse.Namespace, config: AppConfig, mac_lookup: Optional[MacVendorLookup] = None):
        """
        Initialize with device information, a logger, and configuration.
        Sets up caches for port and SSL certificate checks.
        """
        super().__init__(device_type, device, logger, args, config, mac_lookup)

        # Initialize caches
        self.port_status_cache: Dict[int, bool] = {}      # Cache for port open status {port: bool}
        self.ssl_valid_cache: Optional[bool] = None       # Cache for SSL certificate validity
        self.nikto_scanned: bool = False                  # Flag to ensure Nikto scan is performed only once

    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        raise NotImplementedError("Subclasses must implement this method.")

    def perform_standard_checks(self, ip: str, hostname: str, vendor_key: str) -> List[DiagnoseIssue]:
        """
        Perform all standard checks and return a list of issues.
        Utilizes caching to prevent redundant port and SSL certificate checks.
        Each check method decides independently whether to perform its checks.
        """
        issues: List[DiagnoseIssue] = []

        # Check and cache port statuses
        self._check_ports(ip, [80, 443])

        # Determine protocol availability
        protocols = self._determine_protocols()

        # Check SSL validity if HTTPS is available
        if protocols['https']:
            self._check_ssl_validity(ip, hostname)

        # Update protocol availability based on SSL validity
        protocols = self._determine_protocols()

        # Perform individual checks
        issues.extend(self._check_web_services(protocols, hostname, ip, vendor_key))
        issues.extend(self._check_snmp_configuration(ip))

        return issues

    def _check_ports(self, ip: str, ports: List[int]) -> None:
        """
        Check the status of the specified ports and update the cache.
        """
        for port in ports:
            if port not in self.port_status_cache:
                self.port_status_cache[port] = self._is_port_open(ip, port)
                self.logger.debug(f"Port {port} status for {ip}: {self.port_status_cache[port]}")

    def _is_port_open(self, ip: str, port: int) -> bool:
        """
        Check if a specific port is open on the given IP.
        """
        try:
            with socket.create_connection((ip, port), timeout=3):
                self.logger.debug(f"Port {port} on {ip} is open.")
                return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            self.logger.debug(f"Port {port} on {ip} is closed or unreachable: {e}")
            return False

    def _determine_protocols(self) -> Dict[str, bool]:
        """
        Determine the availability of HTTP and HTTPS protocols based on port statuses.
        """
        return {
            'http': self.port_status_cache.get(80, False),
            'https': self.port_status_cache.get(443, False) and self.ssl_valid_cache is True
        }

    def _check_ssl_validity(self, ip: str, hostname: str) -> None:
        """
        Check if HTTPS is valid and cache the result.
        """
        if self.ssl_valid_cache is None:
            self.ssl_valid_cache = self._is_valid_https(ip, hostname)
            self.logger.debug(f"HTTPS validity for {hostname} ({ip}): {self.ssl_valid_cache}")

    def _is_valid_https(self, ip: str, hostname: str) -> bool:
        """
        Check if HTTPS connection can be established using _make_request.
        Returns True if connection is successful, False otherwise.
        """
        try:
            # Attempt to make a simple GET request to the root endpoint over HTTPS
            self._make_request('https', ip, hostname, endpoint="/", timeout=5)
            self.logger.debug(f"Successfully connected to HTTPS {hostname} ({ip}).")
            return True
        except requests.exceptions.SSLError as e:
            self.logger.info(f"SSL Error when connecting to HTTPS {hostname} ({ip}): {e}")
            return False
        except requests.RequestException as e:
            self.logger.info(f"Failed to establish HTTPS connection to {hostname} ({ip}): {e}")
            return False

    def _make_request(self, protocol: str, ip: str, hostname: str, endpoint: str = "/", port: Optional[int] = None, timeout: int = 5) -> requests.Response:
        """
        Helper method to make HTTP/HTTPS requests using the IP in the URL
        and setting the Host header to the hostname.
        """
        if port is None:
            port = 443 if protocol == 'https' else 80

        url = f"{protocol}://{ip}:{port}{endpoint}"
        headers = {'Host': hostname}
        verify = protocol == 'https'

        self.logger.debug(f"Making {protocol.upper()} request to {url} with Host header '{hostname}'.")
        response = requests.get(url, headers=headers, timeout=timeout, verify=verify)
        return response

    def _check_admin_interface(self, protocol: str, ip: str, hostname: str, vendor_key: str) -> List[DiagnoseIssue]:
        """
        Generic method to check admin interfaces over the specified protocol.
        """
        issues = []
        port = 80 if protocol == 'http' else 443

        try:
            self.logger.debug(f"Checking {protocol.upper()} admin interfaces on {ip} for vendor '{vendor_key}'.")
            admin_endpoints = self.config.endpoints.get_vendor_config(vendor_key)['sensitive_endpoints']
            for endpoint in admin_endpoints:
                try:
                    response = self._make_request(protocol, ip, hostname, endpoint=endpoint, port=port)
                    if 200 <= response.status_code < 300:
                        # Admin interface is accessible
                        issues.append(self.create_issue(f"Admin interface {endpoint} is accessible over {protocol.upper()}"))
                        self.logger.info(
                            f"Admin interface {endpoint} is accessible over {protocol.upper()} on {ip}."
                        )
                        break  # Assume one admin interface is sufficient
                except requests.RequestException:
                    continue  # Try the next endpoint
        except Exception as e:
            self.logger.error(f"Error while checking {protocol.upper()} admin interface on {ip}: {e}")
        return issues

    def _check_web_services(self, protocols: Dict[str, bool], hostname: str, ip: str, vendor_key: str) -> List[DiagnoseIssue]:
        """
        Perform web service-specific checks including HTTP response validation and Nikto scanning.
        Each web service check decides independently based on protocol availability.
        """
        issues: List[DiagnoseIssue] = []
        for protocol in ['https', 'http']:
            if protocols.get(protocol):
                issues.extend(self._validate_web_service_response(protocol, hostname, ip))
                issues.extend(self._check_admin_interface(protocol, ip, hostname, vendor_key))
                if self.args.nikto:
                    issues.extend(self._scan_web_service_with_nikto(protocol, hostname, ip))
                if self.args.credentials:
                    issues.extend(self._check_default_credentials(protocol, ip, vendor_key))

        return issues

    def _validate_web_service_response(self, protocol: str, hostname: str, ip: str) -> List[DiagnoseIssue]:
        """
        Perform comprehensive checks on the specified protocol's service, including response codes and security headers.
        """
        issues: List[DiagnoseIssue] = []
        port = 443 if protocol == 'https' else 80

        try:
            response = self._make_request(protocol, ip, hostname, endpoint="/", timeout=5)
            # Check for successful response
            if not response.ok:
                issues.append(self.create_issue(f"{protocol.upper()} response code {response.status_code}"))
                self.logger.info(f"{protocol.upper()} response code {response.status_code} from {hostname} ({ip})")

            # Check for security headers
            issues.extend(self._check_security_headers(response, hostname, ip))
        except requests.exceptions.SSLError as ssl_err:
            issues.append(self.create_issue(f"SSL Error on {protocol.upper()} port {port} - {ssl_err}"))
            self.logger.error(f"SSL Error on {protocol.upper()} port {port} for {hostname} ({ip}): {ssl_err}")
        except requests.exceptions.ConnectionError as conn_err:
            issues.append(self.create_issue(f"Connection Error on {protocol.upper()} port {port} - {conn_err}"))
            self.logger.error(
                f"Connection Error on {protocol.upper()} port {port} for {hostname} ({ip}): {conn_err}"
            )
        except requests.exceptions.Timeout:
            issues.append(self.create_issue(f"Timeout while connecting to {protocol.upper()} port {port}"))
            self.logger.error(f"Timeout while connecting to {protocol.upper()} port {port} for {hostname} ({ip})")
        except Exception as e:
            issues.append(self.create_issue(f"Unexpected error on {protocol.upper()} port {port} - {e}"))
            self.logger.error(f"Unexpected error on {protocol.upper()} port {port} for {hostname} ({ip}): {e}")

        return issues

    def _check_security_headers(self, response: requests.Response, hostname: str, ip: str) -> List[DiagnoseIssue]:
        """
        Check for the presence of critical security headers in the HTTP response.
        """
        issues = []
        security_headers = self.config.http_security.security_headers

        missing_headers = [header for header in security_headers if header.lower() not in response.headers.lower_items()]
        if missing_headers:
            issues.append(self.create_issue(f"Missing security headers: {', '.join(missing_headers)}"))
            self.logger.info(f"Missing security headers on {hostname} ({ip}): {', '.join(missing_headers)}")
        return issues

    def _scan_web_service_with_nikto(self, protocol: str, hostname: str, ip: str) -> List[DiagnoseIssue]:
        """
        Perform a Nikto scan on the web service and create issues based on the findings.
        """
        issues = []

        # Check if Nikto has already been scanned for this device
        if getattr(self, 'nikto_scanned', False):
            self.logger.debug(f"Nikto scan already performed for {hostname} ({ip}). Skipping.")
            return issues

        # Check if Nikto is installed
        if not shutil.which('nikto'):
            self.logger.info("Nikto is not installed. Skipping Nikto scan.")
            return issues  # No issues to add since the scan was skipped

        self.logger.debug(f"Starting Nikto scan on {protocol.upper()}://{ip}.")

        # Construct the Nikto command with XML output to stdout
        nikto_command = [
            'nikto',
            '-h', f"{protocol}://{ip}",
            '-Format', 'xml',
            '-output', '-'  # Output to stdout
        ]

        try:
            # Execute the Nikto scan and capture stdout
            result = subprocess.run(
                nikto_command,
                check=True,
                capture_output=True,
                text=True
            )
            nikto_output = result.stdout
            self.logger.debug("Nikto scan completed. Parsing results from stdout.")

            # Parse the Nikto XML output
            root = ET.fromstring(nikto_output)

            # Iterate through each issue found by Nikto
            for item in root.findall('.//item'):
                description_elem = item.find('description')
                description = description_elem.text.strip() if description_elem is not None else "No description provided."

                uri_elem = item.find('uri')
                uri = uri_elem.text.strip() if uri_elem is not None else "N/A"

                method_elem = item.find('method')
                method = method_elem.text.strip() if method_elem is not None else "UNKNOWN"

                issue_description = f"Nikto: {description} [URI: {uri}, Method: {method}]"

                # Create an issue for each Nikto finding
                issues.append(self.create_issue(issue_description))
                self.logger.info(f"Nikto Issue on {hostname} ({ip}): {issue_description}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Nikto scan failed on {protocol}://{ip}: {e.stderr.strip()}")
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse Nikto XML output for {protocol}://{ip}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during Nikto scan on {protocol}://{ip}: {e}")
        finally:
            self.nikto_scanned = True

        return issues

    def _snmp_query(self, ip: str, community: str) -> bool:
        """
        Perform an SNMP GET request to the specified IP using the provided community string.
        Returns True if the community string is valid (i.e., SNMP is accessible), False otherwise.
        """
        port = 161  # Standard SNMP port
        timeout = 2  # Seconds

        try:
            # Create a UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)

            # SNMP GET Request Packet Construction (SNMPv1)
            # ASN.1 BER encoding

            # Version: SNMPv1 (0)
            version = b'\x02\x01\x00'

            # Community
            community_bytes = community.encode('utf-8')
            community_packed = b'\x04' + struct.pack('B', len(community_bytes)) + community_bytes

            # PDU (GetRequest-PDU)
            pdu_type = 0xA0  # GetRequest-PDU

            # Request ID: arbitrary unique identifier
            request_id = 1
            request_id_packed = b'\x02\x01' + struct.pack('B', request_id)

            # Error Status and Error Index
            error_status = b'\x02\x01\x00'  # noError
            error_index = b'\x02\x01\x00'   # 0

            # Variable Binding: sysDescr.0 OID
            oid = b'\x06\x08\x2B\x06\x01\x02\x01\x01\x01\x00'  # OID for sysDescr.0
            value = b'\x05\x00'  # NULL
            varbind = b'\x30' + struct.pack('B', len(oid) + len(value)) + oid + value

            # Variable Binding List
            varbind_list = b'\x30' + struct.pack('B', len(varbind)) + varbind

            # PDU Body
            pdu_body = request_id_packed + error_status + error_index + varbind_list
            pdu = struct.pack('B', pdu_type) + struct.pack('B', len(pdu_body)) + pdu_body

            # Full SNMP Packet
            snmp_packet = b'\x30' + struct.pack('B', len(version) + len(community_packed) + len(pdu)) + version + community_packed + pdu

            # Send SNMP GET request
            sock.sendto(snmp_packet, (ip, port))
            self.logger.debug(f"Sent SNMP GET request to {ip} with community '{community}'.")

            # Receive response
            try:
                data, _ = sock.recvfrom(4096)
                self.logger.debug(f"Received SNMP response from {ip}.")

                # Basic validation of SNMP response
                if data:
                    # Check if the response is a GetResponse-PDU (0xA2)
                    pdu_response_type = data[0]
                    if pdu_response_type == 0xA2:
                        self.logger.info(f"SNMP community '{community}' is valid on {ip}.")
                        return True
            except socket.timeout:
                self.logger.debug(f"SNMP GET request to {ip} with community '{community}' timed out.")
            finally:
                sock.close()

        except Exception as e:
            self.logger.error(f"Error while performing SNMP GET to {ip} with community '{community}': {e}")

        self.logger.info(f"SNMP community '{community}' is invalid or not accessible on {ip}.")
        return False

    def _check_snmp_configuration(self, ip: str) -> List[DiagnoseIssue]:
        """
        Check SNMP configuration for potential vulnerabilities.
        """
        issues = []
        try:
            self.logger.debug(f"Checking SNMP configuration on {ip}.")
            # Example logic: Attempt SNMP queries with default community strings
            for community in self.config.snmp_communities:
                if self._snmp_query(ip, community):
                    issues.append(self.create_issue(f"SNMP is accessible with community string '{community}'."))
                    self.logger.info(f"SNMP is accessible with community string '{community}' on {ip}.")
                    break  # Assume one accessible SNMP community is sufficient
        except Exception as e:
            self.logger.error(f"Error while checking SNMP configuration on {ip}: {e}")
        return issues

    def _check_default_credentials(self, protocol: str, ip: str, vendor_key: str) -> List[DiagnoseIssue]:
        """
        Check for default credentials on the device based on the vendor.
        """
        issues = []
        try:
            self.logger.debug(f"Checking default credentials on {ip}.")

            # Get the list of credentials to try based on the vendor
            credentials = self.config.credentials.get_vendor_credentials(vendor_key)
            self.logger.debug(f"Total credentials to check for {ip}: {len(credentials)}.")

            # Attempt to authenticate with each credential
            for cred in credentials:
                username = cred['username']
                password = cred['password']
                self.logger.debug(f"Attempting authentication on {ip} with username='{username}' and password='{password}'.")

                if self._authenticate(protocol, ip, username, password):
                    issues.append(self.create_issue(f"Default credentials used: {username}/{password}"))
                    self.logger.info(f"Default credentials used on {ip}: {username}/{password}")
                    break  # Stop after first successful authentication
        except Exception as e:
            self.logger.error(f"Error while checking default credentials on {ip}: {e}")
        return issues

    def _authenticate(self, protocol:str, ip: str, username: str, password: str) -> bool:
        """
        Attempt to authenticate to the device's admin interface using HTTP Basic Auth.
        Returns True if authentication is successful, False otherwise.
        """
        try:
            url = f"{protocol}://{ip}/"
            headers = {'Host': self.device.get("Hostname", "N/A")}
            self.logger.debug(f"Making {protocol.upper()} request to {url} with username='{username}' and password='{password}'.")

            # Make the GET request with Basic Auth
            response = requests.get(url, auth=(username, password), headers=headers, timeout=5, verify=(protocol == 'https'))

            # Define criteria for successful authentication
            if response.status_code in [200, 302]:
                self.logger.debug(f"Authentication succeeded for {username}/{password} on {url} with status code {response.status_code}.")
                return True
            else:
                self.logger.debug(f"Authentication failed for {username}/{password} on {url} with status code {response.status_code}.")
        except requests.RequestException as e:
            self.logger.debug(f"Authentication request to {url} with {username}/{password} failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during authentication to {url} with {username}/{password}: {e}")
        return False

    def create_issue(self, description: str) -> DiagnoseIssue:
        """
        Create a DiagnoseIssue instance with the device's details.
        """
        return DiagnoseIssue(
            device_type=self.device_type,
            hostname=self.device.get("Hostname", "N/A"),
            ip=self.device.get("IP", "N/A"),
            description=description
        )

    def diagnose_common(self) -> Dict:
        """
        Perform common diagnostics steps and return a context dictionary.
        """
        context = {}
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")
        mac = self.device.get("MAC", "").upper()

        if ip == "N/A":
            self.logger.debug(f"{hostname} IP is N/A. Skipping diagnostics.")
            return context

        self.logger.debug(f"Starting common diagnostics for device: {hostname} ({ip})")

        # Ping Check
        if not self.ping_device(ip):
            context['ping'] = False
            self.logger.info(f"Device {hostname} ({ip}) is not responding to ping.")
        else:
            context['ping'] = True
            self.logger.debug(f"Device {hostname} ({ip}) is reachable via ping.")

        # Vendor Determination
        vendor = self.mac_lookup.get_vendor(mac) if mac != 'N/A' else "unknown"
        vendor_key = vendor.lower() if vendor else "unknown"
        context['vendor'] = vendor
        context['vendor_key'] = vendor_key
        self.logger.debug(f"Device {hostname} ({ip}) vendor determined as: {vendor}")

        return context


# Specific Diagnostics Classes
class RouterDiagnostics(CommonDiagnostics):
    """
    Perform diagnostics specific to routers.
    """
    DEVICE_TYPE = "Router"

    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        issues: List[DiagnoseIssue] = []
        ip: str = self.device.get("IP", "N/A")
        hostname: str = self.device.get("Hostname", "N/A")

        context: Dict = self.diagnose_common()
        if not context:
            return issues

        if not context.get('ping', False):
            issues.append(self.create_issue(f"{self.DEVICE_TYPE} is not responding to ping."))
            self.logger.info(f"{self.DEVICE_TYPE} {hostname} ({ip}) is not responding to ping.")
            return issues

        vendor_key: str = context.get('vendor_key', 'unknown')

        # Perform standard checks
        issues.extend(self.perform_standard_checks(ip, hostname, vendor_key))

        return issues


class PrinterDiagnostics(CommonDiagnostics):
    """
    Perform diagnostics specific to printers.
    """
    DEVICE_TYPE = "Printer"

    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        issues: List[DiagnoseIssue] = []
        ip: str = self.device.get("IP", "N/A")
        hostname: str = self.device.get("Hostname", "N/A")

        context: Dict = self.diagnose_common()
        if not context:
            return issues

        if not context.get('ping', False):
            issues.append(self.create_issue(f"{self.DEVICE_TYPE} is not responding to ping."))
            self.logger.info(f"{self.DEVICE_TYPE} {hostname} ({ip}) is not responding to ping.")
            return issues

        vendor_key: str = context.get('vendor_key', 'unknown')

        # Perform standard checks
        issues.extend(self.perform_standard_checks(ip, hostname, vendor_key))

        return issues


class PhoneDiagnostics(CommonDiagnostics):
    """
    Perform diagnostics specific to VoIP and mobile phones.
    """
    DEVICE_TYPE = "Phone"

    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        issues: List[DiagnoseIssue] = []
        ip: str = self.device.get("IP", "N/A")
        hostname: str = self.device.get("Hostname", "N/A")

        context: Dict = self.diagnose_common()
        if not context:
            return issues

        if not context.get('ping', False):
            issues.append(self.create_issue(f"{self.DEVICE_TYPE} is not responding to ping."))
            self.logger.info(f"{self.DEVICE_TYPE} {hostname} ({ip}) is not responding to ping.")
            return issues

        vendor_key: str = context.get('vendor_key', 'unknown')

        # Perform standard checks
        issues.extend(self.perform_standard_checks(ip, hostname, vendor_key))

        # Secure SIP Checks
        issues.extend(self.check_secure_sip(ip, vendor_key))

        return issues

    def check_secure_sip(self, ip: str, vendor_key: str) -> List[DiagnoseIssue]:
        """
        Check if SIP is configured securely on the phone (e.g., using TLS).
        """
        issues: List[DiagnoseIssue] = []
        try:
            self.logger.debug(f"Checking secure SIP configuration on {self.DEVICE_TYPE} {ip} for vendor '{vendor_key}'.")

            # Define secure SIP ports
            secure_sip_ports: List[int] = self.config.vendor_configs.get(vendor_key, {}).get('secure_sip_ports', [5061])

            for port in secure_sip_ports:
                if not self._is_port_open(ip, port):
                    issues.append(self.create_issue(f"SIP over TLS Port {port} is not open"))
                    self.logger.info(f"SIP over TLS Port {port} is not open on {self.DEVICE_TYPE} {ip}.")
                else:
                    self.logger.debug(f"SIP over TLS Port {port} is open on {self.DEVICE_TYPE} {ip}.")
        except Exception as e:
            self.logger.error(f"Error while checking secure SIP on {self.DEVICE_TYPE} {ip}: {e}")
        return issues


class SmartDiagnostics(CommonDiagnostics):
    """
    Perform diagnostics specific to smart devices, including IoT devices.
    """
    DEVICE_TYPE = "SmartDevice"

    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        issues: List[DiagnoseIssue] = []
        ip: str = self.device.get("IP", "N/A")
        hostname: str = self.device.get("Hostname", "N/A")

        context: Dict = self.diagnose_common()
        if not context:
            return issues

        if not context.get('ping', False):
            issues.append(self.create_issue(f"{self.DEVICE_TYPE} is not responding to ping."))
            self.logger.info(f"{self.DEVICE_TYPE} {hostname} ({ip}) is not responding to ping.")
            return issues

        vendor_key: str = context.get('vendor_key', 'unknown')

        # Perform standard checks
        issues.extend(self.perform_standard_checks(ip, hostname, vendor_key))

        return issues


class GameDiagnostics(CommonDiagnostics):
    """
    Perform diagnostics specific to game consoles like PlayStation, Xbox, and Nintendo Switch.
    """
    DEVICE_TYPE = "GameConsole"

    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        issues: List[DiagnoseIssue] = []
        ip: str = self.device.get("IP", "N/A")
        hostname: str = self.device.get("Hostname", "N/A")

        context: Dict = self.diagnose_common()
        if not context:
            return issues

        if not context.get('ping', False):
            issues.append(self.create_issue(f"{self.DEVICE_TYPE} is not responding to ping."))
            self.logger.info(f"{self.DEVICE_TYPE} {hostname} ({ip}) is not responding to ping.")
            return issues

        vendor_key: str = context.get('vendor_key', 'unknown')

        # Perform standard checks
        issues.extend(self.perform_standard_checks(ip, hostname, vendor_key))

        # Gaming-Specific Service Checks
        issues.extend(self.check_gaming_services(ip, vendor_key))

        return issues

    def check_gaming_services(self, ip: str, vendor_key: str) -> List[DiagnoseIssue]:
        """
        Check for vulnerabilities or misconfigurations in gaming-specific services.
        """
        issues: List[DiagnoseIssue] = []
        try:
            self.logger.debug(f"Checking gaming-specific services on {self.DEVICE_TYPE} {ip} for vendor '{vendor_key}'.")
            hostname: str = self.device.get("Hostname", "N/A")

            # Initialize an empty dictionary to collect all matching gaming services
            gaming_services: Dict[int, str] = {}
            for key, services in self.config.gaming_services.gaming_services.items():
                if key.lower() in vendor_key.lower():
                    gaming_services.update(services)

            for port, service in gaming_services.items():
                if self._is_port_open(ip, port):
                    self.logger.debug(f"Gaming service port {port} ({service}) is open on {self.DEVICE_TYPE} {ip}.")

                    # Example check: Ensure that sensitive gaming ports are not accessible over HTTP
                    if service in ["Xbox Live", "PlayStation Network", "Nintendo Switch"]:
                        url: str = f"http://{ip}:{port}"
                        try:
                            response: requests.Response = self._make_request(
                                protocol='http',
                                ip=ip,
                                hostname=hostname,
                                endpoint="/",
                                port=port,
                                timeout=3
                            )
                            if response.status_code == 200:
                                issues.append(self.create_issue(f"Gaming service port {port} ({service}) is accessible over HTTP"))
                                self.logger.info(
                                    f"Gaming service port {port} ({service}) is accessible over HTTP on {self.DEVICE_TYPE} {ip}."
                                )
                        except requests.RequestException as e:
                            self.logger.debug(f"HTTP request to {url} failed: {e}")
                            continue  # Unable to determine, proceed to next check
                else:
                    self.logger.debug(f"Gaming service port {port} ({service}) is closed on {self.DEVICE_TYPE} {ip}.")
        except Exception as e:
            self.logger.error(f"Error while checking gaming-specific services on {self.DEVICE_TYPE} {ip}: {e}")
        return issues


class ComputerDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics for laptops, desktops, and phones.
    """
    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        """
        Diagnose the client device for common issues.
        """
        return []


class OtherDeviceDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics for other types of devices.
    """
    def diagnose(self) -> Optional[List[DiagnoseIssue]]:
        """
        Diagnose other devices for reachability.
        """
        return []


# Network Scanner and Classifier
class NetworkScanner:
    """
    Scan the network and classify connected devices.
    """
    def __init__(self, args: argparse.Namespace, logger: logging.Logger, config: AppConfig, mac_lookup: Optional[MacVendorLookup] = None):
        """
        Initialize the network scanner with arguments and logger.
        """
        self.args = args
        self.logger = logger
        self.config = config
        self.mac_lookup = mac_lookup
        self.ipv6_enabled = args.ipv6  # New line to store IPv6 flag

    def execute(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Execute the network scanning and classification.
        """
        self.logger.info("Scanning the network for connected devices...")
        devices = self.scan_network()
        if not devices:
            self.logger.error("No devices found on the network.")
            sys.exit(1)

        return self.classify_devices(devices)

    def scan_network(self) -> List[Dict[str, str]]:
        """
        Scan the active subnets using nmap to discover devices.
        """
        try:
            # Dynamically determine the active subnets
            subnets = self.get_active_subnets()
            if not subnets:
                self.logger.error("No devices found on the network.")
                return []

            all_devices = []
            for subnet in subnets:
                self.logger.debug(f"Scanning subnet: {subnet}")
                # Determine if subnet is IPv6 based on presence of ':'
                if '/' in subnet and ':' in subnet:
                    # IPv6 subnet
                    scan_command = ['sudo', 'nmap', '-O', '-sV', '-T4', '-6', '-oX', '-', subnet]
                else:
                    # IPv4 subnet
                    scan_command = ['sudo', 'nmap', '-O', '-sV', '-T4', '-oX', '-', subnet]

                self.logger.debug(f"Executing command: {' '.join(scan_command)}")
                result = subprocess.run(scan_command, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.error(f"nmap scan failed for subnet {subnet}: {result.stderr.strip()}")
                    continue

                if not result.stdout.strip():
                    self.logger.error(f"nmap scan for subnet {subnet} returned empty output.")
                    if result.stderr:
                        self.logger.error(f"nmap stderr: {result.stderr.strip()}")
                    continue

                devices = self.parse_nmap_output(result.stdout)
                self.logger.debug(f"Found {len(devices)} devices in subnet {subnet}.")
                all_devices.extend(devices)
            return all_devices
        except FileNotFoundError:
            self.logger.error("nmap is not installed. Install it using your package manager.")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during network scan: {e}")
            return []

    def get_active_subnets(self) -> List[str]:
        """
        Determine the active subnets based on the system's non-loopback IPv4 and IPv6 addresses.
        Excludes virtual interfaces by default unless self.args.include_virtual is True.
        """
        subnets = []
        try:
            # Retrieve IPv4 subnets
            result_v4 = subprocess.run(['ip', '-4', 'addr'], capture_output=True, text=True, check=True)
            lines_v4 = result_v4.stdout.splitlines()
            current_iface = None
            for line in lines_v4:
                if line.startswith(' '):
                    if 'inet ' in line:
                        parts = line.strip().split()
                        ip_cidr = parts[1]  # e.g., '192.168.1.10/24'
                        ip, prefix = ip_cidr.split('/')
                        prefix = int(prefix)
                        subnet = self.calculate_subnet(ip, prefix)
                        # Exclude loopback subnet
                        if not subnet.startswith("127."):
                            # Determine the interface name from previous non-indented line
                            if current_iface and (
                                    not self.is_virtual_interface(current_iface) or self.args.include_virtual):
                                subnets.append(subnet)
                else:
                    # New interface
                    iface_info = line.split(':', 2)
                    if len(iface_info) >= 2:
                        current_iface = iface_info[1].strip().split('@')[0]

            # If IPv6 is enabled, retrieve IPv6 subnets
            if self.ipv6_enabled:
                result_v6 = subprocess.run(['ip', '-6', 'addr'], capture_output=True, text=True, check=True)
                lines_v6 = result_v6.stdout.splitlines()
                current_iface = None
                for line in lines_v6:
                    if line.startswith(' '):
                        if 'inet6 ' in line:
                            parts = line.strip().split()
                            ip_cidr = parts[1]  # e.g., '2001:db8::1/64'
                            ip, prefix = ip_cidr.split('/')
                            prefix = int(prefix)
                            subnet = self.calculate_subnet(ip, prefix)
                            # Exclude loopback subnet
                            if not subnet.startswith("::1"):
                                # Determine the interface name from previous non-indented line
                                if current_iface and (
                                        not self.is_virtual_interface(current_iface) or self.args.include_virtual):
                                    subnets.append(subnet)
                    else:
                        # New interface
                        iface_info = line.split(':', 2)
                        if len(iface_info) >= 2:
                            current_iface = iface_info[1].strip().split('@')[0]

            # Remove duplicates
            subnets = list(set(subnets))
            self.logger.debug(f"Active subnets: {subnets}")
            return subnets
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get IP addresses: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error determining active subnets: {e}")
            return []

    def is_virtual_interface(self, iface: str) -> bool:
        """
        Determine if the given interface is virtual based on its name.
        """
        virtual_prefixes = ['docker', 'br-', 'veth', 'virbr', 'vmnet', 'lo']
        for prefix in virtual_prefixes:
            if iface.startswith(prefix):
                self.logger.debug(f"Interface '{iface}' identified as virtual.")
                return True
        return False

    def calculate_subnet(self, ip: str, prefix: int) -> str:
        """
        Calculate the subnet in CIDR notation based on IP and prefix.
        """
        ip_parts = ip.split('.')
        if prefix == 24:
            subnet = f"{'.'.join(ip_parts[:3])}.0/24"
        elif prefix == 16:
            subnet = f"{'.'.join(ip_parts[:2])}.0.0/16"
        elif prefix == 8:
            subnet = f"{ip_parts[0]}.0.0.0/8"
        else:
            # For other prefixes, use the exact CIDR
            subnet = f"{ip}/{prefix}"
        self.logger.debug(f"Calculated subnet for IP {ip}/{prefix}: {subnet}")
        return subnet

    def parse_nmap_output(self, output: str) -> List[Dict[str, str]]:
        """
        Parse the XML output from nmap to extract device information.
        """
        devices = []
        try:
            root = ET.fromstring(output)
            for host in root.findall('host'):
                status = host.find('status')
                if status is not None and status.get('state') != 'up':
                    continue

                device = {}
                addresses = host.findall('address')
                for addr in addresses:
                    addr_type = addr.get('addrtype')
                    if addr_type == 'ipv4':
                        device['IP'] = addr.get('addr', 'N/A')
                    elif addr_type == 'ipv6':
                        device['IP'] = addr.get('addr', 'N/A')
                    elif addr_type == 'mac':
                        device['MAC'] = addr.get('addr', 'N/A')
                        device['Vendor'] = addr.get('vendor', 'Unknown')

                # Hostnames
                hostnames = host.find('hostnames')
                if hostnames is not None:
                    name = hostnames.find('hostname')
                    device['Hostname'] = name.get('name') if name is not None else "N/A"
                else:
                    device['Hostname'] = "N/A"

                # OS
                os_elem = host.find('os')
                if os_elem is not None:
                    os_matches = os_elem.findall('osmatch')
                    if os_matches:
                        device['OS'] = os_matches[0].get('name', 'Unknown')
                    else:
                        device['OS'] = "Unknown"
                else:
                    device['OS'] = "Unknown"

                # Ports
                ports = set()
                ports_elem = host.find('ports')
                if ports_elem is not None:
                    for port in ports_elem.findall('port'):
                        state = port.find('state')
                        if state is not None and state.get('state') == 'open':
                            service = port.find('service')
                            service_name = service.get('name', 'unknown') if service is not None else 'unknown'
                            portid = port.get('portid')
                            protocol = port.get('protocol')
                            port_info = f"{portid}/{protocol} {service_name}"
                            ports.add(port_info)
                device['Ports'] = ports

                devices.append(device)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse nmap XML output: {e}")
            self.logger.debug(f"nmap output was: {output}")
        except Exception as e:
            self.logger.error(f"Unexpected error during nmap parsing: {e}")
            self.logger.debug(f"nmap output was: {output}")
        return devices

    def classify_devices(self, devices: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Classify devices into different categories.
        """
        classifier = DeviceClassifier(self.logger, self.config, self.mac_lookup)
        classified = classifier.classify(devices)
        return classified


# Diagnostics Command
class DiagnosticsCommand(BaseCommand):
    """
    Perform automated network diagnostics.
    """
    def execute(self) -> None:
        """
        Execute the network diagnostics.
        """
        self.logger.info("Starting automated network diagnostics...")

        # Scan the network
        scanner = NetworkScanner(self.args, self.logger, self.config, self.mac_lookup)
        classified_devices = scanner.execute()

        # Save devices to a JSON file if output is requested
        if self.args.output_file:
            self.save_devices_to_file(classified_devices, self.args.output_file)

        # Display discovered devices
        self.display_devices(classified_devices)

        # Perform diagnostics based on device type unless discovery is enabled
        if not self.args.discovery:
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
                    for issue in issues:
                        if issue not in issues_found:
                            issues_found.append(issue)

        # Prepare rows by extracting values from each issue
        rows = [
            [
                issue.device_type,
                issue.hostname,
                issue.ip,
                issue.description
            ]
            for issue in issues_found
        ]

        # Display issues found
        if rows:
            columns = ["Device Type", "Hostname", "IP Address", "Issue"]
            self.print_table("Diagnostics Issues", columns, rows)
        else:
            self.logger.info("No issues detected during diagnostics.")

    def get_diagnostic_class(self, device_type: str, device: Dict[str, str]) -> GeneralDeviceDiagnostics:
        """
        Get the appropriate diagnostic class based on device type.
        """
        if device_type in ["Router", "Switch"]:
            return RouterDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)
        elif device_type == "Printer":
            return PrinterDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)
        elif device_type == "Phone":
            return PhoneDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)
        elif device_type == "Smart":
            return SmartDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)
        elif device_type == "Game":
            return GameDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)
        elif device_type == "Computer":
            return ComputerDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)
        else:
            return OtherDeviceDiagnostics(device_type, device, self.logger, self.args, self.config, self.mac_lookup)

    def save_devices_to_file(self, classified_devices: Dict[str, List[Dict[str, str]]], filename: str) -> None:
        """
        Save the classified devices to a JSON file.
        """
        class SetEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                return json.JSONEncoder.default(self, obj)

        try:
            with open(filename, 'w') as f:
                json.dump(classified_devices, f, indent=4, cls=SetEncoder)
            self.logger.info(f"Discovered devices saved to '{filename}'.")
        except Exception as e:
            self.logger.error(f"Failed to save devices to file: {e}")

    def display_devices(self, classified_devices: Dict[str, List[Dict[str, str]]]) -> None:
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


# Traffic Monitor Command
class TrafficMonitorCommand(BaseCommand):
    """
    Monitor network traffic to detect anomalies.
    """
    def __init__(self, args: argparse.Namespace, logger: logging.Logger, config: AppConfig, mac_lookup: Optional[MacVendorLookup] = None):
        super().__init__(args, logger, config, mac_lookup)

        # Initialize packet queue and processing thread
        self.packet_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_packets, daemon=True)
        self.processing_thread.start()

        # Extract configurable thresholds from args or use defaults
        self.enable_arp_spoof = True  # Can be made configurable if needed
        self.enable_dhcp_flood = True
        self.enable_port_scan = True
        self.enable_dns_exfiltration = True
        self.enable_bandwidth_abuse = True
        self.enable_icmp_flood = True
        self.enable_syn_flood = True
        self.enable_malformed_packets = True
        self.enable_rogue_dhcp = True
        self.enable_http_abuse = True

        # Data structures for tracking anomalies using deque for efficient time window management
        self.arp_table = {}
        self.dhcp_requests = defaultdict(lambda: deque())
        self.port_scan_attempts = defaultdict(lambda: set())
        self.dns_queries = defaultdict(int)
        self.bandwidth_usage = defaultdict(int)
        self.icmp_requests = defaultdict(lambda: deque())
        self.syn_requests = defaultdict(lambda: deque())
        self.rogue_dhcp_servers = {}
        self.http_requests = defaultdict(int)
        self.malformed_packets = defaultdict(int)

        # Configuration for thresholds
        self.dhcp_threshold = args.dhcp_threshold
        self.port_scan_threshold = args.port_scan_threshold
        self.dns_exfil_threshold = args.dns_exfil_threshold
        self.bandwidth_threshold = args.bandwidth_threshold
        self.icmp_threshold = args.icmp_threshold
        self.syn_threshold = args.syn_threshold
        self.http_threshold = args.http_threshold
        self.malformed_threshold = args.malformed_threshold
        self.rogue_dhcp_threshold = args.rogue_dhcp_threshold

        # Time windows
        self.one_minute = timedelta(minutes=1)
        self.one_hour = timedelta(hours=1)

    def execute(self) -> None:
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
            sniff(iface=interface, prn=lambda pkt: self.packet_queue.put(pkt), store=False)
        except PermissionError:
            self.logger.error("Permission denied. Run the script with elevated privileges.")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.info("Traffic monitoring stopped by user.")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Error during traffic monitoring: {e}")
            sys.exit(1)

    def process_packets(self):
        """
        Continuously process packets from the queue.
        """
        while True:
            packet = self.packet_queue.get()
            try:
                self.process_packet(packet)
            except Exception as e:
                self.logger.error(f"Error processing packet: {e}")
            finally:
                self.packet_queue.task_done()

    def process_packet(self, packet):
        """
        Process each captured packet to detect various anomalies.
        """
        current_time = datetime.now()

        if self.enable_arp_spoof and packet.haslayer(ARP):
            self.detect_arp_spoofing(packet)

        if self.enable_dhcp_flood and packet.haslayer(DHCP):
            self.detect_dhcp_flood(packet, current_time)

        if self.enable_port_scan and packet.haslayer(TCP):
            self.detect_port_scan(packet)

        if self.enable_dns_exfiltration and packet.haslayer(DNS) and packet.getlayer(DNS).qr == 0:
            self.detect_dns_exfiltration(packet, current_time)

        if self.enable_bandwidth_abuse and packet.haslayer(IP):
            self.detect_bandwidth_abuse(packet, current_time)

        if self.enable_icmp_flood and packet.haslayer(ICMP):
            self.detect_icmp_flood(packet, current_time)

        if self.enable_syn_flood and packet.haslayer(TCP):
            self.detect_syn_flood(packet, current_time)

        if self.enable_malformed_packets:
            self.detect_malformed_packets(packet, current_time)

        if self.enable_rogue_dhcp and packet.haslayer(DHCP):
            self.detect_rogue_dhcp(packet, current_time)

        if self.enable_http_abuse and packet.haslayer(TCP) and packet.haslayer(Raw):
            self.detect_http_abuse(packet, current_time)

    def detect_arp_spoofing(self, packet):
        """
        Detect ARP spoofing by monitoring ARP replies.
        """
        arp = packet.getlayer(ARP)
        if arp.op == 2:  # is-at (response)
            sender_ip = arp.psrc
            sender_mac = arp.hwsrc
            if sender_ip in self.arp_table:
                if self.arp_table[sender_ip] != sender_mac:
                    alert = (f"ARP Spoofing detected! IP {sender_ip} is-at {sender_mac} "
                             f"(was {self.arp_table[sender_ip]})")
                    self.report_anomaly(alert)
            self.arp_table[sender_ip] = sender_mac

    def detect_dhcp_flood(self, packet, current_time):
        """
        Detect DHCP flood attacks by monitoring excessive DHCP requests.
        """
        dhcp = packet.getlayer(DHCP)
        client_mac = packet.getlayer(Ether).src

        # Record the timestamp of the DHCP request
        self.dhcp_requests[client_mac].append(current_time)

        # Remove requests older than 1 minute
        while self.dhcp_requests[client_mac] and self.dhcp_requests[client_mac][0] < current_time - self.one_minute:
            self.dhcp_requests[client_mac].popleft()

        if len(self.dhcp_requests[client_mac]) > self.dhcp_threshold:
            alert = (f"DHCP Flood detected from {client_mac}: "
                     f"{len(self.dhcp_requests[client_mac])} requests in the last minute.")
            self.report_anomaly(alert)
            self.dhcp_requests[client_mac].clear()

    def detect_port_scan(self, packet):
        """
        Detect port scanning by monitoring connections to multiple ports from the same IP.
        """
        ip_layer = packet.getlayer(IP)
        tcp_layer = packet.getlayer(TCP)
        src_ip = ip_layer.src
        dst_port = tcp_layer.dport

        self.port_scan_attempts[src_ip].add(dst_port)

        if len(self.port_scan_attempts[src_ip]) > self.port_scan_threshold:
            alert = (f"Port Scan detected from {src_ip}: "
                     f"Accessed {len(self.port_scan_attempts[src_ip])} unique ports.")
            self.report_anomaly(alert)
            # Reset after alert to prevent repeated alerts
            self.port_scan_attempts[src_ip].clear()

    def detect_dns_exfiltration(self, packet, current_time):
        """
        Detect DNS exfiltration by monitoring excessive DNS queries.
        """
        dns_layer = packet.getlayer(DNS)
        ip_layer = packet.getlayer(IP)
        src_ip = ip_layer.src

        # Record the DNS query
        self.dns_queries[src_ip] += 1

        # Remove counts older than 1 hour
        # For precise time window, consider using deque with timestamps
        # Here, resetting count periodically as a simplification
        if self.dns_queries[src_ip] > self.dns_exfil_threshold:
            alert = (f"DNS Exfiltration detected from {src_ip}: "
                     f"{self.dns_queries[src_ip]} DNS queries.")
            self.report_anomaly(alert)
            # Reset after alert
            self.dns_queries[src_ip] = 0

    def detect_bandwidth_abuse(self, packet, current_time):
        """
        Detect bandwidth abuse by monitoring data usage per client.
        """
        ip_layer = packet.getlayer(IP)
        src_ip = ip_layer.src
        packet_size = len(packet)

        self.bandwidth_usage[src_ip] += packet_size

        if self.bandwidth_usage[src_ip] > self.bandwidth_threshold:
            alert = (f"Bandwidth Abuse detected from {src_ip}: "
                     f"{self.bandwidth_usage[src_ip]} bytes in the last minute.")
            self.report_anomaly(alert)
            # Reset after alert
            self.bandwidth_usage[src_ip] = 0

    def detect_icmp_flood(self, packet, current_time):
        """
        Detect ICMP flood attacks by monitoring excessive ICMP requests.
        """
        src_ip = packet.getlayer(IP).src

        # Record the timestamp of the ICMP request
        self.icmp_requests[src_ip].append(current_time)

        # Remove requests older than 1 minute
        while self.icmp_requests[src_ip] and self.icmp_requests[src_ip][0] < current_time - self.one_minute:
            self.icmp_requests[src_ip].popleft()

        if len(self.icmp_requests[src_ip]) > self.icmp_threshold:
            alert = (f"ICMP Flood detected from {src_ip}: "
                     f"{len(self.icmp_requests[src_ip])} ICMP packets in the last minute.")
            self.report_anomaly(alert)

    def detect_syn_flood(self, packet, current_time):
        """
        Detect SYN flood attacks by monitoring excessive TCP SYN packets.
        """
        tcp_layer = packet.getlayer(TCP)
        if tcp_layer.flags & 0x02:  # SYN flag
            src_ip = packet.getlayer(IP).src

            # Record the timestamp of the SYN packet
            self.syn_requests[src_ip].append(current_time)

            # Remove SYNs older than 1 minute
            while self.syn_requests[src_ip] and self.syn_requests[src_ip][0] < current_time - self.one_minute:
                self.syn_requests[src_ip].popleft()

            if len(self.syn_requests[src_ip]) > self.syn_threshold:
                alert = (f"SYN Flood detected from {src_ip}: "
                         f"{len(self.syn_requests[src_ip])} SYN packets in the last minute.")
                self.report_anomaly(alert)
                # Reset after alert
                self.syn_requests[src_ip].clear()

    def detect_malformed_packets(self, packet, current_time):
        """
        Detect malformed packets that do not conform to protocol standards.
        """
        try:
            # Attempt to access packet layers to validate
            if packet.haslayer(IP):
                ip_layer = packet.getlayer(IP)
                if packet.haslayer(TCP):
                    tcp_layer = packet.getlayer(TCP)
                    _ = tcp_layer.flags  # Access a TCP field
                elif packet.haslayer(UDP):
                    udp_layer = packet.getlayer(UDP)
                    _ = udp_layer.sport  # Access a UDP field
        except Exception:
            src_ip = packet.getlayer(IP).src if packet.haslayer(IP) else "Unknown"
            self.malformed_packets[src_ip] += 1
            if self.malformed_packets[src_ip] > self.malformed_threshold:
                alert = (f"Malformed packets detected from {src_ip}: "
                         f"{self.malformed_packets[src_ip]} malformed packets.")
                self.report_anomaly(alert)
                # Reset after alert
                self.malformed_packets[src_ip] = 0

    def detect_rogue_dhcp(self, packet, current_time):
        """
        Detect rogue DHCP servers by monitoring DHCP OFFER messages.
        """
        dhcp = packet.getlayer(DHCP)
        if dhcp.options and any(option[0] == 'message-type' and option[1] == 2 for option in dhcp.options):
            # DHCP Offer
            server_ip = packet.getlayer(IP).src
            self.rogue_dhcp_servers[server_ip] = current_time

            # Remove entries older than 1 hour
            keys_to_remove = [ip for ip, ts in self.rogue_dhcp_servers.items() if ts < current_time - self.one_hour]
            for ip in keys_to_remove:
                del self.rogue_dhcp_servers[ip]

            if len(self.rogue_dhcp_servers) > self.rogue_dhcp_threshold:
                alert = f"Rogue DHCP Server detected: {server_ip}"
                self.report_anomaly(alert)
                # Reset after alert
                self.rogue_dhcp_servers.clear()

    def detect_http_abuse(self, packet, current_time):
        """
        Detect excessive HTTP requests that may indicate scraping or DoS.
        """
        src_ip = packet.getlayer(IP).src
        self.http_requests[src_ip] += 1

        # Reset periodically; precise time window handling can be added
        if self.http_requests[src_ip] > self.http_threshold:
            alert = (f"Excessive HTTP requests detected from {src_ip}: "
                     f"{self.http_requests[src_ip]} requests in the last minute.")
            self.report_anomaly(alert)
            # Reset after alert
            self.http_requests[src_ip] = 0

    def report_anomaly(self, message):
        """
        Report detected anomalies based on the verbosity level.
        """
        if console:
            console.print(message, style="bold red")
        else:
            print(message)


# System Information Command
class SystemInfoCommand(BaseCommand):
    """
    Gather and display system network information, including IPv4 and IPv6.
    """

    def execute(self) -> None:
        """
        Execute the system information gathering.
        """
        self.logger.info("Gathering system network information...")
        ip_info_v4 = self.get_ip_info('inet')
        ip_info_v6 = self.get_ip_info('inet6')
        routing_info_v4 = self.get_routing_info('inet')
        routing_info_v6 = self.get_routing_info('inet6')
        dns_info = self.get_dns_info()

        traceroute_info_v4 = None
        traceroute_info_v6 = None

        for traceroute_target in self.args.traceroute:
            if ':' in traceroute_target:
                traceroute_info_v6 = self.perform_traceroute(traceroute_target)
            else:
                traceroute_info_v4 = self.perform_traceroute(traceroute_target)

        self.display_system_info(ip_info_v4, ip_info_v6, routing_info_v4, routing_info_v6, dns_info, traceroute_info_v4, traceroute_info_v6)

    def get_ip_info(self, family: str = 'inet') -> str:
        """
        Retrieve IP configuration using the 'ip addr' command for the specified family.
        """
        self.logger.debug(f"Retrieving IP configuration for {family}...")
        try:
            if family == 'inet':
                cmd = ['ip', '-4', 'addr', 'show']
            elif family == 'inet6':
                cmd = ['ip', '-6', 'addr', 'show']
            else:
                self.logger.error(f"Unknown IP family: {family}")
                return ""

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            if family == 'inet6':
                self.logger.info("IPv6 is not supported on this system.")
            else:
                self.logger.error(f"Failed to get IP information for {family}: {e}")
            return ""

    def get_routing_info(self, family: str = 'inet') -> str:
        """
        Retrieve routing table using the 'ip route' command for the specified family.
        """
        self.logger.debug(f"Retrieving routing table for {family}...")
        try:
            if family == 'inet6':
                cmd = ['ip', '-6', 'route', 'show']
            else:
                cmd = ['ip', 'route', 'show']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            if family == 'inet6':
                self.logger.info("IPv6 routing information is not available on this system.")
            else:
                self.logger.error(f"Failed to get routing information for {family}: {e}")
            return ""

    def get_dns_info(self) -> Dict[str, List[str]]:
        """
        Retrieve DNS server information, handling systemd-resolved if resolv.conf points to localhost.
        Returns a dictionary mapping network interfaces to their DNS servers for both IPv4 and IPv6.
        """
        self.logger.debug("Retrieving DNS servers...")
        dns_info = {}
        try:
            with open('/etc/resolv.conf', 'r') as f:
                resolv_conf_dns_v4 = []
                resolv_conf_dns_v6 = []
                for line in f:
                    if line.startswith('nameserver'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            dns_server = parts[1]
                            if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', dns_server):
                                resolv_conf_dns_v4.append(dns_server)
                            elif re.match(r'^([0-9a-fA-F]{0,4}:){1,7}[0-9a-fA-F]{0,4}$', dns_server):
                                resolv_conf_dns_v6.append(dns_server)

                if resolv_conf_dns_v4:
                    dns_info['resolv.conf (IPv4)'] = resolv_conf_dns_v4
                    self.logger.debug(f"IPv4 DNS servers from resolv.conf: {resolv_conf_dns_v4}")
                if resolv_conf_dns_v6:
                    dns_info['resolv.conf (IPv6)'] = resolv_conf_dns_v6
                    self.logger.debug(f"IPv6 DNS servers from resolv.conf: {resolv_conf_dns_v6}")

                # Check if resolv.conf points to localhost for IPv4 or IPv6
                localhost_v4 = any(ns.startswith('127.') for ns in resolv_conf_dns_v4)
                localhost_v6 = any(ns.startswith('::1') for ns in resolv_conf_dns_v6)

                if localhost_v4 or localhost_v6:
                    self.logger.debug(
                        "resolv.conf points to localhost. Querying systemd-resolved for real DNS servers.")
                    try:
                        result = subprocess.run(['resolvectl', 'status'], capture_output=True, text=True, check=True)
                        # Use regex to find DNS servers for each interface
                        interface_pattern = re.compile(r'Link\s+\d+\s+\(([^)]+)\)')
                        dns_server_pattern = re.compile(r'DNS Servers:\s+(.+)')

                        current_iface = None
                        for line in result.stdout.splitlines():
                            iface_match = interface_pattern.match(line)
                            if iface_match:
                                current_iface = iface_match.group(1).strip()
                                self.logger.debug(f"Detected interface: {current_iface}")
                            else:
                                dns_match = dns_server_pattern.search(line)
                                if dns_match and current_iface:
                                    servers = dns_match.group(1).strip().split()
                                    ipv4_servers = [s for s in servers if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', s)]
                                    ipv6_servers = [s for s in servers if
                                                    re.match(r'^([0-9a-fA-F]{0,4}:){1,7}[0-9a-fA-F]{0,4}$', s)]
                                    if ipv4_servers:
                                        dns_info.setdefault(f'{current_iface} (IPv4)', []).extend(ipv4_servers)
                                        self.logger.debug(f"Found IPv4 DNS servers for {current_iface}: {ipv4_servers}")
                                    if ipv6_servers:
                                        dns_info.setdefault(f'{current_iface} (IPv6)', []).extend(ipv6_servers)
                                        self.logger.debug(f"Found IPv6 DNS servers for {current_iface}: {ipv6_servers}")
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Failed to run resolvectl: {e}")
                    except FileNotFoundError:
                        self.logger.error("resolvectl command not found. Ensure systemd-resolved is installed.")
        except Exception as e:
            self.logger.error(f"Failed to read DNS information: {e}")

        # Remove duplicates while preserving order for each interface
        for iface, servers in dns_info.items():
            seen = set()
            unique_servers = []
            for dns in servers:
                if dns not in seen:
                    seen.add(dns)
                    unique_servers.append(dns)
            dns_info[iface] = unique_servers

        return dns_info

    def perform_traceroute(self, target: str) -> Optional[str]:
        """
        Perform a traceroute to the specified target and return the output.
        """
        self.logger.info(f"Performing traceroute to {target}...")
        try:
            # Determine if target is IPv6 based on being enclosed in []
            if ':' in target:
                family = 'inet6'
                cmd = ['traceroute', '-n', '-6', target]
            else:
                family = 'inet'
                cmd = ['traceroute', '-n', target]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.debug(f"Traceroute ({family}) completed successfully.")
            return result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            if family == 'inet6' and 'Address family for hostname not supported' in error_msg:
                self.logger.info(f"Traceroute to {target} for {family} failed: IPv6 is not supported.")
            else:
                self.logger.error(f"Traceroute to {target} for {family} failed: {error_msg}")
        except FileNotFoundError:
            self.logger.info("traceroute command not found. Install it using your package manager.")
        except Exception as e:
            self.logger.error(f"Unexpected error during traceroute for {family}: {e}")
        return None

    def display_system_info(
            self,
            ip_info_v4: str,
            ip_info_v6: str,
            routing_info_v4: str,
            routing_info_v6: str,
            dns_info: Dict[str, List[str]],
            traceroute_info_v4: Optional[str],
            traceroute_info_v6: Optional[str]
    ) -> None:
        """
        Display the gathered system information for both IPv4 and IPv6.
        """
        if console:
            if ip_info_v4:
                console.print(Panel("[bold underline]Configuration (IPv4)[/bold underline]", style="cyan"))
                console.print(ip_info_v4)
            if ip_info_v6:
                console.print(Panel("[bold underline]Configuration (IPv6)[/bold underline]", style="cyan"))
                console.print(ip_info_v6)

            if routing_info_v4:
                console.print(Panel("[bold underline]Routing Table (IPv4)[/bold underline]", style="cyan"))
                console.print(routing_info_v4)
            if routing_info_v6:
                console.print(Panel("[bold underline]Routing Table (IPv6)[/bold underline]", style="cyan"))
                console.print(routing_info_v6)

            if dns_info:
                console.print(Panel("[bold underline]DNS Servers[/bold underline]", style="cyan"))
                for iface, dns_servers in dns_info.items():
                    console.print(f"[bold]{iface}:[/bold]")
                    for dns in dns_servers:
                        console.print(f"  - {dns}")

            if traceroute_info_v4:
                console.print(Panel("[bold underline]Traceroute (IPv4)[/bold underline]", style="cyan"))
                console.print(traceroute_info_v4)
            if traceroute_info_v6:
                console.print(Panel("[bold underline]Traceroute (IPv6)[/bold underline]", style="cyan"))
                console.print(traceroute_info_v6)
        else:
            if ip_info_v4:
                print("\n=== Configuration (IPv4) ===")
                print(ip_info_v4)
            if ip_info_v6:
                print("\n=== Configuration (IPv6) ===")
                print(ip_info_v6)

            if routing_info_v4:
                print("\n=== Routing Table (IPv4) ===")
                print(routing_info_v4)
            if routing_info_v6:
                print("\n=== Routing Table (IPv6) ===")
                print(routing_info_v6)

            if dns_info:
                print("\n=== DNS Servers ===")
                for iface, dns_servers in dns_info.items():
                    print(f"\n--- {iface} ---")
                    for dns in dns_servers:
                        print(f"- {dns}")

            if traceroute_info_v4:
                print("\n=== Traceroute (IPv4) ===")
                print(traceroute_info_v4)
            if traceroute_info_v6:
                print("\n=== Traceroute (IPv6) ===")
                print(traceroute_info_v6)


# Wifi Diagnostics Command
class WifiDiagnosticsCommand(BaseCommand):
    """
    Perform WiFi diagnostics and analyze available networks.
    """
    def execute(self) -> None:
        """
        Execute WiFi diagnostics.
        """
        self.logger.info("Starting WiFi diagnostics...")
        wifi_networks = self.scan_wifi_networks()
        if not wifi_networks:
            self.logger.error("No WiFi networks found.")
            sys.exit(1)

        target_networks = None
        if self.args.ssid:
            target_ssid = self.args.ssid
            self.logger.info(f"Performing diagnostics for SSID: {target_ssid}")
            target_networks = self.get_networks_by_ssid(wifi_networks, target_ssid)
            if not target_networks:
                self.logger.error(f"SSID '{target_ssid}' not found among available networks.")
                sys.exit(1)

        self.logger.info("Performing generic WiFi diagnostics.")
        issues = self.diagnose_wifi(wifi_networks, target_networks)

        self.display_issues(issues)

    def scan_wifi_networks(self) -> List[Dict[str, str]]:
        """
        Scan available WiFi networks using nmcli in terse mode for reliable parsing.
        """
        self.logger.debug("Scanning available WiFi networks using nmcli...")
        try:
            scan_command = ['nmcli', '-t', '-f', 'SSID,SIGNAL,CHAN,SECURITY', 'device', 'wifi', 'list', '--rescan', 'yes']
            if self.args.interface:
                scan_command.extend(['ifname', self.args.interface])
                self.logger.debug(f"Using interface: {self.args.interface}")
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
        Parse the output from nmcli in terse mode to handle SSIDs with colons and spaces.
        """
        networks = []
        lines = output.strip().split('\n')
        for line in lines:
            # Split only on the first three colons to handle SSIDs with colons
            parts = line.split(':', 3)
            if len(parts) < 4:
                continue  # Incomplete information
            ssid, signal, channel, security = parts[:4]
            networks.append({
                'SSID': ssid.strip(),
                'Signal': signal.strip(),
                'Channel': channel.strip(),
                'Security': security.strip()
            })
        return networks

    def get_networks_by_ssid(self, networks: List[Dict[str, str]], ssid: str) -> List[Dict[str, str]]:
        """
        Retrieve a specific network's details by its SSID.
        """
        target_networks = []
        for network in networks:
            if network['SSID'] == ssid:
                target_networks.append(network)
        return target_networks

    def diagnose_wifi(self, networks: List[Dict[str, str]], target_networks: List[Dict[str, str]] = None) -> List[WifiIssue]:
        """
        Perform generic diagnostics across all available WiFi networks.
        """
        issues = []

        if not target_networks:
            target_networks = networks

        # Extract unique channels from target_networks
        unique_channels = set()
        for net in target_networks:
            try:
                channel = int(net['Channel'])
                unique_channels.add(channel)
            except ValueError:
                self.logger.error(f"Invalid channel number for network '{net['SSID']}'. Skipping this network.")

        # Analyze each unique channel for interference
        unique_channels = sorted(unique_channels)
        for channel in unique_channels:
            channel_issue = self.analyze_channel_interference(channel, networks)
            if channel_issue:
                issues.append(channel_issue)

        # Check for open (unsecured) networks
        open_networks = [net for net in target_networks if net['Security'].upper() in ['OPEN', '--']]
        for net in open_networks:
            issues.append(WifiIssue(
                issue_type="Authentication",
                location=net['SSID'],
                ip_address="N/A",
                description=f"Open and unsecured network on channel {net['Channel']}."
            ))

        # Check for networks with weak signals
        weak_networks = [net for net in target_networks if self.safe_int(net['Signal']) < self.args.signal_threshold]
        for net in weak_networks:
            issues.append(WifiIssue(
                issue_type="Signal",
                location=net['SSID'],
                ip_address="N/A",
                description=f"Low signal strength: {net['Signal']}% on channel {net['Channel']}."
            ))

        return issues

    def analyze_channel_interference(self, channel: int, networks: List[Dict[str, str]]) -> Optional[WifiIssue]:
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
            return WifiIssue(
                issue_type="Interference",
                location=f"Channel {channel}",
                ip_address="",
                description=f"High number of networks ({count}) on this channel causing interference."
            )
        return None

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

    def display_issues(self, issues: List[WifiIssue]):
        """
        Display the identified WiFi diagnostic issues.
        """
        if not issues:
            self.logger.info("No WiFi issues detected.")
            return

        rows = [
            [
                issue.issue_type,
                issue.location,
                issue.ip_address,
                issue.description
            ]
            for issue in issues
        ]

        columns = ["Issue Type", "SSID/Channel", "IP Address", "Description"]
        self.print_table("WiFi Diagnostics Issues", columns, rows)

    def safe_int(self, value: str) -> int:
        """
        Safely convert a string to an integer, returning 0 on failure.
        """
        try:
            return int(value)
        except ValueError:
            return 0


# Device Classifier Class
class DeviceClassifier:
    """
    Classify devices based on their attributes.
    """
    def __init__(self, logger: logging.Logger, config: AppConfig, mac_lookup: Optional[MacVendorLookup] = None):
        """
        Initialize the DeviceClassifier with a logger and configuration.
        """
        self.logger = logger
        self.config = config
        self.mac_lookup = mac_lookup

    def classify(self, devices: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Classify the list of devices into categories.
        """
        classified = {
            "Router": [],
            "Switch": [],
            "Printer": [],
            "Phone": [],
            "Smart": [],
            "Game": [],
            "Computer": [],
            "Unknown": []
        }

        for device in devices:
            device_type = self.infer_device_type(device)
            classified[device_type].append(device)

        # Remove empty categories
        classified = {k: v for k, v in classified.items() if v}

        return classified

    def infer_device_type(self, device: Dict[str, str]) -> str:
        """
        Infer the device type based on its attributes.
        """
        matched_device_type = "Unknown"
        highest_priority = float('inf')

        for device_type in sorted(self.config.device_types.device_types, key=lambda dt: dt.priority):
            if device_type.matches(device, self.mac_lookup):
                if device_type.priority < highest_priority:
                    matched_device_type = device_type.name
                    highest_priority = device_type.priority
                    self.logger.debug(
                        f"Device {device.get('MAC', 'N/A')} matched {matched_device_type} with priority {device_type.priority}.")

        if matched_device_type == "Unknown":
            self.logger.debug(f"Device {device.get('MAC', 'N/A')} classified as Unknown.")
        else:
            self.logger.debug(f"Device {device.get('MAC', 'N/A')} classified as {matched_device_type}.")

        return matched_device_type


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

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # Subparser for system-info
    sys_info_parser = subparsers.add_parser(
        'system-info',
        aliases=['si'],
        help='Display detailed network information about the system.'
    )
    sys_info_parser.add_argument(
        '--traceroute', '-t',
        type=str,
        nargs='*',
        default=['8.8.8.8', '2001:4860:4860::8888'],
        help='Perform a traceroute to the specified target.'
    )

    # Subparser for diagnose
    diagnose_parser = subparsers.add_parser(
        'diagnose',
        aliases=['dg'],
        help='Perform automated diagnostics on the network.'
    )
    diagnose_parser.add_argument(
        '--include-virtual', '-V',
        action='store_true',
        help='Include virtual devices in diagnostics.'
    )
    diagnose_parser.add_argument(
        '--nikto', '-N',
        action='store_true',
        help='Run Nikto web server scanner on discovered devices.'
    )
    diagnose_parser.add_argument(
        '--credentials', '-C',
        action='store_true',
        help='Check for default credentials on discovered devices.'
    )
    diagnose_parser.add_argument(
        '--discovery', '-d',
        action='store_true',
        help='Perform network discovery to find devices only.'
    )
    diagnose_parser.add_argument(
        '--ipv6', '-6',
        action='store_true',
        help='Enable IPv6 scanning.'
    )
    diagnose_parser.add_argument(
        '--output-file', '-o',
        type=str,
        help='File to store discovered devices.'
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
    # Traffic Monitor Command Options
    traffic_monitor_parser.add_argument(
        '--dhcp-threshold',
        type=int,
        default=100,
        help='Set DHCP flood threshold (default: 100).'
    )
    traffic_monitor_parser.add_argument(
        '--port-scan-threshold',
        type=int,
        default=50,
        help='Set port scan threshold (default: 50).'
    )
    traffic_monitor_parser.add_argument(
        '--dns-exfil-threshold',
        type=int,
        default=1000,
        help='Set DNS exfiltration threshold (default: 1000).'
    )
    traffic_monitor_parser.add_argument(
        '--bandwidth-threshold',
        type=int,
        default=1000000,
        help='Set bandwidth abuse threshold in bytes per minute (default: 1000000).'
    )
    traffic_monitor_parser.add_argument(
        '--icmp-threshold',
        type=int,
        default=500,
        help='Set ICMP flood threshold (default: 500).'
    )
    traffic_monitor_parser.add_argument(
        '--syn-threshold',
        type=int,
        default=1000,
        help='Set SYN flood threshold (default: 1000).'
    )
    traffic_monitor_parser.add_argument(
        '--http-threshold',
        type=int,
        default=1000,
        help='Set HTTP abuse threshold (default: 1000).'
    )
    traffic_monitor_parser.add_argument(
        '--malformed-threshold',
        type=int,
        default=50,
        help='Set malformed packets threshold (default: 50).'
    )
    traffic_monitor_parser.add_argument(
        '--rogue-dhcp-threshold',
        type=int,
        default=1,
        help='Set rogue DHCP server threshold (default: 1).'
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
    wifi_parser.add_argument(
        '--interface', '-i',
        type=str,
        required=False,
        help='Network interface to scan (e.g., wlan0, wlp3s0).'
    )
    wifi_parser.add_argument(
        '--signal-threshold', '-m',
        type=int,
        default=50,
        help='Minimum signal strength threshold (default: 50).'
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
    'traffic-monitor': TrafficMonitorCommand,
    'tm': TrafficMonitorCommand,
    'wifi': WifiDiagnosticsCommand,
    'wf': WifiDiagnosticsCommand,
}


# Main Function
def main() -> None:
    """
    Main function to orchestrate the network diagnostic process.
    """
    args = parse_arguments()
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    config = AppConfig()

    if not RICH_AVAILABLE:
        logger.warning("Rich library not found. Install it using 'pip install rich' for better output formatting. Really recommended.")

    if args.command in ['diagnose', 'dg']:
        mac_lookup = MacVendorLookup(logger)
    else:
        mac_lookup = None  # Not needed for other commands

    # Instantiate and execute the appropriate command
    command_class = COMMAND_CLASSES.get(args.command)
    if not command_class:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

    command = command_class(args, logger, config, mac_lookup)
    command.execute()

    logger.info("Network diagnostics completed successfully.")


if __name__ == "__main__":
    main()
