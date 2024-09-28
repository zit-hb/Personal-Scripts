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
# - scapy (install via: pip install scapy)
# - requests (install via: pip install requests)
# - nmap (install via: apt install nmap)
# - nmcli (install via: apt install network-manager)
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
import ssl
import json
import time
import threading
import queue
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

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

# Define common sensitive endpoints
COMMON_SENSITIVE_ENDPOINTS = [
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
]

# Define common sensitive firmware endpoints
COMMON_SENSITIVE_FIRMWARE_ENDPOINTS = [
    "/firmware/status",
    "/update/status",
    "/api/firmware",
]

# Define per-vendor additional sensitive endpoints
VENDOR_ADDITIONAL_SENSITIVE_ENDPOINTS = {
    'fritz!box': [
        "/admin/config.php",
        "/diag_wps.html",
    ],
    'asus': [
        "/admin/config.php",
    ],
    'netgear': [
        "/cgi-bin/fwupdate",
        "/status/wps",
    ],
    'tp-link': [],
    'd-link': [
        "/status.html",
    ],
    'linksys': [
        "/status.html",
    ],
    'belkin': [
        "/cgi-bin/admin/config",
        "/status.cgi",
    ],
    'synology': [
        "/webman/index.cgi",
        "/status.cgi",
    ],
    'ubiquiti': [
        "/cgi-bin/status.cgi",
    ],
    'mikrotik': [
        "/login",
    ],
    'zyxel': [
        "/cgi-bin/admin/config",
    ],
    'huawei': [
        "/cgi-bin/hwcfg.cgi",
    ],
    'apple': [
        "/airport/admin",
    ],
}

# Define firmware endpoints per vendor
VENDOR_FIRMWARE_ENDPOINTS = {
    'fritz!box': [
        "/status/firmware_update",
    ],
    'asus': [
        "/admin/firmware",
    ],
    'netgear': [
        "/cgi-bin/fwupdate",
    ],
    'tp-link': [
        "/api/firmware",
    ],
    'd-link': [
        "/firmware/status",
    ],
    'linksys': [
        "/update/status",
    ],
    'belkin': [
        "/firmware/status",
    ],
    'synology': [
        "/update/status",
    ],
    'ubiquiti': [
        "/api/firmware",
    ],
    'mikrotik': [
        "/upgrade",
    ],
    'zyxel': [
        "/firmware/update",
    ],
    'huawei': [
        "/firmware/status",
    ],
    'apple': [
        "/update/status",
    ],
}

# Combine to create vendor_configs
VENDOR_CONFIGS = {
    vendor: {
        'sensitive_endpoints': COMMON_SENSITIVE_ENDPOINTS + endpoints,
        'firmware_endpoints': VENDOR_FIRMWARE_ENDPOINTS.get(vendor, [])
    }
    for vendor, endpoints in VENDOR_ADDITIONAL_SENSITIVE_ENDPOINTS.items()
}


# Base class for all commands
class BaseCommand(ABC):
    def __init__(self, args, logger):
        """
        Initialize the BaseCommand with arguments and logger.
        """
        self.args = args
        self.logger = logger

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
                    json.dump({"timestamp": time.time(), "data": data}, f)
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
        oui_dict = {}
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


# General Device Diagnostics Class
class GeneralDeviceDiagnostics(ABC):
    """
    Perform diagnostics for various types of devices.
    """

    def __init__(self, device: Dict[str, str], logger: logging.Logger):
        """
        Initialize the GeneralDeviceDiagnostics with device info and logger.
        """
        self.device = device
        self.logger = logger

    @abstractmethod
    def diagnose(self) -> Optional[List[Dict[str, str]]]:
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
class RouterDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics specific to routers and servers.
    """

    def __init__(self, device: Dict[str, str], logger: logging.Logger):
        super().__init__(device, logger)
        self.mac_lookup = MacVendorLookup(logger)

        # Assign the pre-defined vendor configurations
        self.vendor_configs = VENDOR_CONFIGS

    def diagnose(self) -> Optional[List[Dict[str, str]]]:
        """
        Diagnose the device for common router-related issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")
        os_info = self.device.get("OS", "Unknown").lower()
        mac = self.device.get("MAC", "").upper()

        if ip == "N/A":
            self.logger.debug(f"{hostname} IP is N/A. Skipping diagnostics.")
            return issues

        self.logger.debug(f"Starting diagnostics for router: {hostname} ({ip})")

        # Check if the router is reachable via ping
        if not self.ping_device(ip):
            issues.append({
                "DeviceType": "Router",
                "Hostname": hostname,
                "IP": ip,
                "Description": "Router is not responding to ping."
            })
            self.logger.warning(f"Router {hostname} ({ip}) is not responding to ping.")
            # If the router is not reachable, further port checks might be futile
            return issues
        else:
            self.logger.debug(f"Router {hostname} ({ip}) is reachable via ping.")

        # Determine the vendor using MAC address
        vendor = self.mac_lookup.get_vendor(mac) if mac != 'N/A' else "unknown"
        vendor_key = vendor.lower() if vendor else "unknown"
        self.logger.debug(f"Router {hostname} ({ip}) vendor determined as: {vendor}")

        # Check if SSH port 22 is unexpectedly open
        if any(os in os_info for os in ["linux", "unix"]):
            if not self.check_port(ip, 22):
                issues.append({
                    "DeviceType": "Router",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": "SSH Port 22 Unreachable"
                })
                self.logger.info(f"SSH Port 22 is unreachable on router {hostname} ({ip}).")
            else:
                self.logger.debug(f"SSH Port 22 is open on router {hostname} ({ip}).")

        # Check if Telnet port 23 is open (should typically be closed)
        if self.check_port(ip, 23):
            issues.append({
                "DeviceType": "Router",
                "Hostname": hostname,
                "IP": ip,
                "Description": "Telnet Port 23 is Open"
            })
            self.logger.info(f"Telnet Port 23 is open on router {hostname} ({ip}).")
        else:
            self.logger.debug(f"Telnet Port 23 is closed on router {hostname} ({ip}).")

        # Check if UPnP port 1900 is open (can be a security risk)
        if self.check_port(ip, 1900):
            issues.append({
                "DeviceType": "Router",
                "Hostname": hostname,
                "IP": ip,
                "Description": "UPnP Port 1900 is Open"
            })
            self.logger.info(f"UPnP Port 1900 is open on router {hostname} ({ip}).")
        else:
            self.logger.debug(f"UPnP Port 1900 is closed on router {hostname} ({ip}).")

        # Check common remote management ports (e.g., 8080, 8443)
        remote_management_ports = [8080, 8443]
        for port in remote_management_ports:
            if self.check_port(ip, port):
                issues.append({
                    "DeviceType": "Router",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"Remote Management Port {port} is Open"
                })
                self.logger.info(f"Remote Management Port {port} is open on router {hostname} ({ip}).")
            else:
                self.logger.debug(f"Remote Management Port {port} is closed on router {hostname} ({ip}).")

        # Check if WPS (Wi-Fi Protected Setup) is enabled by attempting to detect it via HTTP requests
        wps_status = self.check_wps_status(ip, vendor_key)
        if wps_status is not None:
            if wps_status:
                issues.append({
                    "DeviceType": "Router",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": "WPS is Enabled"
                })
                self.logger.info(f"WPS is enabled on router {hostname} ({ip}).")
            else:
                self.logger.debug(f"WPS is disabled on router {hostname} ({ip}).")

        # Check for publicly accessible diagnostic or backup pages based on vendor
        exposed_pages = self.check_publicly_accessible_pages(ip, vendor_key)
        for page in exposed_pages:
            issues.append({
                "DeviceType": "Router",
                "Hostname": hostname,
                "IP": ip,
                "Description": f"Publicly Accessible Page Detected: {page}"
            })
            self.logger.info(f"Publicly accessible page '{page}' detected on router {hostname} ({ip}).")

        # Check for firmware updates availability based on vendor
        firmware_update_needed = self.check_firmware_update(ip, vendor_key)
        if firmware_update_needed:
            issues.append({
                "DeviceType": "Router",
                "Hostname": hostname,
                "IP": ip,
                "Description": "Firmware Update Available"
            })
            self.logger.info(f"Firmware update is available for router {hostname} ({ip}).")
        else:
            self.logger.debug(f"No firmware updates available for router {hostname} ({ip}).")

        # Additionally perform web-related checks and merge the results into issues
        web_diagnostics = WebEnabledDeviceDiagnostics(self.device, self.logger)
        issues.extend(web_diagnostics.diagnose())

        return issues

    def ping_device(self, ip: str) -> bool:
        """
        Ping the device to check if it's reachable.
        """
        try:
            self.logger.debug(f"Pinging {ip} to check reachability.")
            # Use the ping command with 1 packet and a timeout of 2 seconds
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '2', ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                self.logger.debug(f"Ping to {ip} successful.")
                return True
            else:
                self.logger.debug(f"Ping to {ip} failed.")
                return False
        except Exception as e:
            self.logger.error(f"Error while pinging {ip}: {e}")
            return False

    def check_wps_status(self, ip: str, vendor_key: str) -> Optional[bool]:
        """
        Check if WPS is enabled on the router by accessing known endpoints based on vendor.
        This is a heuristic approach and may not work for all router brands.
        """
        try:
            self.logger.debug(f"Checking WPS status on router {ip} for vendor '{vendor_key}'.")

            # Define vendor-specific WPS endpoints
            wps_endpoints = {
                'fritz!box': ["/diag_wps.html"],
                'asus': ["/wps_status"],
                'netgear': ["/wps", "/status/wps"],
                'tp-link': ["/wireless/wps_status.asp"],
                'd-link': ["/wireless/wps_status.asp"],
                'linksys': ["/wireless/wps_status.asp"],
                'belkin': ["/wireless/wps_status.asp"],
                'synology': ["/wireless/wps_status.asp"],
                'ubiquiti': ["/wireless/wps_status.asp"],
                'mikrotik': ["/wireless/wps_status.asp"],
                'zyxel': ["/wireless/wps_status.asp"],
                'huawei': ["/wireless/wps_status.asp"],
                'apple': ["/wireless/wps_status.asp"],
            }

            endpoints = wps_endpoints.get(vendor_key, [
                "/wps",
                "/wps_status",
                "/wps.html",
            ])

            for endpoint in endpoints:
                url = f"http://{ip}{endpoint}"
                try:
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        # Heuristic: look for keywords in the response indicating WPS is enabled
                        if "enabled" in response.text.lower():
                            self.logger.debug(f"WPS is enabled on router {ip} via endpoint {endpoint}.")
                            return True
                        elif "disabled" in response.text.lower():
                            self.logger.debug(f"WPS is disabled on router {ip} via endpoint {endpoint}.")
                            return False
                except requests.RequestException:
                    continue  # Try the next endpoint

            self.logger.debug(f"Unable to determine WPS status on router {ip}.")
            return None  # Unable to determine
        except Exception as e:
            self.logger.error(f"Error while checking WPS status on router {ip}: {e}")
            return None

    def check_publicly_accessible_pages(self, ip: str, vendor_key: str) -> List[str]:
        """
        Check for the existence of publicly accessible diagnostic or backup pages
        that should typically be restricted based on vendor.
        """
        accessible_pages = []
        try:
            self.logger.debug(f"Checking for publicly accessible pages on router {ip} for vendor '{vendor_key}'.")

            # Define vendor-specific sensitive pages
            sensitive_endpoints = self.vendor_configs.get(vendor_key, {}).get('sensitive_endpoints', COMMON_SENSITIVE_ENDPOINTS)

            for endpoint in sensitive_endpoints:
                url = f"http://{ip}{endpoint}"
                try:
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        # Heuristic: presence of certain keywords indicating the page exists
                        keywords = ["login", "username", "password", "configuration", "diagnostic", "backup"]
                        if any(keyword in response.text.lower() for keyword in keywords):
                            accessible_pages.append(endpoint)
                            self.logger.debug(f"Accessible page detected: {endpoint} on router {ip}.")
                except requests.RequestException:
                    continue  # Page does not exist or is not accessible

        except Exception as e:
            self.logger.error(f"Error while checking publicly accessible pages on router {ip}: {e}")

        return accessible_pages

    def check_firmware_update(self, ip: str, vendor_key: str) -> bool:
        """
        Check if a firmware update is available for the router based on vendor-specific endpoints.
        This is a heuristic approach and may not work for all router brands.
        """
        try:
            self.logger.debug(f"Checking firmware update status for router {ip} for vendor '{vendor_key}'.")
            # Define vendor-specific firmware endpoints
            firmware_endpoints = self.vendor_configs.get(vendor_key, {}).get('firmware_endpoints', COMMON_SENSITIVE_FIRMWARE_ENDPOINTS)

            for endpoint in firmware_endpoints:
                url = f"http://{ip}{endpoint}"
                try:
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        # Heuristic: look for keywords indicating update availability
                        if "update available" in response.text.lower():
                            self.logger.debug(f"Firmware update available via endpoint {endpoint} on router {ip}.")
                            return True
                        elif "up to date" in response.text.lower():
                            self.logger.debug(f"Firmware is up to date via endpoint {endpoint} on router {ip}.")
                            return False
                except requests.RequestException:
                    continue  # Try the next endpoint

            self.logger.debug(f"Unable to determine firmware update status on router {ip}.")
            return False  # Assume no update available if status is unknown
        except Exception as e:
            self.logger.error(f"Error while checking firmware update for router {ip}: {e}")
            return False


# Printer Diagnostics
class PrinterDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics specific to printers.
    """

    def diagnose(self) -> Optional[List[Dict[str, str]]]:
        """
        Diagnose the printer for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug(f"Printer {hostname} IP is N/A. Skipping diagnostics.")
            return issues

        # Check if port 9100 is open (common for network printers)
        if not self.check_port(ip, 9100):
            issues.append({
                "DeviceType": "Printer",
                "Hostname": hostname,
                "IP": ip,
                "Description": "Port 9100 Unreachable"
            })

        return issues


# Web-Enabled Device Diagnostics Class
class WebEnabledDeviceDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics specific to web-enabled devices.
    """

    def diagnose(self) -> Optional[List[Dict[str, str]]]:
        """
        Diagnose the web-enabled device for common issues, including SSL certificate validity,
        security headers, HTTP response codes, and protocol usage.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug(f"Web-Enabled Device {hostname} IP is N/A. Skipping diagnostics.")
            return issues

        # Define common web ports to check
        web_ports = [80, 443]
        for port in web_ports:
            if not self.check_port(ip, port):
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"Port {port} Unreachable"
                })

        # Perform additional diagnostics only if port 80 or 443 is open
        if any(self.check_port(ip, port) for port in web_ports):
            # Check HTTP and HTTPS services
            service_issues = self.check_http_services(ip, hostname)
            issues.extend(service_issues)

        return issues

    def check_http_services(self, ip: str, hostname: str) -> List[Dict[str, str]]:
        """
        Perform comprehensive checks on HTTP and HTTPS services, including SSL certificate
        validity, security headers, HTTP response codes, and protocol usage.
        """
        issues = []
        services = []

        # Determine available services
        if self.check_port(ip, 80):
            services.append(('http', 80))
        if self.check_port(ip, 443):
            services.append(('https', 443))

        for protocol, port in services:
            url = f"http://{ip}:{port}" if protocol == 'http' else f"https://{ip}:{port}"
            self.logger.debug(f"Performing {protocol.upper()} diagnostics on {url}")

            try:
                if protocol == 'http':
                    response = requests.get(url, timeout=5)
                    self.logger.debug(f"Received HTTP response code {response.status_code} from {url}")
                    # Check HTTP response and security headers
                    response_issues = self.check_http_response(response, hostname, ip, protocol)
                    issues.extend(response_issues)
                else:
                    response = requests.get(url, timeout=5, verify=True)
                    self.logger.debug(f"Received HTTPS response code {response.status_code} from {url}")
                    # Check HTTPS response and security headers
                    response_issues = self.check_http_response(response, hostname, ip, protocol)
                    issues.extend(response_issues)
                    # Check SSL certificate
                    ssl_issues = self.check_ssl_certificate(ip, port, hostname)
                    issues.extend(ssl_issues)
            except requests.exceptions.SSLError as ssl_err:
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"SSL Error on {protocol.upper()} port {port} - {ssl_err}"
                })
            except requests.exceptions.ConnectionError as conn_err:
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"Connection Error on {protocol.upper()} port {port} - {conn_err}"
                })
            except requests.exceptions.Timeout:
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"Timeout while connecting to {protocol.upper()} port {port}"
                })
            except Exception as e:
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"Unexpected error on {protocol.upper()} port {port} - {e}"
                })

        return issues

    def check_http_response(self, response: requests.Response, hostname: str, ip: str, protocol: str) -> List[Dict[str, str]]:
        """
        Check HTTP response codes and security headers.
        """
        issues = []
        # Check for successful response
        if not response.ok:
            issues.append({
                "DeviceType": "Web-Enabled Device",
                "Hostname": hostname,
                "IP": ip,
                "Description": f"{protocol.upper()} response code {response.status_code}"
            })

        # Check for security headers using the unused method
        security_header_issues = self.check_security_headers(response, hostname, ip)
        issues.extend(security_header_issues)

        return issues

    def check_ssl_certificate(self, ip: str, port: int, hostname: str) -> List[Dict[str, str]]:
        """
        Check the SSL certificate validity, expiration, issuer, and whether it's self-signed.
        """
        issues = []
        context = ssl.create_default_context()
        try:
            with socket.create_connection((ip, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()

            # Check certificate expiration
            not_after = cert.get('notAfter')
            if not_after:
                expiration_date = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
                if expiration_date < datetime.utcnow():
                    issues.append({
                        "DeviceType": "Web-Enabled Device",
                        "Hostname": hostname,
                        "IP": ip,
                        "Description": f"SSL certificate expired on {expiration_date.strftime('%Y-%m-%d')}"
                    })
                else:
                    days_left = (expiration_date - datetime.utcnow()).days
                    if days_left < 30:
                        issues.append({
                            "DeviceType": "Web-Enabled Device",
                            "Hostname": hostname,
                            "IP": ip,
                            "Description": f"SSL certificate expires in {days_left} days on {expiration_date.strftime('%Y-%m-%d')}"
                        })

            # Check issuer
            issuer = dict(x[0] for x in cert.get('issuer', ()))
            issuer_common_name = issuer.get('commonName', '')
            trusted_issuers = self.get_trusted_issuers()
            if issuer_common_name not in trusted_issuers:
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": f"SSL certificate issued by untrusted issuer '{issuer_common_name}'"
                })

            # Check if certificate is self-signed
            subject = dict(x[0] for x in cert.get('subject', ()))
            subject_common_name = subject.get('commonName', '')
            issuer_common_name = issuer.get('commonName', '')
            if subject_common_name == issuer_common_name:
                issues.append({
                    "DeviceType": "Web-Enabled Device",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": "SSL certificate is self-signed"
                })

        except ssl.SSLError as e:
            issues.append({
                "DeviceType": "Web-Enabled Device",
                "Hostname": hostname,
                "IP": ip,
                "Description": f"SSL error - {e}"
            })
        except socket.timeout:
            issues.append({
                "DeviceType": "Web-Enabled Device",
                "Hostname": hostname,
                "IP": ip,
                "Description": "SSL connection timed out"
            })
        except Exception as e:
            issues.append({
                "DeviceType": "Web-Enabled Device",
                "Hostname": hostname,
                "IP": ip,
                "Description": f"Error retrieving SSL certificate - {e}"
            })

        return issues

    def get_trusted_issuers(self) -> set:
        """
        Retrieve a set of trusted certificate authorities.
        This list can be expanded as needed.
        """
        # For simplicity, define a static list of common trusted CAs
        return {
            'DigiCert Inc',
            'Let\'s Encrypt',
            'Comodo CA',
            'GlobalSign',
            'GoDaddy.com, Inc.',
            'Entrust, Inc.',
            'GeoTrust, Inc.',
            'Symantec Corporation',
            'Thawte',
            'Amazon',
            'VeriSign, Inc.'
        }

    def check_security_headers(self, response: requests.Response, hostname: str, ip: str) -> List[Dict[str, str]]:
        """
        Check for the presence of critical security headers in the HTTP response.
        """
        issues = []
        security_headers = {
            'Content-Security-Policy',
            'Strict-Transport-Security',
            'X-Frame-Options',
            'X-Content-Type-Options',
            'Referrer-Policy',
            'Feature-Policy',
            'Permissions-Policy'
        }
        missing_headers = [header for header in security_headers if header not in response.headers]
        if missing_headers:
            issues.append({
                "DeviceType": "Web-Enabled Device",
                "Hostname": hostname,
                "IP": ip,
                "Description": f"Missing security headers: {', '.join(missing_headers)}"
            })
        return issues


# Laptop, Desktop, and Phone Diagnostics
class GeneralClientDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics for laptops, desktops, and phones.
    """

    def diagnose(self) -> Optional[List[Dict[str, str]]]:
        """
        Diagnose the client device for common issues.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")
        os_info = self.device.get("OS", "Unknown").lower()

        if ip == "N/A":
            self.logger.debug(f"Device {hostname} IP is N/A. Skipping diagnostics.")
            return issues

        # Check SSH port 22 for Linux/Unix
        if any(os in os_info for os in ["linux", "unix"]):
            if not self.check_port(ip, 22):
                issues.append({
                    "DeviceType": "Desktop Computer",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": "SSH Port 22 Unreachable"
                })

        # Check RDP port 3389 for Windows
        if "windows" in os_info:
            if not self.check_port(ip, 3389):
                issues.append({
                    "DeviceType": "Desktop Computer",
                    "Hostname": hostname,
                    "IP": ip,
                    "Description": "RDP Port 3389 Unreachable"
                })

        # Additional checks based on device type
        device_type = self.device.get("DeviceType", "").lower()
        if "laptop" in device_type:
            # Placeholder for laptop-specific diagnostics
            pass
        elif "phone" in device_type:
            # Placeholder for phone-specific diagnostics
            # Example: Check ADB port for Android
            if "android" in os_info:
                if not self.check_port(ip, 5555):
                    issues.append({
                        "DeviceType": "Phone",
                        "Hostname": hostname,
                        "IP": ip,
                        "Description": "ADB Port 5555 Unreachable"
                    })
            # Add more phone-specific diagnostics as needed

        return issues


# Other Device Diagnostics
class OtherDeviceDiagnostics(GeneralDeviceDiagnostics):
    """
    Perform diagnostics for other types of devices.
    """

    def diagnose(self) -> Optional[List[Dict[str, str]]]:
        """
        Diagnose other devices for reachability.
        """
        issues = []
        ip = self.device.get("IP", "N/A")
        hostname = self.device.get("Hostname", "N/A")

        if ip == "N/A":
            self.logger.debug(f"Other Device {hostname} IP is N/A. Skipping diagnostics.")
            return issues

        # Generic diagnostic: Check if device is reachable
        status = self.ping_device(ip)
        if status != "Reachable":
            issues.append({
                "DeviceType": "Other",
                "Hostname": hostname,
                "IP": ip,
                "Description": status
            })

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

        # Save devices to a JSON file if output is requested
        if self.args.output_file:
            self.save_devices_to_file(classified_devices, self.args.output_file)

        return classified_devices

    def scan_network(self) -> List[Dict[str, str]]:
        """
        Scan the active subnets using nmap to discover devices.
        """
        try:
            # Dynamically determine the active subnets
            subnets = self.get_active_subnets()
            if not subnets:
                self.logger.error("No active subnets detected.")
                return []

            all_devices = []
            for subnet in subnets:
                self.logger.debug(f"Scanning subnet: {subnet}")
                # Use nmap for comprehensive scanning with OS detection and service enumeration
                # -O: Enable OS detection
                # -sV: Probe open ports to determine service/version info
                # -T4: Faster execution
                # -oX: Output in XML format to stdout
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
        Determine the active subnets based on the system's non-loopback IPv4 addresses.
        Excludes virtual interfaces by default unless self.args.include_virtual is True.
        """
        subnets = []
        try:
            result = subprocess.run(['ip', '-4', 'addr'], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            current_iface = None
            for line in lines:
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
                ports = []
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
                            ports.append(port_info)
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
        scanner = NetworkScannerCommand(self.args, self.logger)
        classified_devices = scanner.execute()

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
                            issues_found.append([
                                issue.get("DeviceType", "Unknown"),
                                issue.get("Hostname", "N/A"),
                                issue.get("IP", "N/A"),
                                issue.get("Description", "")
                            ])

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
            return RouterDiagnostics(device, self.logger)
        elif device_type == "Desktop Computer":
            return GeneralClientDiagnostics(device, self.logger)
        elif device_type == "Printer":
            return PrinterDiagnostics(device, self.logger)
        elif device_type == "Phone":
            return GeneralClientDiagnostics(device, self.logger)
        elif device_type == "Web-Enabled Device":
            return WebEnabledDeviceDiagnostics(device, self.logger)
        else:
            return OtherDeviceDiagnostics(device, self.logger)


# Traffic Monitor Command
class TrafficMonitorCommand(BaseCommand):
    """
    Monitor network traffic to detect anomalies.
    """
    def __init__(self, args, logger):
        super().__init__(args, logger)

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
        if console:
            console.print(Panel("[bold underline]IP Configuration[/bold underline]", style="cyan"))
            console.print(ip_info)
            console.print(Panel("[bold underline]Routing Table[/bold underline]", style="cyan"))
            console.print(routing_info)
            console.print(Panel("[bold underline]DNS Servers[/bold underline]", style="cyan"))
            for dns in dns_info:
                console.print(f"- {dns}")
        else:
            print("\n=== IP Configuration ===")
            print(ip_info)
            print("\n=== Routing Table ===")
            print(routing_info)
            print("\n=== DNS Servers ===")
            for dns in dns_info:
                print(f"- {dns}")


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

    def diagnose_wifi(self, networks: List[Dict[str, str]], target_networks: List[Dict[str, str]] = None) -> List[List[str]]:
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
            channel_issues = self.analyze_channel_interference(channel, networks)
            if channel_issues:
                issues.append(channel_issues)

        # Check for open (unsecured) networks
        open_networks = [net for net in target_networks if net['Security'].upper() in ['OPEN', '--']]
        for net in open_networks:
            issues.append([
                "WiFi Network",
                net['SSID'],
                "N/A",
                f"Open and unsecured network on channel {net['Channel']}."
            ])

        # Check for networks with weak signals
        weak_networks = [net for net in target_networks if self.safe_int(net['Signal']) < self.args.signal_threshold]
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
            return [
                "WiFi Channel",
                f"Channel {channel}",
                "N/A",
                f"High number of networks ({count}) on this channel causing interference."
            ]
        return []

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
        router_vendors = [
            'fritz!box', 'asus', 'netgear', 'tp-link', 'd-link', 'linksys',
            'belkin', 'synology', 'ubiquiti', 'mikrotik', 'zyxel'
        ]

        # Heuristic rules for classification
        if any("9100/tcp" in port for port in ports):
            return "Printer"
        elif vendor.lower() in router_vendors:
            return "Router"  # Additional vendor-based classification
        elif "router" in os_info:
            return "Router"
        elif any(port.startswith("80/tcp") or port.startswith("443/tcp") for port in ports):
            return "Web-Enabled Device"
        elif "phone" in os_info or "voip" in os_info:
            return "Phone"
        elif "desktop" in os_info or "workstation" in os_info:
            return "Desktop Computer"
        else:
            return "Unknown"


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

    command = command_class(args, logger)
    command.execute()

    logger.info("Network diagnostics completed successfully.")


if __name__ == "__main__":
    main()
