#!/usr/bin/env python3

# -------------------------------------------------------
# Script: detect_security.py
#
# Description:
# This script detects various security tools installed or enabled on the system.
# It covers antivirus software, firewalls, VPN clients, intrusion detection systems,
# honeypots, and other security-related tools. Each security tool provider has its own
# dedicated detection class, allowing for a modular and extensible design.
#
# Usage:
# ./detect_security.py [options]
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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# Base class for Security Tools
class SecSecurityToolBase(ABC):
    @abstractmethod
    def matches(self) -> bool:
        """Determine if this security tool is installed or active."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Gather security tool-specific information."""
        pass


# Windows Defender Detector
class SecWindowsDefender(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecWindowsDefender.matches() called.")
        try:
            result = subprocess.run(['sc', 'query', 'WinDefend'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "RUNNING" in result.stdout:
                logging.debug("Windows Defender service is running.")
                return True
            else:
                logging.debug("Windows Defender service is not running.")
                return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Windows Defender service.")
            return False
        except Exception as e:
            logging.error(f"Error checking Windows Defender: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Windows Defender information.")
        info = {"Name": "Windows Defender", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['powershell', '-Command', "Get-MpComputerStatus | Select-Object AntivirusSignatureVersion"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.strip().split('\n')[-1]
                info["Version"] = version_line
                logging.debug(f"Windows Defender version: {version_line}")
            else:
                logging.warning(f"Failed to retrieve Windows Defender version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Windows Defender version: {e}")
            info["Version"] = "Unknown"
        return info


# McAfee Antivirus Detector
class SecMcAfee(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecMcAfee.matches() called.")
        try:
            # Check if McAfee services are running
            services = ['McAfeeFramework', 'MFEvtMgr', 'McAfeeSetup', 'MFEvtMgr']
            for service in services:
                result = subprocess.run(['sc', 'query', service],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if "RUNNING" in result.stdout:
                    logging.debug(f"McAfee service '{service}' is running.")
                    return True
            logging.debug("No McAfee services are running.")
            return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking McAfee services.")
            return False
        except Exception as e:
            logging.error(f"Error checking McAfee: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering McAfee information.")
        info = {"Name": "McAfee Antivirus", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['wmic', 'product', 'where', 'name like "McAfee%%"', 'get', 'version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    version = lines[1].strip()
                    info["Version"] = version
                    logging.debug(f"McAfee version: {version}")
                else:
                    logging.debug("McAfee version information not found.")
                    info["Version"] = "Unknown"
            else:
                logging.warning(f"Failed to retrieve McAfee version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving McAfee version: {e}")
            info["Version"] = "Unknown"
        return info


# Avast Antivirus Detector
class SecAvast(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecAvast.matches() called.")
        try:
            # Check if Avast services are running
            services = ['AvastSvc', 'AvastUI']
            for service in services:
                result = subprocess.run(['sc', 'query', service],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if "RUNNING" in result.stdout:
                    logging.debug(f"Avast service '{service}' is running.")
                    return True
            logging.debug("No Avast services are running.")
            return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Avast services.")
            return False
        except Exception as e:
            logging.error(f"Error checking Avast: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Avast information.")
        info = {"Name": "Avast Antivirus", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['wmic', 'product', 'where', 'name like "Avast%%"', 'get', 'version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    version = lines[1].strip()
                    info["Version"] = version
                    logging.debug(f"Avast version: {version}")
                else:
                    logging.debug("Avast version information not found.")
                    info["Version"] = "Unknown"
            else:
                logging.warning(f"Failed to retrieve Avast version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Avast version: {e}")
            info["Version"] = "Unknown"
        return info


# Bitdefender Antivirus Detector
class SecBitdefender(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecBitdefender.matches() called.")
        try:
            result = subprocess.run(['sc', 'query', 'vsserv'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "RUNNING" in result.stdout:
                logging.debug("Bitdefender service is running.")
                return True
            else:
                logging.debug("Bitdefender service is not running.")
                return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Bitdefender service.")
            return False
        except Exception as e:
            logging.error(f"Error checking Bitdefender: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Bitdefender information.")
        info = {"Name": "Bitdefender Antivirus", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['wmic', 'product', 'where', 'name like "Bitdefender%%"', 'get', 'version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    version = lines[1].strip()
                    info["Version"] = version
                    logging.debug(f"Bitdefender version: {version}")
                else:
                    logging.debug("Bitdefender version information not found.")
                    info["Version"] = "Unknown"
            else:
                logging.warning(f"Failed to retrieve Bitdefender version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Bitdefender version: {e}")
            info["Version"] = "Unknown"
        return info


# Kaspersky Antivirus Detector
class SecKaspersky(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecKaspersky.matches() called.")
        try:
            result = subprocess.run(['sc', 'query', 'kavsvc'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "RUNNING" in result.stdout:
                logging.debug("Kaspersky service is running.")
                return True
            else:
                logging.debug("Kaspersky service is not running.")
                return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Kaspersky service.")
            return False
        except Exception as e:
            logging.error(f"Error checking Kaspersky: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Kaspersky information.")
        info = {"Name": "Kaspersky Antivirus", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['wmic', 'product', 'where', 'name like "Kaspersky%%"', 'get', 'version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    version = lines[1].strip()
                    info["Version"] = version
                    logging.debug(f"Kaspersky version: {version}")
                else:
                    logging.debug("Kaspersky version information not found.")
                    info["Version"] = "Unknown"
            else:
                logging.warning(f"Failed to retrieve Kaspersky version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Kaspersky version: {e}")
            info["Version"] = "Unknown"
        return info


# Sophos Antivirus Detector
class SecSophos(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecSophos.matches() called.")
        try:
            result = subprocess.run(['sc', 'query', 'SophosAntiVirus'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "RUNNING" in result.stdout:
                logging.debug("Sophos Antivirus service is running.")
                return True
            else:
                logging.debug("Sophos Antivirus service is not running.")
                return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Sophos Antivirus service.")
            return False
        except Exception as e:
            logging.error(f"Error checking Sophos Antivirus: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Sophos Antivirus information.")
        info = {"Name": "Sophos Antivirus", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['wmic', 'product', 'where', 'name like "Sophos%%"', 'get', 'version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    version = lines[1].strip()
                    info["Version"] = version
                    logging.debug(f"Sophos Antivirus version: {version}")
                else:
                    logging.debug("Sophos Antivirus version information not found.")
                    info["Version"] = "Unknown"
            else:
                logging.warning(f"Failed to retrieve Sophos Antivirus version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Sophos Antivirus version: {e}")
            info["Version"] = "Unknown"
        return info


# Symantec Endpoint Protection Detector
class SecSymantecEndpointProtection(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecSymantecEndpointProtection.matches() called.")
        try:
            result = subprocess.run(['sc', 'query', 'Symantec Endpoint Protection'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "RUNNING" in result.stdout:
                logging.debug("Symantec Endpoint Protection service is running.")
                return True
            else:
                logging.debug("Symantec Endpoint Protection service is not running.")
                return False
        except FileNotFoundError:
            logging.warning("sc command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Symantec Endpoint Protection service.")
            return False
        except Exception as e:
            logging.error(f"Error checking Symantec Endpoint Protection: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Symantec Endpoint Protection information.")
        info = {"Name": "Symantec Endpoint Protection", "Status": "Running"}
        # Attempt to get version information
        try:
            result = subprocess.run(['wmic', 'product', 'where', 'name like "Symantec%%"', 'get', 'version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    version = lines[1].strip()
                    info["Version"] = version
                    logging.debug(f"Symantec Endpoint Protection version: {version}")
                else:
                    logging.debug("Symantec Endpoint Protection version information not found.")
                    info["Version"] = "Unknown"
            else:
                logging.warning(f"Failed to retrieve Symantec Endpoint Protection version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Symantec Endpoint Protection version: {e}")
            info["Version"] = "Unknown"
        return info


# Cisco AnyConnect VPN Detector
class SecCiscoAnyConnect(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecCiscoAnyConnect.matches() called.")
        try:
            result = subprocess.run(['pgrep', '-f', 'vpnagentd'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("Cisco AnyConnect VPN agent process detected.")
                return True
            else:
                logging.debug("No Cisco AnyConnect VPN agent process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Cisco AnyConnect VPN agent.")
            return False
        except Exception as e:
            logging.error(f"Error checking Cisco AnyConnect VPN: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Cisco AnyConnect VPN information.")
        info = {"Name": "Cisco AnyConnect VPN"}
        try:
            result = subprocess.run(['vpn', '-version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Version"] = result.stdout.strip().split('\n')[0]
                logging.debug(f"Cisco AnyConnect VPN version: {info['Version']}")
            else:
                logging.warning(f"Failed to retrieve Cisco AnyConnect VPN version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except FileNotFoundError:
            logging.warning("Cisco AnyConnect VPN is not installed.")
            info["Version"] = "Not Installed"
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Cisco AnyConnect VPN version.")
            info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Cisco AnyConnect VPN version: {e}")
            info["Version"] = "Unknown"
        return info


# OpenVPN Detector
class SecOpenVPN(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecOpenVPN.matches() called.")
        try:
            result = subprocess.run(['pgrep', '-f', 'openvpn'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("OpenVPN process detected.")
                return True
            else:
                logging.debug("No OpenVPN process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking OpenVPN.")
            return False
        except Exception as e:
            logging.error(f"Error checking OpenVPN: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering OpenVPN information.")
        info = {"Name": "OpenVPN"}
        try:
            result = subprocess.run(['openvpn', '--version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Version"] = result.stdout.strip().split('\n')[0]
                logging.debug(f"OpenVPN version: {info['Version']}")
            else:
                logging.warning(f"Failed to retrieve OpenVPN version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except FileNotFoundError:
            logging.warning("OpenVPN is not installed.")
            info["Version"] = "Not Installed"
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving OpenVPN version.")
            info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving OpenVPN version: {e}")
            info["Version"] = "Unknown"
        return info


# NordVPN Detector
class SecNordVPN(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecNordVPN.matches() called.")
        try:
            result = subprocess.run(['nordvpn', 'status'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "Status: Connected" in result.stdout:
                logging.debug("NordVPN is connected.")
                return True
            else:
                logging.debug("NordVPN is not connected.")
                return False
        except FileNotFoundError:
            logging.debug("NordVPN is not installed.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking NordVPN status.")
            return False
        except Exception as e:
            logging.error(f"Error checking NordVPN: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering NordVPN information.")
        info = {"Name": "NordVPN"}
        try:
            result = subprocess.run(['nordvpn', 'status'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        info[key.strip()] = value.strip()
                logging.debug(f"NordVPN Status: {info}")
            else:
                logging.warning(f"Failed to retrieve NordVPN status: {result.stderr.strip()}")
                info["Status"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving NordVPN information: {e}")
            info["Status"] = "Unknown"
        return info


# Snort Detector
class SecSnort(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecSnort.matches() called.")
        try:
            # Check if Snort process is running
            result = subprocess.run(['pgrep', '-f', 'snort'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("Snort process detected.")
                return True
            else:
                logging.debug("No Snort process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Snort.")
            return False
        except Exception as e:
            logging.error(f"Error checking Snort: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Snort information.")
        info = {"Name": "Snort"}
        try:
            result = subprocess.run(['snort', '-V'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Version"] = result.stdout.strip().split('\n')[0]
                logging.debug(f"Snort version: {info['Version']}")
            else:
                logging.warning(f"Failed to retrieve Snort version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except FileNotFoundError:
            logging.warning("Snort is not installed.")
            info["Version"] = "Not Installed"
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Snort version.")
            info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Snort version: {e}")
            info["Version"] = "Unknown"
        return info


# Suricata Detector
class SecSuricata(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecSuricata.matches() called.")
        try:
            # Check if Suricata process is running
            result = subprocess.run(['pgrep', '-f', 'suricata'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("Suricata process detected.")
                return True
            else:
                logging.debug("No Suricata process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Suricata.")
            return False
        except Exception as e:
            logging.error(f"Error checking Suricata: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Suricata information.")
        info = {"Name": "Suricata"}
        try:
            result = subprocess.run(['suricata', '--build-info'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["BuildInfo"] = result.stdout.strip()
                logging.debug(f"Suricata Build Info: {info['BuildInfo']}")
            else:
                logging.warning(f"Failed to retrieve Suricata build info: {result.stderr.strip()}")
                info["BuildInfo"] = "Unknown"
        except FileNotFoundError:
            logging.warning("Suricata is not installed.")
            info["BuildInfo"] = "Not Installed"
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Suricata build info.")
            info["BuildInfo"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Suricata build info: {e}")
            info["BuildInfo"] = "Unknown"
        return info


# Honeyd Detector
class SecHoneyd(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecHoneyd.matches() called.")
        try:
            # Check if Honeyd process is running
            result = subprocess.run(['pgrep', '-f', 'honeyd'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("Honeyd process detected.")
                return True
            else:
                logging.debug("No Honeyd process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Honeyd.")
            return False
        except Exception as e:
            logging.error(f"Error checking Honeyd: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Honeyd information.")
        info = {"Name": "Honeyd"}
        try:
            result = subprocess.run(['honeyd', '-v'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Version"] = result.stdout.strip().split('\n')[0]
                logging.debug(f"Honeyd version: {info['Version']}")
            else:
                logging.warning(f"Failed to retrieve Honeyd version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except FileNotFoundError:
            logging.warning("Honeyd is not installed.")
            info["Version"] = "Not Installed"
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Honeyd version.")
            info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Honeyd version: {e}")
            info["Version"] = "Unknown"
        return info


# Kippo Honeypot Detector
class SecKippo(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecKippo.matches() called.")
        try:
            # Check if Kippo process is running
            result = subprocess.run(['pgrep', '-f', 'kippo'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("Kippo process detected.")
                return True
            else:
                logging.debug("No Kippo process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Kippo.")
            return False
        except Exception as e:
            logging.error(f"Error checking Kippo: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Kippo information.")
        info = {"Name": "Kippo Honeypot"}
        try:
            # Kippo may not have a version flag; retrieve process details instead
            result = subprocess.run(['ps', '-C', 'kippo', '-o', 'pid=,cmd='],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                info["Process"] = result.stdout.strip()
                logging.debug(f"Kippo Process Info: {result.stdout.strip()}")
            else:
                logging.warning("Kippo process details not found.")
                info["Process"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Kippo information: {e}")
            info["Process"] = "Unknown"
        return info


# Cowrie Honeypot Detector
class SecCowrie(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecCowrie.matches() called.")
        try:
            # Check if Cowrie process is running
            result = subprocess.run(['pgrep', '-f', 'cowrie'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.stdout.strip():
                logging.debug("Cowrie process detected.")
                return True
            else:
                logging.debug("No Cowrie process detected.")
                return False
        except FileNotFoundError:
            logging.warning("pgrep command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Cowrie.")
            return False
        except Exception as e:
            logging.error(f"Error checking Cowrie: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Cowrie information.")
        info = {"Name": "Cowrie Honeypot"}
        try:
            result = subprocess.run(['cowrie', '--version'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Version"] = result.stdout.strip().split('\n')[0]
                logging.debug(f"Cowrie version: {info['Version']}")
            else:
                logging.warning(f"Failed to retrieve Cowrie version: {result.stderr.strip()}")
                info["Version"] = "Unknown"
        except FileNotFoundError:
            logging.warning("Cowrie is not installed.")
            info["Version"] = "Not Installed"
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Cowrie version.")
            info["Version"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Cowrie version: {e}")
            info["Version"] = "Unknown"
        return info


class SecWindowsFirewall(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecWindowsFirewall.matches() called.")
        try:
            result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "State ON" in result.stdout:
                logging.debug("Windows Firewall is enabled.")
                return True
            else:
                logging.debug("Windows Firewall is disabled.")
                return False
        except FileNotFoundError:
            logging.warning("netsh command not found.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Windows Firewall.")
            return False
        except Exception as e:
            logging.error(f"Error checking Windows Firewall: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Windows Firewall information.")
        info = {"Name": "Windows Firewall"}
        try:
            result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                profiles = {}
                current_profile = None
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "Profile" in line:
                        current_profile = line.split(" : ")[0]
                        profiles[current_profile] = {}
                    elif ":" in line and current_profile:
                        key, value = line.split(" : ", 1)
                        profiles[current_profile][key.strip()] = value.strip()
                info["Profiles"] = profiles
                logging.debug(f"Windows Firewall profiles: {profiles}")
            else:
                logging.warning(f"Failed to retrieve Windows Firewall profiles: {result.stderr.strip()}")
                info["Status"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving Windows Firewall information: {e}")
            info["Status"] = "Unknown"
        return info


class SecUFW(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecUFW.matches() called.")
        if platform.system() != "Linux":
            logging.debug("UFW detection skipped: Not a Linux system.")
            return False
        try:
            result = subprocess.run(['ufw', 'status'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "Status: active" in result.stdout:
                logging.debug("UFW is active.")
                return True
            else:
                logging.debug("UFW is inactive.")
                return False
        except FileNotFoundError:
            logging.debug("UFW is not installed.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking UFW.")
            return False
        except Exception as e:
            logging.error(f"Error checking UFW: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering UFW information.")
        info = {"Name": "UFW (Uncomplicated Firewall)"}
        try:
            result = subprocess.run(['ufw', 'status', 'verbose'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Status"] = "Active"
                info["Rules"] = result.stdout.strip()
                logging.debug(f"UFW Rules: {result.stdout.strip()}")
            else:
                logging.warning(f"Failed to retrieve UFW status: {result.stderr.strip()}")
                info["Status"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving UFW information: {e}")
            info["Status"] = "Unknown"
        return info


class SecIPTables(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecIPTables.matches() called.")
        if platform.system() != "Linux":
            logging.debug("IPTables detection skipped: Not a Linux system.")
            return False
        try:
            result = subprocess.run(['iptables', '-L'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if "Chain" in result.stdout:
                logging.debug("iptables rules detected.")
                return True
            else:
                logging.debug("No iptables rules detected.")
                return False
        except FileNotFoundError:
            logging.debug("iptables is not installed.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking iptables.")
            return False
        except Exception as e:
            logging.error(f"Error checking iptables: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering iptables information.")
        info = {"Name": "iptables"}
        try:
            result = subprocess.run(['iptables', '-L', '-n'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                info["Rules"] = result.stdout.strip()
                logging.debug(f"iptables Rules: {result.stdout.strip()}")
            else:
                logging.warning(f"Failed to retrieve iptables rules: {result.stderr.strip()}")
                info["Rules"] = "Unknown"
        except Exception as e:
            logging.error(f"Error retrieving iptables information: {e}")
            info["Rules"] = "Unknown"
        return info


# Zscaler Detector
class SecZscaler(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecZscaler.matches() called.")
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['Zscaler Service', 'Zscaler Tunnel']
                for service in services:
                    result = subprocess.run(['sc', 'query', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if "RUNNING" in result.stdout:
                        logging.debug(f"Zscaler service '{service}' is running.")
                        return True
                logging.debug("No Zscaler services are running on Windows.")
            elif os_type in ["Linux", "Darwin"]:
                processes = ['zscaler', 'zscaler-cloud']
                for proc in processes:
                    result = subprocess.run(['pgrep', '-f', proc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.stdout.strip():
                        logging.debug(f"Zscaler process '{proc}' detected.")
                        return True
                logging.debug("No Zscaler processes detected on Linux/Darwin.")
            else:
                logging.debug("Unsupported OS for Zscaler detection.")
                return False
            return False
        except FileNotFoundError:
            logging.warning("Required command not found for Zscaler detection.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Zscaler.")
            return False
        except Exception as e:
            logging.error(f"Error checking Zscaler: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Zscaler information.")
        info = {"Name": "Zscaler"}
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['Zscaler Service', 'Zscaler Tunnel']
                versions = []
                for service in services:
                    result = subprocess.run(['sc', 'qc', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "BINARY_PATH_NAME" in line:
                                versions.append(line.split(":")[1].strip())
                info["Services"] = services
                info["BinaryPaths"] = versions if versions else "Unknown"
                logging.debug(f"Zscaler Services: {services}, Binary Paths: {versions}")
            elif os_type in ["Linux", "Darwin"]:
                result = subprocess.run(['zscaler', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if result.returncode == 0:
                    info["Version"] = result.stdout.strip().split('\n')[0]
                    logging.debug(f"Zscaler version: {info['Version']}")
                else:
                    logging.warning(f"Failed to retrieve Zscaler version: {result.stderr.strip()}")
                    info["Version"] = "Unknown"
            else:
                info["Details"] = "Unsupported OS for detailed Zscaler information."
            return info
        except FileNotFoundError:
            logging.warning("Zscaler is not installed.")
            info["Status"] = "Not Installed"
            return info
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Zscaler information.")
            info["Status"] = "Unknown"
            return info
        except Exception as e:
            logging.error(f"Error retrieving Zscaler information: {e}")
            info["Status"] = "Unknown"
            return info


# FortiClient Detector
class SecFortiClient(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecFortiClient.matches() called.")
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['FortiClient', 'FortiClient Service']
                for service in services:
                    result = subprocess.run(['sc', 'query', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if "RUNNING" in result.stdout:
                        logging.debug(f"FortiClient service '{service}' is running.")
                        return True
                logging.debug("No FortiClient services are running on Windows.")
            elif os_type in ["Linux", "Darwin"]:
                processes = ['forticlient', 'FortiClient']
                for proc in processes:
                    result = subprocess.run(['pgrep', '-f', proc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.stdout.strip():
                        logging.debug(f"FortiClient process '{proc}' detected.")
                        return True
                logging.debug("No FortiClient processes detected on Linux/Darwin.")
            else:
                logging.debug("Unsupported OS for FortiClient detection.")
                return False
            return False
        except FileNotFoundError:
            logging.warning("Required command not found for FortiClient detection.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking FortiClient.")
            return False
        except Exception as e:
            logging.error(f"Error checking FortiClient: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering FortiClient information.")
        info = {"Name": "FortiClient"}
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['FortiClient', 'FortiClient Service']
                versions = []
                for service in services:
                    result = subprocess.run(['sc', 'qc', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "BINARY_PATH_NAME" in line:
                                versions.append(line.split(":")[1].strip())
                info["Services"] = services
                info["BinaryPaths"] = versions if versions else "Unknown"
                logging.debug(f"FortiClient Services: {services}, Binary Paths: {versions}")
            elif os_type in ["Linux", "Darwin"]:
                result = subprocess.run(['forticlient', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if result.returncode == 0:
                    info["Version"] = result.stdout.strip().split('\n')[0]
                    logging.debug(f"FortiClient version: {info['Version']}")
                else:
                    logging.warning(f"Failed to retrieve FortiClient version: {result.stderr.strip()}")
                    info["Version"] = "Unknown"
            else:
                info["Details"] = "Unsupported OS for detailed FortiClient information."
            return info
        except FileNotFoundError:
            logging.warning("FortiClient is not installed.")
            info["Status"] = "Not Installed"
            return info
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving FortiClient information.")
            info["Status"] = "Unknown"
            return info
        except Exception as e:
            logging.error(f"Error retrieving FortiClient information: {e}")
            info["Status"] = "Unknown"
            return info


# CrowdStrike Detector
class SecCrowdStrike(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecCrowdStrike.matches() called.")
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['CrowdStrike Falcon Sensor']
                for service in services:
                    result = subprocess.run(['sc', 'query', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if "RUNNING" in result.stdout:
                        logging.debug(f"CrowdStrike service '{service}' is running.")
                        return True
                logging.debug("No CrowdStrike services are running on Windows.")
            elif os_type in ["Linux", "Darwin"]:
                processes = ['falcon-sensor']
                for proc in processes:
                    result = subprocess.run(['pgrep', '-f', proc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.stdout.strip():
                        logging.debug(f"CrowdStrike process '{proc}' detected.")
                        return True
                logging.debug("No CrowdStrike processes detected on Linux/Darwin.")
            else:
                logging.debug("Unsupported OS for CrowdStrike detection.")
                return False
            return False
        except FileNotFoundError:
            logging.warning("Required command not found for CrowdStrike detection.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking CrowdStrike.")
            return False
        except Exception as e:
            logging.error(f"Error checking CrowdStrike: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering CrowdStrike information.")
        info = {"Name": "CrowdStrike Falcon"}
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['CrowdStrike Falcon Sensor']
                versions = []
                for service in services:
                    result = subprocess.run(['sc', 'qc', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "BINARY_PATH_NAME" in line:
                                versions.append(line.split(":")[1].strip())
                info["Services"] = services
                info["BinaryPaths"] = versions if versions else "Unknown"
                logging.debug(f"CrowdStrike Services: {services}, Binary Paths: {versions}")
            elif os_type in ["Linux", "Darwin"]:
                result = subprocess.run(['falcon-sensor', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if result.returncode == 0:
                    info["Version"] = result.stdout.strip().split('\n')[0]
                    logging.debug(f"CrowdStrike version: {info['Version']}")
                else:
                    logging.warning(f"Failed to retrieve CrowdStrike version: {result.stderr.strip()}")
                    info["Version"] = "Unknown"
            else:
                info["Details"] = "Unsupported OS for detailed CrowdStrike information."
            return info
        except FileNotFoundError:
            logging.warning("CrowdStrike Falcon is not installed.")
            info["Status"] = "Not Installed"
            return info
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving CrowdStrike information.")
            info["Status"] = "Unknown"
            return info
        except Exception as e:
            logging.error(f"Error retrieving CrowdStrike information: {e}")
            info["Status"] = "Unknown"
            return info


# CyberHeaven Detector
class SecCyberHeaven(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecCyberHeaven.matches() called.")
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['CyberHeavenService']
                for service in services:
                    result = subprocess.run(['sc', 'query', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if "RUNNING" in result.stdout:
                        logging.debug(f"CyberHeaven service '{service}' is running.")
                        return True
                logging.debug("No CyberHeaven services are running on Windows.")
            elif os_type in ["Linux", "Darwin"]:
                processes = ['cyberheaven']
                for proc in processes:
                    result = subprocess.run(['pgrep', '-f', proc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.stdout.strip():
                        logging.debug(f"CyberHeaven process '{proc}' detected.")
                        return True
                logging.debug("No CyberHeaven processes detected on Linux/Darwin.")
            else:
                logging.debug("Unsupported OS for CyberHeaven detection.")
                return False
            return False
        except FileNotFoundError:
            logging.warning("Required command not found for CyberHeaven detection.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking CyberHeaven.")
            return False
        except Exception as e:
            logging.error(f"Error checking CyberHeaven: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering CyberHeaven information.")
        info = {"Name": "CyberHeaven"}
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['CyberHeavenService']
                versions = []
                for service in services:
                    result = subprocess.run(['sc', 'qc', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "BINARY_PATH_NAME" in line:
                                versions.append(line.split(":")[1].strip())
                info["Services"] = services
                info["BinaryPaths"] = versions if versions else "Unknown"
                logging.debug(f"CyberHeaven Services: {services}, Binary Paths: {versions}")
            elif os_type in ["Linux", "Darwin"]:
                result = subprocess.run(['cyberheaven', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if result.returncode == 0:
                    info["Version"] = result.stdout.strip().split('\n')[0]
                    logging.debug(f"CyberHeaven version: {info['Version']}")
                else:
                    logging.warning(f"Failed to retrieve CyberHeaven version: {result.stderr.strip()}")
                    info["Version"] = "Unknown"
            else:
                info["Details"] = "Unsupported OS for detailed CyberHeaven information."
            return info
        except FileNotFoundError:
            logging.warning("CyberHeaven is not installed.")
            info["Status"] = "Not Installed"
            return info
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving CyberHeaven information.")
            info["Status"] = "Unknown"
            return info
        except Exception as e:
            logging.error(f"Error retrieving CyberHeaven information: {e}")
            info["Status"] = "Unknown"
            return info


# SentinelOne Detector
class SecSentinelOne(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecSentinelOne.matches() called.")
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['SentinelAgent']
                for service in services:
                    result = subprocess.run(['sc', 'query', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if "RUNNING" in result.stdout:
                        logging.debug(f"SentinelOne service '{service}' is running.")
                        return True
                logging.debug("No SentinelOne services are running on Windows.")
            elif os_type in ["Linux", "Darwin"]:
                processes = ['sentinel-agent']
                for proc in processes:
                    result = subprocess.run(['pgrep', '-f', proc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.stdout.strip():
                        logging.debug(f"SentinelOne process '{proc}' detected.")
                        return True
                logging.debug("No SentinelOne processes detected on Linux/Darwin.")
            else:
                logging.debug("Unsupported OS for SentinelOne detection.")
                return False
            return False
        except FileNotFoundError:
            logging.warning("Required command not found for SentinelOne detection.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking SentinelOne.")
            return False
        except Exception as e:
            logging.error(f"Error checking SentinelOne: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering SentinelOne information.")
        info = {"Name": "SentinelOne"}
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['SentinelAgent']
                versions = []
                for service in services:
                    result = subprocess.run(['sc', 'qc', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "BINARY_PATH_NAME" in line:
                                versions.append(line.split(":")[1].strip())
                info["Services"] = services
                info["BinaryPaths"] = versions if versions else "Unknown"
                logging.debug(f"SentinelOne Services: {services}, Binary Paths: {versions}")
            elif os_type in ["Linux", "Darwin"]:
                result = subprocess.run(['sentinel-agent', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if result.returncode == 0:
                    info["Version"] = result.stdout.strip().split('\n')[0]
                    logging.debug(f"SentinelOne version: {info['Version']}")
                else:
                    logging.warning(f"Failed to retrieve SentinelOne version: {result.stderr.strip()}")
                    info["Version"] = "Unknown"
            else:
                info["Details"] = "Unsupported OS for detailed SentinelOne information."
            return info
        except FileNotFoundError:
            logging.warning("SentinelOne is not installed.")
            info["Status"] = "Not Installed"
            return info
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving SentinelOne information.")
            info["Status"] = "Unknown"
            return info
        except Exception as e:
            logging.error(f"Error retrieving SentinelOne information: {e}")
            info["Status"] = "Unknown"
            return info


# Palo Alto Networks Cortex XDR Detector
class SecCortexXDR(SecSecurityToolBase):
    def matches(self) -> bool:
        logging.debug("SecCortexXDR.matches() called.")
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['CortexService']
                for service in services:
                    result = subprocess.run(['sc', 'query', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if "RUNNING" in result.stdout:
                        logging.debug(f"Cortex XDR service '{service}' is running.")
                        return True
                logging.debug("No Cortex XDR services are running on Windows.")
            elif os_type in ["Linux", "Darwin"]:
                processes = ['cortex-xdr']
                for proc in processes:
                    result = subprocess.run(['pgrep', '-f', proc],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.stdout.strip():
                        logging.debug(f"Cortex XDR process '{proc}' detected.")
                        return True
                logging.debug("No Cortex XDR processes detected on Linux/Darwin.")
            else:
                logging.debug("Unsupported OS for Cortex XDR detection.")
                return False
            return False
        except FileNotFoundError:
            logging.warning("Required command not found for Cortex XDR detection.")
            return False
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while checking Cortex XDR.")
            return False
        except Exception as e:
            logging.error(f"Error checking Cortex XDR: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        logging.debug("Gathering Cortex XDR information.")
        info = {"Name": "Palo Alto Networks Cortex XDR"}
        os_type = platform.system()
        try:
            if os_type == "Windows":
                services = ['CortexService']
                versions = []
                for service in services:
                    result = subprocess.run(['sc', 'qc', service],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            timeout=5)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "BINARY_PATH_NAME" in line:
                                versions.append(line.split(":")[1].strip())
                info["Services"] = services
                info["BinaryPaths"] = versions if versions else "Unknown"
                logging.debug(f"Cortex XDR Services: {services}, Binary Paths: {versions}")
            elif os_type in ["Linux", "Darwin"]:
                result = subprocess.run(['cortex-xdr', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        timeout=5)
                if result.returncode == 0:
                    info["Version"] = result.stdout.strip().split('\n')[0]
                    logging.debug(f"Cortex XDR version: {info['Version']}")
                else:
                    logging.warning(f"Failed to retrieve Cortex XDR version: {result.stderr.strip()}")
                    info["Version"] = "Unknown"
            else:
                info["Details"] = "Unsupported OS for detailed Cortex XDR information."
            return info
        except FileNotFoundError:
            logging.warning("Cortex XDR is not installed.")
            info["Status"] = "Not Installed"
            return info
        except subprocess.TimeoutExpired:
            logging.warning("Timeout expired while retrieving Cortex XDR information.")
            info["Status"] = "Unknown"
            return info
        except Exception as e:
            logging.error(f"Error retrieving Cortex XDR information: {e}")
            info["Status"] = "Unknown"
            return info


# System Detector Class
class SecurityDetector:
    def __init__(self, perform_all_checks: bool = False) -> None:
        self.perform_all_checks = perform_all_checks
        self.security_detectors = self._initialize_security_detectors()

    def _initialize_security_detectors(self) -> List[SecSecurityToolBase]:
        logging.debug("Initializing security detectors.")
        detectors = [
            # Antivirus Detectors
            SecWindowsDefender(),
            SecMcAfee(),
            SecAvast(),
            SecBitdefender(),
            SecKaspersky(),
            SecSophos(),
            SecSymantecEndpointProtection(),
            # Firewall Detectors
            SecWindowsFirewall(),
            SecUFW(),
            SecIPTables(),
            # VPN Detectors
            SecOpenVPN(),
            SecNordVPN(),
            SecCiscoAnyConnect(),
            # Intrusion Detection Systems Detectors
            SecSnort(),
            SecSuricata(),
            # Honeypot Detectors
            SecHoneyd(),
            SecKippo(),
            SecCowrie(),
            # Enterprise Security Software Detectors
            SecZscaler(),
            SecFortiClient(),
            SecCrowdStrike(),
            SecCyberHeaven(),
            SecSentinelOne(),
            SecCortexXDR(),
            # Add more detectors here as needed
        ]
        return detectors

    def detect_security_tools(self) -> List[Dict[str, Any]]:
        logging.debug("Starting security tools detection.")
        detected_tools = []
        for detector in self.security_detectors:
            if self.perform_all_checks or self._is_relevant(detector):
                if detector.matches():
                    info = detector.get_info()
                    detected_tools.append(info)
                    logging.debug(f"Detected security tool: {info}")
        logging.debug("Security tools detection completed.")
        return detected_tools

    def _is_relevant(self, detector: SecSecurityToolBase) -> bool:
        """
        Determine if the detector is relevant based on the current OS.
        """
        os_type = platform.system()
        # Mapping of detector classes to supported operating systems
        supported_os_map = {
            # Antivirus Detectors
            SecWindowsDefender: ["Windows"],
            SecMcAfee: ["Windows"],
            SecAvast: ["Windows"],
            SecBitdefender: ["Windows"],
            SecKaspersky: ["Windows"],
            SecSophos: ["Windows"],
            SecSymantecEndpointProtection: ["Windows"],
            # Firewall Detectors
            SecWindowsFirewall: ["Windows"],
            SecUFW: ["Linux"],
            SecIPTables: ["Linux"],
            # VPN Detectors
            SecOpenVPN: ["Windows", "Linux", "Darwin"],
            SecNordVPN: ["Windows", "Linux", "Darwin"],
            SecCiscoAnyConnect: ["Windows", "Linux", "Darwin"],
            # Intrusion Detection Systems Detectors
            SecSnort: ["Linux"],
            SecSuricata: ["Linux"],
            # Honeypot Detectors
            SecHoneyd: ["Windows", "Linux", "Darwin"],
            SecKippo: ["Windows", "Linux", "Darwin"],
            SecCowrie: ["Windows", "Linux", "Darwin"],
            # Enterprise Security Software Detectors
            SecZscaler: ["Windows", "Linux", "Darwin"],
            SecFortiClient: ["Windows", "Linux", "Darwin"],
            SecCrowdStrike: ["Windows", "Linux", "Darwin"],
            SecCyberHeaven: ["Windows", "Linux", "Darwin"],
            SecSentinelOne: ["Windows", "Linux", "Darwin"],
            SecCortexXDR: ["Windows", "Linux", "Darwin"],
            # Add more detectors here as needed
        }

        detector_class = type(detector)
        supported_os = supported_os_map.get(detector_class, [])

        if self.perform_all_checks:
            return True

        return os_type in supported_os


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect installed or active security tools on the system.",
        formatter_class=argparse.RawTextHelpFormatter
    )

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


def save_output(data: List[Dict[str, Any]], filepath: str) -> bool:
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


def display_results(data: List[Dict[str, Any]]) -> None:
    """
    Displays the detection results in a formatted manner.
    """
    if not data:
        print("No security tools detected.")
        return

    for tool in data:
        print(f"Name: {tool.get('Name', 'Unknown')}")
        for key, value in tool.items():
            if key != "Name":
                print(f"  {key}: {value}")
        print("-" * 40)


def main() -> None:
    """
    Main function to orchestrate the security tools detection.
    """
    args = parse_arguments()
    setup_logging(
        verbose=args.verbose,
        debug=args.debug
    )

    detector = SecurityDetector(perform_all_checks=args.all)
    security_tools = detector.detect_security_tools()

    display_results(security_tools)

    if args.output:
        if not save_output(security_tools, args.output):
            logging.error("Failed to save detection results.")
            sys.exit(1)

    logging.info("Security tools detection completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
