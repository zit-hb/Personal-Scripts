#!/usr/bin/env python3

# -------------------------------------------------------
# Script: tcp_honeypot.py
#
# Description:
# A robust honeypot script that listens on one or more TCP ports.
# Any incoming connection is immediately closed, and a detailed
# notification (via one or more configured notifiers) is sent.
#
# Usage:
#   ./tcp_honeypot.py [options]
#
# Options:
#   -p, --port PORT               TCP port or port range to bind.
#                                 Can be specified multiple times (default: 8080).
#   -r, --reverse-dns             Attempt reverse DNS lookup (passive).
#   -e, --email EMAIL             Enable email notification to this recipient.
#   -H, --smtp-server SERVER      SMTP server hostname or IP (default: localhost).
#   -O, --smtp-port PORT          SMTP server port (default: 25).
#   -u, --smtp-user USERNAME      SMTP username (if auth is required).
#   -a, --smtp-password PASSWORD  SMTP password (if auth is required).
#   -f, --smtp-sender ADDRESS     Sender email address (default: hendrik@example.com).
#   -t, --smtp-tls                Use STARTTLS after connecting.
#   -x, --smtp-ssl                Use SSL/TLS for the entire SMTP connection.
#   -b, --webhook URL             Enable webhook notification to the given URL.
#   -K, --webhook-key KEY         Optional API key to include if webhook is used.
#   -o, --stdout                  Print a notification to stdout for each connection.
#   -L, --flood-limit INT         Number of notifications per IP within the flood
#                                 interval. 0 = unlimited (default: 10).
#   -I, --flood-interval INT      Flood protection interval in seconds (default: 300).
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#   -h, --help                    Show this help message and exit.
#
# Template: ubuntu24.04
#
# Requirements:
#   - requests (install via: pip install requests==2.32.3)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

try:
    import smtplib
    from email.mime.text import MIMEText
except ImportError:
    smtplib = None


@dataclass
class ConnectionInfo:
    """Data class holding information about each connection."""

    timestamp: datetime
    ip: str
    input_port: int
    output_port: int
    reverse_dns: Optional[str] = None


class Notifier(ABC):
    """Abstract base class for any notification method."""

    @abstractmethod
    def notify(self, info: ConnectionInfo) -> None:
        """Send a notification about the given connection info."""
        pass


class StdoutNotifier(Notifier):
    """Notifier that prints the connection info to stdout."""

    def notify(self, info: ConnectionInfo) -> None:
        print(
            f"[{info.timestamp}] Connection from {info.ip}:{info.output_port} "
            f"to {info.input_port} (rDNS: {info.reverse_dns or 'N/A'})"
        )


class EmailNotifier(Notifier):
    """Notifier to send email notifications (with optional auth, TLS/SSL)."""

    def __init__(
        self,
        recipient_email: str,
        smtp_server: str,
        smtp_port: int,
        smtp_user: Optional[str],
        smtp_password: Optional[str],
        smtp_sender: str,
        use_tls: bool,
        use_ssl: bool,
    ) -> None:
        """
        :param recipient_email: Email address to send alerts to.
        :param smtp_server: SMTP server hostname or IP.
        :param smtp_port: SMTP server port.
        :param smtp_user: SMTP username (if authentication is required).
        :param smtp_password: SMTP password (if authentication is required).
        :param smtp_sender: 'From' email address.
        :param use_tls: Whether to enable STARTTLS after connecting.
        :param use_ssl: Whether to use SSL/TLS from the start.
        """
        self.recipient_email = recipient_email
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.smtp_sender = smtp_sender
        self.use_tls = use_tls
        self.use_ssl = use_ssl

    def notify(self, info: ConnectionInfo) -> None:
        """Send an email containing the connection info."""
        if not smtplib:
            logging.error("smtplib is not available. Cannot send email.")
            return

        message_body = (
            f"Honeypot Alert!\n\n"
            f"Timestamp: {info.timestamp}\n"
            f"Client IP: {info.ip}\n"
            f"Client Port: {info.output_port}\n"
            f"Server Port: {info.input_port}\n"
            f"Reverse DNS: {info.reverse_dns or 'N/A'}\n"
        )

        msg = MIMEText(message_body)
        msg["Subject"] = "Honeypot Connection Alert"
        msg["From"] = self.smtp_sender
        msg["To"] = self.recipient_email

        try:
            if self.use_ssl:
                smtp_class = smtplib.SMTP_SSL
            else:
                smtp_class = smtplib.SMTP

            with smtp_class(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.ehlo_or_helo_if_needed()
                if not self.use_ssl and self.use_tls:
                    server.starttls()
                    server.ehlo_or_helo_if_needed()

                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)

                server.send_message(msg)
            logging.info(f"Email sent to {self.recipient_email}")

        except Exception as e:
            logging.error(f"Error sending email to {self.recipient_email}: {e}")


class WebhookNotifier(Notifier):
    """Notifier to send notifications via a webhook URL."""

    def __init__(self, webhook_url: str, webhook_key: Optional[str] = None) -> None:
        """
        :param webhook_url: The URL to post notifications to.
        :param webhook_key: Optional key to include in the X-API-Key header.
        """
        self.webhook_url = webhook_url
        self.webhook_key = webhook_key

    def notify(self, info: ConnectionInfo) -> None:
        """Send an HTTP POST request with the connection info as JSON."""
        if not requests:
            logging.error("requests module is not available. Cannot send webhook.")
            return

        payload = {
            "timestamp": str(info.timestamp),
            "protocol": "TCP",
            "ip": info.ip,
            "input_port": info.input_port,
            "output_port": info.output_port,
            "reverse_dns": info.reverse_dns or "N/A",
        }

        headers = {}
        if self.webhook_key:
            headers["X-API-Key"] = self.webhook_key

        try:
            response = requests.post(
                self.webhook_url, json=payload, headers=headers, timeout=5
            )
            if response.ok:
                logging.info(f"Webhook sent to {self.webhook_url}")
            else:
                logging.warning(
                    f"Webhook to {self.webhook_url} returned status code {response.status_code}"
                )
        except Exception as e:
            logging.error(f"Error sending webhook to {self.webhook_url}: {e}")


class Honeypot:
    """
    A honeypot that listens on specified TCP ports. Whenever a client connects,
    the connection is immediately closed, and a ConnectionInfo object is
    passed to all notifiers. Flood protection ensures that only a certain number
    of notifications per IP are sent within a configured time interval, while
    still logging all connections.
    """

    def __init__(
        self,
        ports: List[int],
        notifiers: List[Notifier],
        perform_rdns: bool = False,
        flood_limit: int = 10,
        flood_interval: int = 300,
    ) -> None:
        """
        :param ports: List of TCP ports to listen on.
        :param notifiers: List of Notifier instances for sending alerts.
        :param perform_rdns: Whether to attempt a reverse DNS lookup.
        :param flood_limit: Max notifications per IP within the flood_interval. 0 = no limit.
        :param flood_interval: Flood protection interval in seconds.
        """
        self.ports = ports
        self.notifiers = notifiers
        self.perform_rdns = perform_rdns
        self.flood_limit = flood_limit
        self.flood_interval = flood_interval

        # Keep track of how many notifications we've sent per IP
        # within the current interval, plus the next reset time.
        # Example structure:
        #   self._ip_notify_count[ip] = {"count": X, "reset_time": T}
        self._ip_notify_count: Dict[str, Dict[str, float]] = {}

        self._threads: List[threading.Thread] = []
        self._stop_flag = False

    def start(self) -> None:
        """Start the honeypot by binding to each port in a separate thread."""
        for port in self.ports:
            thread = threading.Thread(
                target=self._listen_on_port, args=(port,), daemon=True
            )
            thread.start()
            self._threads.append(thread)
            logging.info(f"Started listening on port {port}")

        logging.info("Honeypot is running. Press Ctrl+C to stop.")

        # Keep main thread alive; if interrupted, set stop flag
        try:
            while not self._stop_flag:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.warning("Keyboard interrupt received. Shutting down.")
            self.stop()

    def stop(self) -> None:
        """Signal the honeypot to stop."""
        self._stop_flag = True
        logging.info("Stopping honeypot...")

    def _listen_on_port(self, port: int) -> None:
        """Listen for incoming connections on the specified port, close them, then notify."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.listen(5)
            logging.debug(f"Socket bound to 0.0.0.0:{port}")
        except Exception as e:
            logging.error(f"Error binding to port {port}: {e}")
            return

        while not self._stop_flag:
            try:
                s.settimeout(1)  # short timeout to allow stop checking
                conn, addr = s.accept()
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"Error accepting connection on port {port}: {e}")
                continue

            ip, src_port = addr[0], addr[1]
            timestamp = datetime.now()
            reverse_dns = None
            if self.perform_rdns:
                reverse_dns = self._do_reverse_dns(ip)

            # Always log every incoming connection.
            logging.info(f"Incoming connection from {ip}:{src_port} on port {port}")

            # Build connection info dataclass
            conn_info = ConnectionInfo(
                timestamp=timestamp,
                ip=ip,
                input_port=port,
                output_port=src_port,
                reverse_dns=reverse_dns,
            )

            # Immediately close the connection
            conn.close()

            # Check flood protection before notifying
            if not self._flood_exceeded(ip):
                # Flood limit not exceeded; send all configured notifications
                for notifier in self.notifiers:
                    notifier.notify(conn_info)
            else:
                logging.info(
                    f"Flood limit reached for IP {ip}. No notifications sent for this connection."
                )

        s.close()

    def _do_reverse_dns(self, ip: str) -> Optional[str]:
        """Attempt a passive reverse DNS lookup for the given IP address."""
        try:
            host, _, _ = socket.gethostbyaddr(ip)
            return host
        except Exception:
            return None

    def _flood_exceeded(self, ip: str) -> bool:
        """
        Check if the given IP has exceeded the flood limit.
        Reset the counter if the interval has passed.
        If flood_limit == 0, then no limit applies.
        """
        if self.flood_limit == 0:
            # No flood protection
            return False

        now = time.time()
        data = self._ip_notify_count.setdefault(
            ip, {"count": 0, "reset_time": now + self.flood_interval}
        )

        # If current time is beyond the reset_time, reset the count and reset_time.
        if now > data["reset_time"]:
            data["count"] = 0
            data["reset_time"] = now + self.flood_interval

        # Check if count is at or above limit
        if data["count"] >= self.flood_limit:
            return True

        # If below the limit, increment the count and allow notifications
        data["count"] += 1
        return False


def parse_port_or_range(port_str: str) -> List[int]:
    """
    Parses a string representing a single port OR a port range ("1000-1005").
    e.g. "80" -> [80], "1000-1003" -> [1000, 1001, 1002, 1003].
    """
    ports = []
    if "-" in port_str:
        start, end = port_str.split("-", 1)
        try:
            start_port = int(start)
            end_port = int(end)
            ports.extend(range(start_port, end_port + 1))
        except ValueError:
            logging.error(f"Invalid port range specified: {port_str}")
    else:
        try:
            ports.append(int(port_str))
        except ValueError:
            logging.error(f"Invalid port specified: {port_str}")
    return ports


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments (with short aliases and multiple --port usage)."""
    parser = argparse.ArgumentParser(
        description="A robust honeypot that sends detailed notifications.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Ports and features
    parser.add_argument(
        "-p",
        "--port",
        action="append",
        type=str,
        help="TCP port or port range to bind. Can be specified multiple times.",
    )
    parser.add_argument(
        "-r",
        "--reverse-dns",
        action="store_true",
        help="Attempt reverse DNS lookup for each connection.",
    )

    # Email
    parser.add_argument(
        "-e",
        "--email",
        type=str,
        help="Send email notification to the given address.",
    )
    parser.add_argument(
        "-H",
        "--smtp-server",
        type=str,
        default="localhost",
        help="SMTP server hostname or IP.",
    )
    parser.add_argument(
        "-O",
        "--smtp-port",
        type=int,
        default=25,
        help="SMTP server port (default: 25).",
    )
    parser.add_argument(
        "-u",
        "--smtp-user",
        type=str,
        default=None,
        help="SMTP username for authentication.",
    )
    parser.add_argument(
        "-a",
        "--smtp-password",
        type=str,
        default=None,
        help="SMTP password for authentication.",
    )
    parser.add_argument(
        "-f",
        "--smtp-sender",
        type=str,
        default="hendrik@example.com",
        help="Email sender address.",
    )
    parser.add_argument(
        "-t",
        "--smtp-tls",
        action="store_true",
        help="Use STARTTLS for SMTP.",
    )
    parser.add_argument(
        "-x",
        "--smtp-ssl",
        action="store_true",
        help="Use SSL/TLS for the entire SMTP connection.",
    )

    # Webhook
    parser.add_argument(
        "-b", "--webhook", type=str, help="Send webhook notification to the given URL."
    )
    parser.add_argument(
        "-K",
        "--webhook-key",
        type=str,
        default=None,
        help="Optional API key to include if webhook is used.",
    )

    # Stdout
    parser.add_argument(
        "-o",
        "--stdout",
        action="store_true",
        help="Print a message to stdout for every connection.",
    )

    # Flood protection
    parser.add_argument(
        "-L",
        "--flood-limit",
        type=int,
        default=10,
        help=("Number of notifications per IP within the flood interval."),
    )
    parser.add_argument(
        "-I",
        "--flood-interval",
        type=int,
        default=300,
        help="Flood protection interval in seconds.",
    )

    # Logging
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
    """Sets up the logging configuration."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_smtp_config(args: argparse.Namespace) -> Dict[str, any]:
    """
    Combine CLI arguments with environment variables to build the final
    SMTP configuration. CLI arguments take precedence over environment vars.
    """
    config = {
        "smtp_server": args.smtp_server or os.getenv("HB_SMTP_SERVER", "localhost"),
        "smtp_port": args.smtp_port or int(os.getenv("HB_SMTP_PORT", "25")),
        "smtp_user": args.smtp_user or os.getenv("HB_SMTP_USER"),
        "smtp_password": args.smtp_password or os.getenv("HB_SMTP_PASSWORD"),
        "smtp_sender": args.smtp_sender
        or os.getenv("HB_SMTP_SENDER", "hendrik@example.com"),
        "use_tls": args.smtp_tls
        or (os.getenv("HB_SMTP_TLS", "false").lower() == "true"),
        "use_ssl": args.smtp_ssl
        or (os.getenv("HB_SMTP_SSL", "false").lower() == "true"),
    }

    # If environment has set a custom recipient, allow it to override:
    env_recipient = os.getenv("HB_SMTP_RECEIVER")
    if env_recipient and not args.email:
        config["recipient_email"] = env_recipient
    else:
        config["recipient_email"] = args.email

    return config


def main():
    """Main entry point for the honeypot script."""
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    logging.info("Initializing honeypot...")

    # Collect all ports
    all_ports: List[int] = []
    if args.port:
        for port_str in args.port:
            all_ports.extend(parse_port_or_range(port_str))
    else:
        all_ports = [8080]

    # Remove duplicates and sort
    unique_ports = sorted(set(all_ports))
    if not unique_ports:
        logging.error("No valid ports provided. Exiting.")
        sys.exit(1)

    # Build the final SMTP config from CLI + environment
    smtp_config = resolve_smtp_config(args)

    # Collect notifiers
    notifiers: List[Notifier] = []

    # Stdout Notifier
    if args.stdout:
        notifiers.append(StdoutNotifier())

    # Email Notifier
    if smtp_config["recipient_email"]:
        notifiers.append(
            EmailNotifier(
                recipient_email=smtp_config["recipient_email"],
                smtp_server=smtp_config["smtp_server"],
                smtp_port=smtp_config["smtp_port"],
                smtp_user=smtp_config["smtp_user"],
                smtp_password=smtp_config["smtp_password"],
                smtp_sender=smtp_config["smtp_sender"],
                use_tls=smtp_config["use_tls"],
                use_ssl=smtp_config["use_ssl"],
            )
        )

    # Webhook Notifier
    if args.webhook:
        notifiers.append(WebhookNotifier(args.webhook, args.webhook_key))

    if not notifiers:
        logging.warning("No notifiers configured. Only logging will be used.")

    # Start honeypot
    honeypot = Honeypot(
        ports=unique_ports,
        notifiers=notifiers,
        perform_rdns=args.reverse_dns,
        flood_limit=args.flood_limit,
        flood_interval=args.flood_interval,
    )
    honeypot.start()

    logging.info("Honeypot terminated successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
