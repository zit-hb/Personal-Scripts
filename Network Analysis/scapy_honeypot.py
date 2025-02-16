#!/usr/bin/env python3

# -------------------------------------------------------
# Script: scapy_honeypot.py
#
# Description:
# A flexible honeypot script that uses Scapy to listen
# passively for incoming TCP, UDP, or ICMP traffic on
# specified ports (and optionally, multiple port/type ranges).
#
# For TCP/UDP, you can specify port(s) or none for all:
#   - ":tcp" ⇒ all TCP ports
#   - "80:tcp" ⇒ only TCP port 80
#   - "1337-1339:udp" ⇒ UDP ports 1337 through 1339
#
# For ICMP, the "port" portion is interpreted as the
# ICMP type. For example:
#   - ":icmp" ⇒ all ICMP
#   - "8:icmp" ⇒ only ICMP type 8 (ping/echo request)
#
# Any incoming connection-like event (e.g., TCP SYN,
# any UDP packet, or relevant ICMP type) triggers a
# detailed notification (via one or more configured
# notifiers).
#
# Usage:
#   ./scapy_honeypot.py [options]
#
# Options:
#   -p, --port PORT_SPEC         TCP/UDP/ICMP port (or type) or range
#                                in "[start[-end]]:protocol" format.
#                                Can be specified multiple times
#                                (default: 8080:tcp).
#   -r, --reverse-dns            Attempt reverse DNS lookup (passive).
#   -e, --email EMAIL            Enable email notification to this recipient.
#   -H, --smtp-server SERVER     SMTP server hostname or IP (default: localhost).
#   -O, --smtp-port PORT         SMTP server port (default: 25).
#   -u, --smtp-user USERNAME     SMTP username (if auth is required).
#   -a, --smtp-password PASSWORD SMTP password (if auth is required).
#   -f, --smtp-sender ADDRESS    Sender email address (default: hendrik@example.com).
#   -t, --smtp-tls               Use STARTTLS after connecting.
#   -x, --smtp-ssl               Use SSL/TLS for the entire SMTP connection.
#   -b, --webhook URL            Enable webhook notification to the given URL.
#   -o, --stdout                 Print a notification to stdout for each connection.
#   -L, --flood-limit INT        Number of notifications per IP within the flood
#                                interval. 0 = unlimited (default: 10).
#   -I, --flood-interval INT     Flood protection interval in seconds (default: 300).
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#   -h, --help                   Show this help message and exit.
#
# Template: ubuntu24.04
#
# Requirements:
#   - scapy (install via: pip install scapy==2.6.1)
#   - requests (install via: pip install requests==2.32.3)
#   - pcap (install via: apt-get install -y libpcap-dev)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Any
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

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Packet
except ImportError:
    print("Scapy is not installed. Please install scapy and rerun.")
    sys.exit(1)


@dataclass
class ConnectionInfo:
    """Data class holding information about each connection."""

    timestamp: datetime
    ip: str
    # For TCP/UDP, input_port = destination port;
    # for ICMP, input_port = ICMP type;
    # can be None if not applicable.
    input_port: Optional[int]
    # For TCP/UDP, output_port = source port;
    # for ICMP, this is always None.
    output_port: Optional[int]
    protocol: str
    reverse_dns: Optional[str] = None
    raw_packet: Optional[str] = None


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
            f"[{info.timestamp}] {info.protocol.upper()} connection from "
            f"{info.ip}:{info.output_port if info.output_port else 'N/A'} "
            f"to {info.input_port if info.input_port else 'N/A'} "
            f"(rDNS: {info.reverse_dns or 'N/A'})"
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

        message_body: str = (
            f"Honeypot Alert!\n\n"
            f"Timestamp: {info.timestamp}\n"
            f"Protocol: {info.protocol.upper()}\n"
            f"Client IP: {info.ip}\n"
            f"Client Port: {info.output_port if info.output_port else 'N/A'}\n"
            f"Server Port/Type: {info.input_port if info.input_port else 'N/A'}\n"
            f"Reverse DNS: {info.reverse_dns or 'N/A'}\n"
            f"\n--- Raw Packet Data ---\n{info.raw_packet}\n"
        )

        msg: MIMEText = MIMEText(message_body)
        msg["Subject"] = "Honeypot Connection Alert"
        msg["From"] = self.smtp_sender
        msg["To"] = self.recipient_email

        try:
            smtp_class = smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP
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

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def notify(self, info: ConnectionInfo) -> None:
        """Send an HTTP POST request with the connection info as JSON."""
        if not requests:
            logging.error("requests module is not available. Cannot send webhook.")
            return

        payload: Dict[str, Any] = {
            "timestamp": str(info.timestamp),
            "protocol": info.protocol.upper(),
            "ip": info.ip,
            "input_port_or_type": info.input_port if info.input_port else "N/A",
            "output_port": info.output_port if info.output_port else "N/A",
            "reverse_dns": info.reverse_dns or "N/A",
            "raw_packet": info.raw_packet,
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            if response.status_code == 200:
                logging.info(f"Webhook sent to {self.webhook_url}")
            else:
                logging.warning(
                    f"Webhook to {self.webhook_url} returned status code {response.status_code}"
                )
        except Exception as e:
            logging.error(f"Error sending webhook to {self.webhook_url}: {e}")


class Honeypot:
    """
    A honeypot that passively listens (via Scapy) on specified ports/protocols.
    Whenever a matching packet is seen that indicates a new "connection-like"
    event (for TCP: SYN; for UDP: any packet; for ICMP: user-defined types),
    a ConnectionInfo object is passed to all notifiers. Flood protection ensures
    that only a certain number of notifications per IP are sent within a
    configured time interval, while still logging all events.
    """

    def __init__(
        self,
        port_protocols: List[Tuple[List[int], str]],
        notifiers: List[Notifier],
        perform_rdns: bool = False,
        flood_limit: int = 10,
        flood_interval: int = 300,
    ) -> None:
        """
        :param port_protocols: A list of tuples: ([ports or types], protocol).
                              - If protocol is "icmp", the list holds ICMP types.
                                An empty list means "all ICMP".
                              - If protocol is "tcp"/"udp", the list holds ports.
                                An empty list means "all TCP/UDP".
        :param notifiers: List of Notifier instances for sending alerts.
        :param perform_rdns: Whether to attempt a reverse DNS lookup.
        :param flood_limit: Max notifications per IP within the flood_interval. 0 = no limit.
        :param flood_interval: Flood protection interval in seconds.
        """
        self.port_protocols: List[Tuple[List[int], str]] = port_protocols
        self.notifiers: List[Notifier] = notifiers
        self.perform_rdns: bool = perform_rdns
        self.flood_limit: int = flood_limit
        self.flood_interval: int = flood_interval

        # Keep track of how many notifications we've sent per IP
        # within the current interval, plus the next reset time.
        # Example:
        #   self._ip_notify_count[ip] = {"count": X, "reset_time": T}
        self._ip_notify_count: Dict[str, Dict[str, float]] = {}

        self._stop_flag: bool = False

    def start(self) -> None:
        """
        Start the honeypot by sniffing packets in a background thread.
        We build a BPF filter covering all requested ports/protocols/types
        using "port range" if needed.
        """
        logging.info("Generating BPF filter for scapy...")

        bpf_expressions: List[str] = []
        for values, proto in self.port_protocols:
            proto_lower = proto.lower()
            if proto_lower in ("tcp", "udp"):
                if not values:
                    # e.g. ":tcp" => all tcp
                    bpf_expressions.append(proto_lower)
                else:
                    bpf_expressions.append(self._build_tcp_udp_bpf(proto_lower, values))
            elif proto_lower == "icmp":
                if not values:
                    bpf_expressions.append("icmp")
                else:
                    # For ICMP, we keep the small expansions for types
                    type_exprs = [f"(icmp and icmp[0] = {t})" for t in values]
                    bpf_expressions.append(" or ".join(type_exprs))
            else:
                logging.warning(f"Unsupported protocol: {proto}. Skipping.")

        # Remove any empty expressions
        bpf_expressions = [expr for expr in bpf_expressions if expr.strip()]

        if not bpf_expressions:
            logging.error("No valid port/protocol filters specified. Exiting.")
            sys.exit(1)

        final_filter: str = " or ".join(f"({expr})" for expr in bpf_expressions)
        logging.info(f"Using BPF filter: {final_filter}")

        def stop_sniff_filter(_: Packet) -> bool:
            return self._stop_flag

        logging.info("Starting scapy sniff. Press Ctrl+C to stop.")
        try:
            sniff(
                filter=final_filter,
                prn=self._packet_handler,
                store=False,
                stop_filter=stop_sniff_filter,
            )
        except KeyboardInterrupt:
            logging.warning("Keyboard interrupt received. Shutting down.")
            self.stop()

        logging.info("Scapy sniff stopped.")

    def stop(self) -> None:
        """Signal the honeypot to stop."""
        self._stop_flag = True
        logging.info("Stopping honeypot...")

    def _build_tcp_udp_bpf(self, proto: str, ports: List[int]) -> str:
        """
        Given "tcp" or "udp" and a list of ports (which might be multiple discrete
        or a single contiguous range), build a BPF expression using either
        'portrange' for a contiguous range or 'port' for single ports.

        If multiple discrete ports or multiple ranges are needed, we join them
        with ' or '.
        """
        if not ports:
            return proto

        # We'll detect contiguous sequences in "ports". For each sequence,
        # if it's length 1, do "proto port X"; if bigger, do "proto portrange start-end".
        # Then we join them with " or ".
        sorted_ports = sorted(ports)
        expressions: List[str] = []

        start = sorted_ports[0]
        prev = start

        def flush_range(s: int, e: int) -> None:
            if s == e:
                expressions.append(f"{proto} port {s}")
            else:
                expressions.append(f"{proto} portrange {s}-{e}")

        for p in sorted_ports[1:]:
            if p == prev + 1:
                # contiguous
                prev = p
            else:
                # flush previous range
                flush_range(start, prev)
                start = p
                prev = p

        # flush last range
        flush_range(start, prev)

        if len(expressions) == 1:
            return expressions[0]
        return " or ".join(f"({expr})" for expr in expressions)

    def _packet_handler(self, packet: Packet) -> None:
        """
        Callback for each packet that passes the filter. We parse it
        and decide if it is a "new connection-like" event. If so,
        we generate a ConnectionInfo and notify.
        """
        if IP not in packet:
            return  # we only care about IP-based packets

        ip_src: str = packet[IP].src
        ip_dst: str = packet[IP].dst
        timestamp: datetime = datetime.now()
        raw_str: str = packet.summary() + "\n" + str(packet.show(dump=True))

        protocol: Optional[str] = None
        input_port: Optional[int] = None
        output_port: Optional[int] = None

        # Detect protocol
        if TCP in packet:
            protocol = "tcp"
            tcp_flags: int = packet[TCP].flags
            # Only treat a packet as a new "connection" if the SYN flag is set
            if tcp_flags & 0x02 == 0:
                return
            output_port = packet[TCP].sport
            input_port = packet[TCP].dport

        elif UDP in packet:
            protocol = "udp"
            output_port = packet[UDP].sport
            input_port = packet[UDP].dport

        elif ICMP in packet:
            protocol = "icmp"
            # The "input_port" field is used as "ICMP type"
            input_port = packet[ICMP].type

        else:
            return  # Not a protocol we handle

        # Attempt reverse DNS if requested
        reverse_dns: Optional[str] = None
        if self.perform_rdns:
            reverse_dns = self._do_reverse_dns(ip_src)

        log_port_type: str = str(input_port) if input_port is not None else "N/A"
        logging.info(
            f"Detected {protocol.upper()} from {ip_src} to {ip_dst}:{log_port_type}"
        )

        conn_info = ConnectionInfo(
            timestamp=timestamp,
            ip=ip_src,
            input_port=input_port,
            output_port=output_port,
            protocol=protocol,
            reverse_dns=reverse_dns,
            raw_packet=raw_str,
        )

        if not self._flood_exceeded(ip_src):
            for notifier in self.notifiers:
                notifier.notify(conn_info)
        else:
            logging.info(
                f"Flood limit reached for IP {ip_src}. No notifications sent for this connection."
            )

    def _do_reverse_dns(self, ip: str) -> Optional[str]:
        """Attempt a passive reverse DNS lookup for the given IP address."""
        import socket

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
            return False

        now: float = time.time()
        data: Dict[str, float] = self._ip_notify_count.setdefault(
            ip, {"count": 0, "reset_time": now + self.flood_interval}
        )

        # If current time is beyond the reset_time, reset
        if now > data["reset_time"]:
            data["count"] = 0
            data["reset_time"] = now + self.flood_interval

        if data["count"] >= self.flood_limit:
            return True

        data["count"] += 1
        return False


def parse_port_protocol(port_str: str) -> Tuple[List[int], str]:
    """
    Parses a string "[start[-end]]:protocol". For ICMP, 'start[-end]' is the ICMP type or range;
    for TCP/UDP, empty means all ports, or we parse single ports or port ranges.
    """
    if ":" not in port_str:
        logging.error(f"Invalid port+protocol format (missing colon): {port_str}")
        return ([], "")

    port_part, proto_part = port_str.split(":", 1)
    proto_part = proto_part.lower().strip()
    port_part = port_part.strip()

    if proto_part not in ("tcp", "udp", "icmp"):
        logging.error(f"Unsupported protocol '{proto_part}' in: {port_str}")
        return ([], "")

    # Handle ICMP
    if proto_part == "icmp":
        if not port_part:
            # e.g. ":icmp" => all icmp
            return ([], "icmp")
        if "-" in port_part:
            # Range of types
            try:
                start_str, end_str = port_part.split("-", 1)
                start_t = int(start_str)
                end_t = int(end_str)
                return (list(range(start_t, end_t + 1)), "icmp")
            except ValueError:
                logging.error(f"Invalid ICMP type range specified: {port_part}")
                return ([], "icmp")
        else:
            # Single type
            try:
                icmp_type = int(port_part)
                return ([icmp_type], "icmp")
            except ValueError:
                logging.error(f"Invalid ICMP type specified: {port_part}")
                return ([], "icmp")

    # TCP/UDP:
    if not port_part:
        # e.g. ":tcp" => all tcp
        # e.g. ":udp" => all udp
        return ([], proto_part)

    if "-" in port_part:
        # Range
        try:
            start_port, end_port = port_part.split("-", 1)
            start_p = int(start_port)
            end_p = int(end_port)
            return (list(range(start_p, end_p + 1)), proto_part)
        except ValueError:
            logging.error(f"Invalid port range specified: {port_part}")
            return ([], proto_part)
    else:
        # Single port
        try:
            single_p = int(port_part)
            return ([single_p], proto_part)
        except ValueError:
            logging.error(f"Invalid port specified: {port_part}")
            return ([], proto_part)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments (with short aliases and multiple --port usage)."""
    parser = argparse.ArgumentParser(
        description="A Scapy-based honeypot that passively captures connection attempts.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--port",
        action="append",
        type=str,
        help=(
            'TCP/UDP/ICMP port (or type) or range in "[start[-end]]:protocol" format. '
            "Can be specified multiple times."
        ),
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
        "-b",
        "--webhook",
        type=str,
        help="Send webhook notification to the given URL.",
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
        help="Number of notifications per IP within the flood interval. 0 = unlimited.",
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


def resolve_smtp_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Combine CLI arguments with environment variables to build the final
    SMTP configuration. CLI arguments take precedence over environment vars.
    """
    config: Dict[str, Any] = {
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

    env_recipient: Optional[str] = os.getenv("HB_SMTP_RECEIVER")
    if env_recipient and not args.email:
        config["recipient_email"] = env_recipient
    else:
        config["recipient_email"] = args.email

    return config


def main() -> None:
    args: argparse.Namespace = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    logging.info("Initializing scapy honeypot...")

    port_protocols: List[Tuple[List[int], str]] = []
    if args.port:
        for port_str in args.port:
            values, proto = parse_port_protocol(port_str)
            if proto:
                port_protocols.append((values, proto))
    else:
        # Default to 8080:tcp
        port_protocols.append(([8080], "tcp"))

    smtp_config: Dict[str, Any] = resolve_smtp_config(args)

    notifiers: List[Notifier] = []

    if args.stdout:
        notifiers.append(StdoutNotifier())

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

    if args.webhook:
        notifiers.append(WebhookNotifier(args.webhook))

    if not notifiers:
        logging.warning("No notifiers configured. Only logging will be used.")

    honeypot = Honeypot(
        port_protocols=port_protocols,
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
