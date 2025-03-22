#!/usr/bin/env python3

# -------------------------------------------------------
# Script: receive_honeypot_webhooks.py
#
# Description:
# A Flask-based service that receives WebhookNotifier incidents
# from the honeypots, stores them in a local SQLite database, and
# displays them on an interactive OpenStreetMap with a timeline slider.
#
# Usage:
#   ./receive_honeypot_webhooks.py [options]
#
# Options:
#   -p, --port PORT              Port to run the server on (default: 8080).
#   -H, --host HOST              Host to run the server on (default: 0.0.0.0).
#   -d, --db DB_PATH             Path to the SQLite database file (default: /data/honeypots.db).
#   -k, --incident-api-key KEY   One or more API keys to require in the "X-API-Key" header for /incident.
#   -c, --dbip-csv PATH          Path to the DB-IP LITE CSV or CSV.GZ (default: /dbip-city-lite-2025-03.csv.gz).
#   -u, --username USERNAME      Basic auth username for the interface (default: admin).
#   -P, --password PASSWORD      Basic auth password for the interface (required).
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - flask (install via: pip install flask==3.1.0)
#   - wget (install via: apt-get install -y wget)
#   - DB-IP (install via: wget https://download.db-ip.com/free/dbip-city-lite-2025-03.csv.gz -O /dbip-city-lite-2025-03.csv.gz)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
import sqlite3
import gzip
import ipaddress
from typing import List, Optional, Dict, Any, Tuple
from functools import wraps

from flask import Flask, request, jsonify, Response

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Honeypot Incidents</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H" crossorigin="anonymous">
  <style>
    html, body {
      margin: 0; padding: 0; height: 100%; width: 100%;
      font-family: Arial, sans-serif;
    }
    #map {
      position: absolute;
      top: 0; bottom: 0; width: 100%;
    }
    #controls {
      position: absolute;
      top: 10px; 
      right: 10px;
      z-index: 9999;
      display: flex;
      flex-direction: row;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 0 8px rgba(0,0,0,0.2);
    }
    .play-controls {
      display: flex;
      flex-direction: row;
      align-items: center;
    }
    button {
      background: #fff;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 16px;
      padding: 6px 8px;
      cursor: pointer;
      margin: 2px;
      transition: background 0.2s, transform 0.2s;
    }
    button:hover {
      background: #eaeaea;
    }
    button:active {
      transform: scale(0.95);
    }
    #more-settings {
      position: absolute;
      top: 60px;
      right: 10px;
      background: #fff;
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 10px;
      box-shadow: 0 0 8px rgba(0,0,0,0.2);
      display: none;
      z-index: 9999;
    }
    label {
      display: inline-block;
      min-width: 80px;
    }
    .slider-container {
      margin-top: 10px;
    }
    .toggle-btn {
      font-size: 18px;
      line-height: 24px;
      padding: 4px 8px;
    }
    #dbip-attribution {
      position: absolute;
      bottom: 0px;
      right: 175px;
      z-index: 9998;
      font-family: "Helvetica Neue", Arial, Helvetica, sans-serif;
      font-size: 0.75rem;
      line-height: 1.5;
      background: rgba(255, 255, 255, 0.8);
      padding: 0;
      padding-left: 4px;
      padding-right: 4px;
    }
    #dbip-attribution a {
      color: #333;
      text-decoration: none;
    }
    .emoji-marker {
      font-size: 24px;
      line-height: 1;
      text-align: center;
    }
    #current-range {
      position: absolute;
      top: 23px;
      right: 175px;
      z-index: 9999;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 0 8px rgba(0,0,0,0.2);
      display: none;
      font-size: 14px;
      max-width: 360px;
      word-wrap: break-word;
    }
  </style>
</head>
<body>
<div id="map"></div>

<div id="controls">
  <div class="play-controls">
    <button id="play-pause-btn">▶️</button>
    <button id="stop-btn">⏹️</button>
    <button class="toggle-btn" id="settingsToggle" title="Toggle advanced settings">⚙️</button>
  </div>
</div>

<!-- A small floating panel to show the current time window when playing -->
<div id="current-range"></div>

<div id="more-settings">
  <div>
    <label for="startDate">Start:</label>
    <input type="date" id="startDate" />
  </div>
  <div>
    <label for="endDate">End:</label>
    <input type="date" id="endDate" />
  </div>
  <div class="slider-container">
    <label for="time-step">Time step (hrs):</label>
    <input type="number" id="time-step" value="24" min="1" style="width:50px"/>
  </div>
  <div class="slider-container">
    <label for="play-speed">Playback interval (ms):</label>
    <input type="number" id="play-speed" value="1000" min="100" step="100" style="width:70px"/>
  </div>
</div>

<div id="dbip-attribution">
  <a href="https://db-ip.com" target="_blank">IP Geolocation by DB-IP</a>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH" crossorigin="anonymous"></script>
<script src="https://unpkg.com/overlapping-marker-spiderfier-leaflet@0.2.7/build/oms.js" integrity="sha384-+xGyHTBPwdxzQlyp6ZXr1jVYPqAJ2247gTscpcKaQUwRNyQo9zoXh/EDIKkLCiSK" crossorigin="anonymous"></script>
<script>
  let map;
  let markers = [];
  let timer = null;
  let isPlaying = false;
  let stepHours = 24;
  let oms; // OverlappingMarkerSpiderfier instance

  function initMap() {
    map = L.map('map').setView([20, 0], 3);

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap'
    }).addTo(map);

    // Initialize OverlappingMarkerSpiderfier
    oms = new OverlappingMarkerSpiderfier(map, {
      keepSpiderfied: true,
      nearbyDistance: 5,
      circleSpiralSwitchover: 9,
      spiralFootSeparation: 26
    });

    // Make sure only the clicked marker’s popup opens
    oms.addListener('click', function(marker) {
      marker.openPopup();
    });
    // Close all popups on spiderfy so only the final clicked one is open
    oms.addListener('spiderfy', function(spideredMarkers) {
      spideredMarkers.forEach(m => {
        m.closePopup();
      });
    });

    // Load all incidents by default
    fetchIncidentsAndRender();
  }

  function fetchIncidentsAndRender(start=null, end=null) {
    let url = '/incidents';
    if (start || end) {
      const params = new URLSearchParams();
      if (start) params.set('start', start);
      if (end) params.set('end', end);
      url += '?' + params.toString();
    }
    fetch(url)
      .then(resp => resp.json())
      .then(data => {
        clearMarkers();
        data.forEach(inc => addMarker(inc));
      })
      .catch(err => console.error(err));
  }

  function addMarker(incident) {
    // Skip if lat or lon are null/undefined
    if (incident.lat == null || incident.lon == null) {
      return;
    }

    // Use an emoji-based icon for the marker
    const markerEmoji = incident.emoji || "📍";
    const icon = L.divIcon({
      className: "emoji-marker",
      html: markerEmoji,
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    });

    const marker = L.marker([incident.lat, incident.lon], {
      title: incident.ip,
      icon: icon
    });

    // Add a small two-line tooltip: date (timestamp) and ip:port
    marker.bindTooltip(`
${incident.timestamp}<br>
${incident.ip}:${incident.input_port ?? ''}
`);

    // Full popup content with more detailed info
    const popupContent = `
      <b>Timestamp:</b> ${incident.timestamp}<br/>
      <b>Protocol:</b> ${incident.protocol}<br/>
      <b>IP:</b> ${incident.ip}<br/>
      <b>Input Port:</b> ${incident.input_port}<br/>
      <b>Output Port:</b> ${incident.output_port}<br/>
      <b>Reverse DNS:</b> ${incident.reverse_dns || ''}<br/>
      <b>Raw Packet:</b><br/>
      <pre style="max-width:300px;white-space:pre-wrap;">${incident.raw_packet || ''}</pre>
    `;
    marker.bindPopup(popupContent);

    // Add marker to the map and OverlappingMarkerSpiderfier
    marker.addTo(map);
    oms.addMarker(marker);
    markers.push(marker);
  }

  function clearMarkers() {
    markers.forEach(m => {
      map.removeLayer(m);
    });
    markers = [];
  }

  function onPlayPause() {
    const playPauseBtn = document.getElementById('play-pause-btn');
    const currentRangeEl = document.getElementById('current-range');

    if (isPlaying) {
      // Pause
      isPlaying = false;
      if (timer) clearInterval(timer);
      timer = null;
      playPauseBtn.textContent = '▶️';
      return;
    }

    // If we're here, the user clicked "Play"
    isPlaying = true;
    playPauseBtn.textContent = '⏸️';

    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    stepHours = parseInt(document.getElementById('time-step').value) || 24;
    let playInterval = parseInt(document.getElementById('play-speed').value) || 1000;

    let startTime = startDate ? new Date(startDate + 'T00:00:00Z').getTime() : 0;
    let endTime = endDate ? new Date(endDate + 'T23:59:59Z').getTime() : (Date.now() + 24*3600000);

    let currentTime = startTime;

    function stepFunction() {
      if (!isPlaying) return;

      if (currentTime > endTime) {
        clearInterval(timer);
        timer = null;
        isPlaying = false;
        playPauseBtn.textContent = '▶️';
        currentRangeEl.style.display = 'none';
        fetchIncidentsAndRender();
        return;
      }

      const rangeStart = new Date(currentTime);
      const nextTime = currentTime + (stepHours * 3600000);
      const rangeEnd = new Date(nextTime - 1);

      const isoStart = rangeStart.toISOString();
      const isoEnd = rangeEnd.toISOString();
      fetchIncidentsAndRender(isoStart, isoEnd);

      // Update the time window display
      const formattedStart = rangeStart.toISOString().slice(0,16).replace('T',' ');
      const formattedEnd = rangeEnd.toISOString().slice(0,16).replace('T',' ');
      currentRangeEl.textContent = `Time Window: ${formattedStart} → ${formattedEnd}`;
      currentRangeEl.style.display = 'block';

      currentTime = nextTime;
    }

    // Do the first fetch immediately
    stepFunction();

    // Then wait playInterval for subsequent steps
    timer = setInterval(stepFunction, playInterval);
  }

  function onStop() {
    const playPauseBtn = document.getElementById('play-pause-btn');
    const currentRangeEl = document.getElementById('current-range');
    isPlaying = false;
    if (timer) clearInterval(timer);
    timer = null;
    playPauseBtn.textContent = '▶️';
    currentRangeEl.style.display = 'none';
    // Show all incidents again
    fetchIncidentsAndRender();
  }

  function onManualRange() {
    if (isPlaying) return;
    const start = document.getElementById('startDate').value;
    const end = document.getElementById('endDate').value;
    if (start || end) {
      fetchIncidentsAndRender(start, end);
    } else {
      fetchIncidentsAndRender();
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('play-pause-btn').addEventListener('click', onPlayPause);
    document.getElementById('stop-btn').addEventListener('click', onStop);

    const settingsToggler = document.getElementById('settingsToggle');
    settingsToggler.addEventListener('click', () => {
      const moreSettings = document.getElementById('more-settings');
      if (moreSettings.style.display === '' || moreSettings.style.display === 'none') {
        moreSettings.style.display = 'block';
      } else {
        moreSettings.style.display = 'none';
      }
    });

    document.getElementById('startDate').addEventListener('change', onManualRange);
    document.getElementById('endDate').addEventListener('change', onManualRange);

    initMap();
  });
</script>
</body>
</html>
"""

PORT_EMOJI_MAPPING = {
    (0,): "❓",  # Unknown / catch-all for port 0
    (21,): "🐣",  # FTP
    (22, 23): "🔑",  # SSH & Telnet
    (25, 465, 587, 2525): "✉️",  # SMTP (various default mail ports)
    (110, 995): "📩",  # POP3 (plain and SSL)
    (143, 993): "📩",  # IMAP (plain and SSL)
    (53,): "📖",  # DNS
    (80, 8000, 8080, 8888): "🌐",  # HTTP/alternative
    (443, 8443): "🔐",  # HTTPS/alternative
    (445,): "🪟",  # SMB
    (
        3306,
        5432,
        1521,
        1433,
        27017,
        6379,
    ): "🗄️",  # DB (MySQL, PostgreSQL, Oracle, MSSQL, MongoDB, Redis)
    (1337,): "💩",  # Debug
}

DBIP_RANGES: List[Tuple[int, int, float, float]] = []  # (start_int, end_int, lat, lon)


def port_to_emoji(port_num: Optional[int]) -> str:
    """
    Returns the emoji associated with a specific port number.
    If multiple ports appear in one tuple, they share the same emoji.
    Falls back to "📍" if the port is not matched, or "❓" if None.
    """
    if port_num is None:
        return "❓"
    for port_tuple, emoji in PORT_EMOJI_MAPPING.items():
        if port_num in port_tuple:
            return emoji
    return "📍"


def ip_str_to_int(ip_str: str) -> int:
    """
    Convert an IPv4 string like '1.2.3.4' to an integer.
    Raises ValueError if not a valid IPv4.
    """
    return int(ipaddress.IPv4Address(ip_str))


def scale_ipv4_to_longitude(ip_val: int) -> float:
    """
    Scale an IPv4 integer (0 to 4294967295) to a longitude range [-180, 180].
    """
    # Full range of IPv4 addresses: 0 to 2^32 - 1 = 4294967295
    min_val = 0
    max_val = 4294967295
    # Linear interpolation to [-180, 180]
    return -180 + (ip_val - min_val) * 360.0 / (max_val - min_val)


def load_dbip_ranges(db_path: str) -> None:
    """
    Parse the DB-IP LITE CSV file into the global DBIP_RANGES list.
    Supports CSV or gzip-compressed CSV (i.e. .gz).
    """
    global DBIP_RANGES
    DBIP_RANGES.clear()

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB-IP file not found: {db_path}")

    open_func = open
    if db_path.endswith(".gz"):
        open_func = gzip.open

    line_count = 0
    with open_func(db_path, mode="rt", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line or "," not in line:
                continue

            parts = line.split(",")
            # Skip IPv6 lines (check for ':')
            if ":" in parts[0] or ":" in parts[1]:
                continue

            # We expect at least 7 parts: start_ip, end_ip, country, region, city, lat, lon
            if len(parts) < 7:
                continue

            try:
                start_int = ip_str_to_int(parts[0])
                end_int = ip_str_to_int(parts[1])
                # lat, lon are in the last two columns
                lat = float(parts[-2])
                lon = float(parts[-1])
            except ValueError:
                continue

            DBIP_RANGES.append((start_int, end_int, lat, lon))

    # Sort by start_int
    DBIP_RANGES.sort(key=lambda x: x[0])
    logging.info(
        f"Loaded {len(DBIP_RANGES)} IPv4 ranges from DB-IP LITE (parsed {line_count} lines)."
    )


def geolocate_ip(ip: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Attempt to retrieve approximate lat/lon from the in-memory DBIP_RANGES.
    - If IP is not global (LAN, loopback, etc.), place it at lat=-80 but
      spread longitude from -180 to 180 based on the IP integer.
    - Otherwise, use binary search to find the address in DBIP_RANGES.
    Returns (lat, lon) or (None, None) if not found / not IPv4.
    """
    try:
        addr = ipaddress.ip_address(ip)
        if addr.version != 4:
            logging.warning(f"The IP address {ip} is odd")
            return (None, None)

        ip_val = int(addr)

        # If the IP is not global (e.g., private, loopback, link-local), we spread them at latitude -80
        # and compute longitude proportionally across the full IPv4 space.
        if not addr.is_global:
            return (-80.0, scale_ipv4_to_longitude(ip_val))

    except ValueError as e:
        logging.warning(f"The IP address {ip} caused a value error: {e}")
        return (None, None)

    # Binary search in DBIP_RANGES for global IPs
    low, high = 0, len(DBIP_RANGES) - 1
    while low <= high:
        mid = (low + high) // 2
        start_int, end_int, lat, lon = DBIP_RANGES[mid]
        if ip_val < start_int:
            high = mid - 1
        elif ip_val > end_int:
            low = mid + 1
        else:
            return (lat, lon)

    return (None, None)


def init_db(db_path: str) -> None:
    """
    Initializes the SQLite database and creates the 'incidents' table if needed.
    """
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.OperationalError as e:
        logging.error(f"Failed to open or create SQLite database at '{db_path}': {e}")
        sys.exit(1)

    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                protocol TEXT,
                ip TEXT,
                input_port INTEGER,
                output_port INTEGER,
                reverse_dns TEXT,
                raw_packet TEXT,
                remote_addr TEXT,
                lat REAL,
                lon REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except sqlite3.OperationalError as e:
        logging.error(
            f"Failed to create 'incidents' table in database '{db_path}': {e}"
        )
        sys.exit(1)
    finally:
        conn.close()


def insert_incident(db_path: str, data: Dict[str, Any], remote_addr: str) -> None:
    """
    Insert a single incident record into the database.
    """
    lat, lon = geolocate_ip(data["ip"])
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO incidents(timestamp, protocol, ip, input_port, output_port,
                                  reverse_dns, raw_packet, remote_addr, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("timestamp"),
                data.get("protocol"),
                data.get("ip"),
                data.get("input_port"),
                data.get("output_port"),
                data.get("reverse_dns"),
                data.get("raw_packet"),
                remote_addr,
                lat,
                lon,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def query_incidents(
    db_path: str, start: Optional[str], end: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Query the 'incidents' table, optionally filtering by timestamp range.
    Using datetime() around both sides to ensure string timestamps that
    look like ISO 8601 (e.g., 2025-03-12T14:00:00Z) compare properly.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        sql = """
            SELECT id, timestamp, protocol, ip, input_port, output_port,
                   reverse_dns, raw_packet, lat, lon
            FROM incidents
        """
        params: List[Any] = []
        conditions: List[str] = []

        if start:
            conditions.append("datetime(timestamp) >= datetime(?)")
            params.append(start)
        if end:
            conditions.append("datetime(timestamp) <= datetime(?)")
            params.append(end)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC"

        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        for row in rows:
            (row_id, ts, protocol, ip, in_port, out_port, rdns, raw_pkt, lat, lon) = row
            results.append(
                {
                    "id": row_id,
                    "timestamp": ts,
                    "protocol": protocol,
                    "ip": ip,
                    "input_port": in_port,
                    "output_port": out_port,
                    "reverse_dns": rdns,
                    "raw_packet": raw_pkt,
                    "lat": lat,
                    "lon": lon,
                    "emoji": port_to_emoji(in_port),
                }
            )
        return results
    finally:
        conn.close()


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments from sys.argv into an argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Receives webhook incidents from a honeypot and displays them on a map."
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080).",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0).",
    )
    parser.add_argument(
        "-d",
        "--db",
        type=str,
        default="/data/honeypots.db",
        help="Path to the SQLite database (default: /data/honeypots.db).",
    )
    parser.add_argument(
        "-k",
        "--incident-api-key",
        action="append",
        help="Incident API key(s) required in the 'X-API-Key' header. Can be specified multiple times.",
    )
    parser.add_argument(
        "-c",
        "--dbip-csv",
        type=str,
        default="/dbip-city-lite-2025-03.csv.gz",
        help="Path to the DB-IP LITE CSV or CSV.GZ file (default: /dbip-city-lite-2025-03.csv.gz).",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default="admin",
        help="Basic auth username for the interface (default: admin).",
    )
    parser.add_argument(
        "-P",
        "--password",
        type=str,
        required=True,
        help="Basic auth password for the interface (required).",
    )
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
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def require_basic_auth(username: str, password: str):
    """
    Returns a decorator that enforces Basic Auth using the given username and password.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            auth = request.authorization
            if not auth or auth.username != username or auth.password != password:
                return Response(
                    "Unauthorized",
                    401,
                    {"WWW-Authenticate": 'Basic realm="Login Required"'},
                )
            return f(*args, **kwargs)

        return wrapper

    return decorator


def create_app(
    db_path: str,
    incident_api_keys: Optional[List[str]],
    basic_auth_username: str,
    basic_auth_password: str,
) -> Flask:
    """
    Creates and configures the Flask application with routes.
    """
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    @require_basic_auth(basic_auth_username, basic_auth_password)
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.route("/incidents", methods=["GET"])
    @require_basic_auth(basic_auth_username, basic_auth_password)
    def list_incidents() -> Response:
        start = request.args.get("start")
        end = request.args.get("end")
        data = query_incidents(db_path, start, end)
        return jsonify(data)

    @app.route("/incident", methods=["POST"])
    def receive_incident() -> Response:
        if incident_api_keys:
            header_key = request.headers.get("X-API-Key", None)
            if header_key not in incident_api_keys:
                return jsonify({"error": "Unauthorized"}), 401

        content_type = request.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        payload = request.json
        required_fields = {"timestamp", "protocol", "ip"}
        if not all(fld in payload for fld in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        if "input_port" not in payload:
            payload["input_port"] = None

        remote_addr = request.remote_addr or "unknown"
        try:
            insert_incident(db_path, payload, remote_addr)
            logging.info(f"Incident from {remote_addr} stored.")
            return jsonify({"status": "ok"}), 201
        except Exception as e:
            logging.error(f"Failed to insert incident: {e}")
            return jsonify({"error": str(e)}), 500

    return app


def main() -> None:
    """
    The main entry point to parse arguments, initialize the database, and run the Flask app.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    logging.info("Loading DB-IP geolocation data.")
    load_dbip_ranges(args.dbip_csv)

    logging.info("Initializing SQLite database.")
    init_db(args.db)

    app = create_app(
        db_path=args.db,
        incident_api_keys=args.incident_api_key,
        basic_auth_username=args.username,
        basic_auth_password=args.password,
    )
    logging.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
