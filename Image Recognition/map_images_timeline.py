#!/usr/bin/env python3

# -------------------------------------------------------
# Script: map_images_timeline.py
#
# Description:
# Starts a small Flask web application that shows a Leaflet
# map with every image in the chosen directory rendered as a
# miniature thumbnail at its GPS position.
#
# A timeline control (play button + slider) animates the map
# through time: markers appear as the timeline passes the
# moment their photo was taken (EXIF DateTimeOriginal). The
# timeline starts at the earliest photo and ends at the
# latest one. By default one year of photos is played back
# in one minute of real time.
#
# Usage:
#   ./map_images_timeline.py [options] directory
#
# Arguments:
#   - [directory]: Path to the directory containing images with metadata.
#
# Options:
#   -p, --port PORT         Port to run the server on (default: 5000).
#   -H, --host HOST         Host to run the server on (default: 0.0.0.0).
#   -t, --title TITLE       Page title (default: Hendrik's Image Map).
#   -q, --quality QUALITY   JPEG quality for the *full* images (default: 85).
#   -TW, --thumb-width PX   Thumbnail width in pixels (default: 600).
#   -TH, --thumb-height PX  Thumbnail height in pixels (default: 400).
#   -S, --speed MINUTES     Playback speed: real minutes per photo year
#                           (default: 1.0, i.e. 1 year per minute).
#   -L, --locate            Enable browser geolocation button on the map.
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - Flask (install via: pip install flask==3.1.0)
#   - Pillow (install via: pip install Pillow==11.1.0)
#   - piexif (install via: pip install piexif==1.1.3)
#
# -------------------------------------------------------
# © 2026 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import hashlib
import io
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from flask import Flask, Response, abort, render_template_string
from PIL import Image, ImageOps
import piexif


TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{{ page_title }}</title>

  <!-- Leaflet core -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H"
    crossorigin="anonymous"
  />

  <!-- Marker-cluster plugin -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"
    integrity="sha384-pmjIAcz2bAn0xukfxADbZIb3t8oRT9Sv0rvO+BR5Csr6Dhqq+nZs59P0pPKQJkEV"
    crossorigin="anonymous"
  />
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"
    integrity="sha384-wgw+aLYNQ7dlhK47ZPK7FRACiq7ROZwgFNg0m04avm4CaXS+Z9Y7nMu8yNjBKYC+"
    crossorigin="anonymous"
  />

  <style>
    html, body { height:100%; margin:0; }
    #map       { height:100%; width:100%; position:relative; }

    /* Thumbnail popup */
    .thumb-popup { position:relative; }
    .thumb-popup img {
      display:block;
      border-radius:6px;
      box-shadow:0 2px 6px rgba(0,0,0,.4);
      cursor:pointer;
    }
    .thumb-popup .close-btn {
      position:absolute;
      top:-8px; right:-8px;
      width:20px; height:20px;
      border-radius:50%;
      background:#fff;
      box-shadow:0 1px 3px rgba(0,0,0,.5);
      font:16px/20px sans-serif;
      text-align:center;
      cursor:pointer;
      user-select:none;
    }

    /* Timeline control bar */
    #timeline {
      position:absolute;
      left:50%;
      bottom:18px;
      transform:translateX(-50%);
      display:flex;
      align-items:center;
      gap:10px;
      width:min(720px, calc(100% - 40px));
      padding:8px 14px;
      background:rgba(255,255,255,.92);
      border-radius:8px;
      box-shadow:0 2px 8px rgba(0,0,0,.35);
      font:13px/1.4 sans-serif;
      z-index:1000;
    }
    #timeline button {
      width:34px; height:34px;
      flex:none;
      border:none;
      border-radius:50%;
      background:#2b6cb0;
      color:#fff;
      font-size:15px;
      cursor:pointer;
    }
    #timeline input[type=range] {
      flex:1;
      min-width:0;
      cursor:pointer;
    }
    #timeline .date-label {
      flex:none;
      min-width:86px;
      text-align:right;
      font-variant-numeric:tabular-nums;
      color:#222;
    }
  </style>
</head>
<body>
  <div id="map"></div>

  {% if has_timeline %}
  <div id="timeline">
    <button id="play-btn" title="Play">&#9654;</button>
    <input id="time-slider" type="range" />
    <span class="date-label" id="date-label"></span>
  </div>
  {% endif %}

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH"
    crossorigin="anonymous">
  </script>
  <script
    src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster-src.js"
    integrity="sha384-xLgzMQOvDhPE6lQoFpJJOFU2aMYsKD5eSSt9q3aR1RREx3Y+XsnqtSDZd+PhAcob"
    crossorigin="anonymous">
  </script>

  <script>
    const map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 21,
      attribution: '© OpenStreetMap'
    }).addTo(map);

    /* Group markers to prevent overlap */
    const cluster = L.markerClusterGroup({
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false,
      maxClusterRadius: 80
    });
    map.addLayer(cluster);

    const imagesData = {{ images_data|tojson }};
    const bounds = L.latLngBounds([]);

    /* Markers that participate in the timeline animation */
    const timedMarkers = [];   // { marker, ts }

    imagesData.forEach(img => {
      if (img.lat !== null && img.lon !== null) {
        const marker = L.marker([img.lat, img.lon]);
        bounds.extend([img.lat, img.lon]);

        marker.on('click', () => {
          const popupHtml = `
            <div class="thumb-popup">
              <span class="close-btn">&times;</span>
              <img
                src="/thumbnail/${img.sha256}"
                width="${img.thumb_w}"
                height="${img.thumb_h}"
                alt="thumbnail"
              />
            </div>`;

          const popup = L.popup({
              closeButton: false,
              offset: [0, -10],
              className: 'thumb-popup-leaflet',
              maxWidth: 820
            })
            .setLatLng(marker.getLatLng())
            .setContent(popupHtml)
            .openOn(map);

          /* After the popup is in the DOM, wire up handlers */
          setTimeout(() => {
            const container = popup.getElement();
            if (!container) return;

            const close = container.querySelector('.close-btn');
            const image = container.querySelector('img');

            if (close) close.addEventListener('click', () => map.closePopup(popup));
            if (image) image.addEventListener('click', () =>
              window.open(`/image/${img.sha256}`, '_blank'));
          }, 0);
        });

        if (img.timestamp !== null) {
          timedMarkers.push({ marker: marker, ts: img.timestamp });
        } else {
          /* No capture date in EXIF: always visible */
          cluster.addLayer(marker);
        }
      }
    });

    // Auto-fit (small padding keeps markers from hugging edges)
    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(0.05));
    } else {
      map.setView([20, 0], 2);  // fallback: world view
    }

    {% if has_timeline %}
    /* ---------------- Timeline animation ---------------- */

    const minTs = {{ min_timestamp|tojson }};
    const maxTs = {{ max_timestamp|tojson }};

    /* Playback speed: photo-seconds advanced per real second.
       One photo year corresponds to `minutes_per_year` real minutes. */
    const SECONDS_PER_YEAR = 365.25 * 24 * 3600;
    const playbackRate = SECONDS_PER_YEAR / ({{ minutes_per_year|tojson }} * 60);

    const playBtn   = document.getElementById('play-btn');
    const slider    = document.getElementById('time-slider');
    const dateLabel = document.getElementById('date-label');

    slider.min  = minTs;
    slider.max  = maxTs;
    slider.step = 'any';
    slider.value = maxTs;   // start with everything visible

    let currentTime = maxTs;
    let playing = false;
    let lastFrame = null;

    function formatDate(ts) {
      const d = new Date(ts * 1000);
      const pad = n => String(n).padStart(2, '0');
      return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
    }

    /* Show every timed marker whose capture time is <= t. */
    function applyTime(t) {
      const toAdd = [];
      const toRemove = [];
      timedMarkers.forEach(o => {
        const visible = cluster.hasLayer(o.marker);
        if (o.ts <= t && !visible) toAdd.push(o.marker);
        else if (o.ts > t && visible) toRemove.push(o.marker);
      });
      if (toAdd.length)    cluster.addLayers(toAdd);
      if (toRemove.length) cluster.removeLayers(toRemove);
      dateLabel.textContent = formatDate(t);
    }

    function setPlaying(state) {
      playing = state;
      playBtn.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
      playBtn.title = playing ? 'Pause' : 'Play';
      lastFrame = null;
      if (playing) requestAnimationFrame(tick);
    }

    function tick(frameTime) {
      if (!playing) return;
      if (lastFrame !== null) {
        const dt = (frameTime - lastFrame) / 1000;  // real seconds
        currentTime += dt * playbackRate;
        if (currentTime >= maxTs) {
          currentTime = maxTs;
          setPlaying(false);
        }
        slider.value = currentTime;
        applyTime(currentTime);
      }
      lastFrame = frameTime;
      if (playing) requestAnimationFrame(tick);
    }

    playBtn.addEventListener('click', () => {
      if (!playing && currentTime >= maxTs) {
        /* Restart from the beginning */
        currentTime = minTs;
        slider.value = currentTime;
        applyTime(currentTime);
      }
      setPlaying(!playing);
    });

    slider.addEventListener('input', () => {
      setPlaying(false);
      currentTime = parseFloat(slider.value);
      applyTime(currentTime);
    });

    /* Initial state: full timeline shown */
    applyTime(currentTime);
    {% endif %}

    {% if enable_geolocate %}
    // Layer group for user location markers
    const locateLayer = L.layerGroup().addTo(map);

    const locateBtn = L.DomUtil.create('button', '', map.getContainer());
    locateBtn.id = 'locate-btn';
    locateBtn.setAttribute('title', 'Locate me');
    Object.assign(locateBtn.style, {
      position: 'absolute',
      top: '0',
      right: '10px',
      background: 'transparent',
      border: 'none',
      padding: '0',
      margin: '0',
      width: '32px',
      height: '32px',
      fontSize: '32px',
      cursor: 'pointer',
      zIndex: 1000
    });
    locateBtn.innerHTML = '🌐';

    L.DomEvent.on(locateBtn, 'click', () => {
      // Clear previous location markers
      locateLayer.clearLayers();
      map.locate({ setView: true, maxZoom: 18 });
    });

    map.on('locationfound', (e) => {
      // Ensure old markers are removed
      locateLayer.clearLayers();
      const radius = e.accuracy / 2;
      // Yellow accuracy circle
      L.circle(e.latlng, {
        radius,
        color: 'yellow',
        fillColor: 'yellow',
        fillOpacity: 0.3
      }).addTo(locateLayer);
      // Yellow circle marker for precise location
      L.circleMarker(e.latlng, {
        radius: 8,
        color: '#000',
        weight: 1,
        fillColor: 'yellow',
        fillOpacity: 1
      }).addTo(locateLayer);
    });

    map.on('locationerror', (e) => {
      alert(e.message);
    });
    {% endif %}

  </script>
</body>
</html>
"""


@dataclass
class ImageMetadata:
    """
    Information stored for every photo.

    sha256      – SHA-256 hash used as a stable public identifier
    lat/lon     – GPS position in decimal degrees
    thumb_w/h   – pixel dimensions of the thumbnail
    timestamp   – capture time as Unix epoch seconds (None if unknown)
    """

    sha256: str
    lat: float
    lon: float
    thumb_w: int
    thumb_h: int
    timestamp: Optional[float]


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Display pictures on a map using their GPS EXIF metadata."
    )
    p.add_argument(
        "directory",
        help="Directory with images",
    )
    p.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="Port (default 5000)",
    )
    p.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="Host (default 0.0.0.0)",
    )
    p.add_argument(
        "-t",
        "--title",
        default="Hendrik's Image Map",
        help="Page title (default: Hendrik's Image Map)",
    )
    p.add_argument(
        "-q",
        "--quality",
        type=int,
        default=85,
        metavar="QUALITY",
        help="JPEG quality for full images (0-100, default 85)",
    )
    p.add_argument(
        "-TW",
        "--thumb-width",
        type=int,
        default=600,
        metavar="PX",
        help="Thumbnail width in pixels (default: 600)",
    )
    p.add_argument(
        "-TH",
        "--thumb-height",
        type=int,
        default=400,
        metavar="PX",
        help="Thumbnail height in pixels (default: 400)",
    )
    p.add_argument(
        "-S",
        "--speed",
        type=float,
        default=1.0,
        metavar="MINUTES",
        help="Playback speed: real minutes per photo year (default: 1.0)",
    )
    p.add_argument(
        "-L",
        "--locate",
        action="store_true",
        help="Enable browser geolocation button on the map",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="INFO logging",
    )
    p.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="DEBUG logging",
    )
    return p.parse_args()


def setup_logging(verbose: bool, debug: bool) -> None:
    """
    Sets up logging based on verbosity level.
    """
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _convert_to_degrees(
    value: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
) -> float:
    """
    Converts GPS coordinates from EXIF format to decimal degrees.
    """
    d = value[0][0] / value[0][1] if value[0][1] else 0
    m = value[1][0] / value[1][1] if value[1][1] else 0
    s = value[2][0] / value[2][1] if value[2][1] else 0
    return d + m / 60.0 + s / 3600.0


def _extract_gps_info(exif: dict) -> Tuple[float, float]:
    """
    Extracts GPS coordinates from EXIF data.
    """
    gps = exif.get("GPS", {})
    if not gps:
        raise ValueError("No GPS data found in EXIF")

    lat_val = gps.get(piexif.GPSIFD.GPSLatitude)
    lon_val = gps.get(piexif.GPSIFD.GPSLongitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)

    if not (lat_val and lon_val and lat_ref and lon_ref):
        raise ValueError("Incomplete GPS data in EXIF")

    lat = _convert_to_degrees(lat_val)
    lon = _convert_to_degrees(lon_val)
    if lat_ref in [b"S", "S"]:
        lat = -lat
    if lon_ref in [b"W", "W"]:
        lon = -lon
    return lat, lon


def _extract_timestamp(exif: dict) -> Optional[float]:
    """
    Extracts the capture time from EXIF data as Unix epoch seconds.

    Tries DateTimeOriginal first, then DateTimeDigitized, then the
    generic DateTime tag. Returns None if no parsable date is found.
    """
    candidates = (
        ("Exif", piexif.ExifIFD.DateTimeOriginal),
        ("Exif", piexif.ExifIFD.DateTimeDigitized),
        ("0th", piexif.ImageIFD.DateTime),
    )

    for ifd, tag in candidates:
        value = exif.get(ifd, {}).get(tag)
        if not value:
            continue
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="ignore")
        try:
            dt = datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            continue

    return None


def _sha256_of_file(path: str) -> str:
    """
    Calculates the SHA-256 hash of a file in chunks.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image_files(directory: str) -> List[str]:
    """
    Returns a list of absolute paths to supported images in directory.
    """
    supported = (".jpg", ".jpeg")
    files: List[str] = []

    if not os.path.isdir(directory):
        logging.error("‘%s’ is not a directory", directory)
        return files

    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(supported):
            files.append(entry.path)

    return files


def process_images(
    files: List[str], thumb_size: Tuple[int, int], quality: int
) -> Tuple[List[ImageMetadata], Dict[str, bytes], Dict[str, bytes]]:
    """
    Reads each image, extracts EXIF/GPS/timestamp, generates full-size and
    thumbnail JPEGs once, and returns:
      - list of ImageMetadata,
      - dict sha256→thumbnail bytes,
      - dict sha256→full-image bytes.
    """
    metadata: List[ImageMetadata] = []
    thumbnails: Dict[str, bytes] = {}
    full_images: Dict[str, bytes] = {}

    for path in files:
        try:
            with Image.open(path) as img:
                exif_bytes = img.info.get("exif")
                lat = lon = None
                timestamp = None
                if exif_bytes:
                    exif = piexif.load(exif_bytes)
                    lat, lon = _extract_gps_info(exif)
                    timestamp = _extract_timestamp(exif)

                img = ImageOps.exif_transpose(img)

                # Full image
                filehash = _sha256_of_file(path)
                buf_full = io.BytesIO()
                img.convert("RGB").save(
                    buf_full, format="JPEG", quality=quality, optimize=True
                )
                full_images[filehash] = buf_full.getvalue()

                # Thumbnail
                thumb = img.copy()
                thumb.thumbnail(thumb_size)
                w, h = thumb.size
                buf_thumb = io.BytesIO()
                thumb.save(buf_thumb, format="JPEG", quality=85)
                thumbnails[filehash] = buf_thumb.getvalue()

                # Record metadata
                metadata.append(
                    ImageMetadata(
                        sha256=filehash,
                        lat=lat,
                        lon=lon,
                        thumb_w=w,
                        thumb_h=h,
                        timestamp=timestamp,
                    )
                )

        except Exception as exc:  # noqa: BLE001
            logging.warning("Error processing '%s': %s", os.path.basename(path), exc)
            continue

    return metadata, thumbnails, full_images


def create_flask_app(
    image_objects: List[ImageMetadata],
    thumbnails: Dict[str, bytes],
    full_images: Dict[str, bytes],
    title: str,
    minutes_per_year: float,
    enable_geolocate: bool,
) -> Flask:
    """
    Creates a Flask web application to display images on a map.
    """
    app = Flask(__name__)

    # Serialize dataclass list into plain dicts
    images_json = [asdict(img) for img in image_objects]

    # Timeline range across all images that have a capture time
    timestamps = [img.timestamp for img in image_objects if img.timestamp is not None]
    min_timestamp = min(timestamps) if timestamps else None
    max_timestamp = max(timestamps) if timestamps else None
    # The timeline only makes sense with at least two distinct moments
    has_timeline = (
        min_timestamp is not None
        and max_timestamp is not None
        and max_timestamp > min_timestamp
    )

    @app.route("/")
    def index():
        return render_template_string(
            TEMPLATE,
            images_data=images_json,
            page_title=title,
            has_timeline=has_timeline,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            minutes_per_year=minutes_per_year,
            enable_geolocate=enable_geolocate,
        )

    @app.route("/image/<path:filehash>")
    def serve_image(filehash: str):
        data = full_images.get(filehash)
        if data is None:
            abort(404)
        return Response(data, mimetype="image/jpeg")

    @app.route("/thumbnail/<path:filehash>")
    def serve_thumbnail(filehash: str):
        """
        Serves a pre-generated thumbnail for the given filehash.
        """
        data = thumbnails.get(filehash)
        if data is None:
            abort(404)
        return Response(data, mimetype="image/jpeg")

    return app


def main() -> None:
    """
    Main entry point.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    files = find_image_files(args.directory)
    logging.info("Found %d images in ‘%s’.", len(files), args.directory)

    thumb_size = (args.thumb_width, args.thumb_height)
    metadata, thumbnails, full_images = process_images(files, thumb_size, args.quality)

    app = create_flask_app(
        metadata,
        thumbnails,
        full_images,
        args.title,
        args.speed,
        args.locate,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
