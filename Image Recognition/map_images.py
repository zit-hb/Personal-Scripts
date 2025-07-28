#!/usr/bin/env python3

# -------------------------------------------------------
# Script: map_images.py
#
# Description:
# Starts a small Flask web application that shows a Leaflet
# map with every image in the chosen directory rendered as a
# miniature thumbnail at its GPS position.
#
# Usage:
#   ./map_images.py [options] directory
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
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import hashlib
import io
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

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
  </style>
</head>
<body>
  <div id="map"></div>

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

    imagesData.forEach(img => {
      if (img.lat !== null && img.lon !== null) {
        const marker = L.marker([img.lat, img.lon]);
        cluster.addLayer(marker);
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
      }
    });

    // Auto-fit (small padding keeps markers from hugging edges)
    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(0.05));
    } else {
      map.setView([20, 0], 2);  // fallback: world view
    }

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
    """

    sha256: str
    lat: float
    lon: float
    thumb_w: int
    thumb_h: int


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
    Reads each image, extracts EXIF/GPS, generates full-size and thumbnail
    JPEGs once, and returns:
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
                if exif_bytes:
                    lat, lon = _extract_gps_info(piexif.load(exif_bytes))

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
    enable_geolocate: bool,
) -> Flask:
    """
    Creates a Flask web application to display images on a map.
    """
    app = Flask(__name__)

    # Serialize dataclass list into plain dicts
    images_json = [asdict(img) for img in image_objects]

    @app.route("/")
    def index():
        return render_template_string(
            TEMPLATE,
            images_data=images_json,
            page_title=title,
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
        args.locate,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
