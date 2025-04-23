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
import base64
import hashlib
import io
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from flask import Flask, Response, abort, render_template_string
from PIL import Image
import piexif


TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{{ page_title }}</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H"
    crossorigin="anonymous"
  />
  <style>
    html, body { height:100%; margin:0; }
    #map { height:100%; width:100%; }
    /* Make the thumbnail markers a little softer */
    .image-icon img { border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,.4); }
  </style>
</head>
<body>
  <div id="map"></div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH"
    crossorigin="anonymous">
  </script>
  <script>
    const map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 21,
      attribution: '© OpenStreetMap'
    }).addTo(map);

    const imagesData = {{ images_data|tojson }};
    const bounds = L.latLngBounds([]);

    imagesData.forEach(img => {
      if (img.lat !== null && img.lon !== null) {
        const size = [img.thumb_w, img.thumb_h];
        const icon = L.icon({
          iconUrl: img.thumbnail,
          iconSize: size,
          /* centre of the thumbnail = anchor, use integer values */
          iconAnchor: [Math.round(size[0] / 2), Math.round(size[1] / 2)],
          className: 'image-icon'
        });

        const marker = L.marker([img.lat, img.lon], { icon }).addTo(map);
        bounds.extend([img.lat, img.lon]);

        // Click opens full-size image from Flask route
        marker.on('click', () => window.open(`/images/${img.sha256}`, '_blank'));
      }
    });

    // Auto-fit (small padding keeps thumbnails from hugging edges)
    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(0.05));
    } else {
      map.setView([20, 0], 2);  // fallback: world view
    }
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
    thumbnail   – data-URI (JPEG) used directly by Leaflet markers
    thumb_w/h   – actual pixel dimensions of the thumbnail
    """

    sha256: str
    lat: float
    lon: float
    thumbnail: str
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


def _make_thumbnail(
    img: Image.Image, size: Tuple[int, int] = (100, 100)
) -> Tuple[str, int, int]:
    """
    Returns a base64 data-URI containing a JPEG thumbnail of the image
    plus its actual width & height.
    """
    thumb = img.copy()
    thumb.thumbnail(size)
    w, h = thumb.size
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", w, h


def _make_full_image_bytes(img: Image.Image, quality: int) -> bytes:
    """
    Returns the stripped & re-compressed full-size image as raw JPEG bytes
    (no metadata, RGB, optimised).
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


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


def get_image_metadata(files: List[str]) -> List[ImageMetadata]:
    """
    Reads each image from files and returns a list of ImageMetadata.
    """
    out: List[ImageMetadata] = []

    for path in files:
        lat, lon = None, None
        try:
            with Image.open(path) as img:
                exif_bytes = img.info.get("exif")
                if exif_bytes:
                    lat, lon = _extract_gps_info(piexif.load(exif_bytes))

                thumb_uri, tw, th = _make_thumbnail(img)

        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Thumbnail/EXIF error on ‘%s’: %s", os.path.basename(path), exc
            )
            continue  # skip corrupt images

        out.append(
            ImageMetadata(
                sha256=_sha256_of_file(path),
                lat=lat,
                lon=lon,
                thumbnail=thumb_uri,
                thumb_w=tw,
                thumb_h=th,
            )
        )
    return out


def get_full_images_dict(files: List[str], quality: int) -> Dict[str, bytes]:
    """
    Walks through *files* and builds a dictionary {sha256 → full-size JPEG bytes}.
    """
    result: Dict[str, bytes] = {}
    for path in files:
        try:
            with Image.open(path) as img:
                filehash = _sha256_of_file(path)
                result[filehash] = _make_full_image_bytes(img, quality)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Full-image error on ‘%s’: %s", os.path.basename(path), exc)
    return result


def create_flask_app(
    image_objects: List[ImageMetadata], full_images: Dict[str, bytes], title: str
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
        )

    @app.route("/images/<path:filehash>")
    def serve_image(filehash: str):
        data = full_images.get(filehash)
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

    metadata = get_image_metadata(files)
    full_images = get_full_images_dict(files, args.quality)

    app = create_flask_app(metadata, full_images, args.title)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
