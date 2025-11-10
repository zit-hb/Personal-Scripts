#!/usr/bin/env python3

# -------------------------------------------------------
# Script: download_streetview_panorama.py
#
# Description:
# Downloads and stitches Google Street View tiles into
# a single panorama image for given GPS coordinates.
#
# Usage:
#   ./download_streetview_panorama.py [options] latitude longitude
#
# Arguments:
#   - latitude: Latitude of the location.
#   - longitude: Longitude of the location.
#
# Options:
#   -z, --zoom ZOOM          Zoom level for tiles (0-5; default: 2).
#   -o, --output OUTPUT      Output image file path (default: panorama.jpg).
#   -k, --api-key KEY        Google Maps Platform API key
#                            (default: read from GOOGLE_MAPS_API_KEY).
#   -v, --verbose            Enable verbose logging (INFO level).
#   -vv, --debug             Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - requests (install via: pip install requests==2.32.5)
#   - pillow (install via: pip install pillow==10.0.0)
#   - piexif (install via: pip install piexif==1.1.3)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from io import BytesIO
from fractions import Fraction
import piexif


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download and stitch Street View panorama tiles."
    )
    parser.add_argument(
        "latitude",
        type=float,
        help="Latitude of the location.",
    )
    parser.add_argument(
        "longitude",
        type=float,
        help="Longitude of the location.",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=int,
        default=2,
        help="Zoom level for tiles (0-5; default: 2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="panorama.jpg",
        help="Output image file path (default: panorama.jpg).",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        default=None,
        help=(
            "Google Maps Platform API key. "
            "If omitted, GOOGLE_MAPS_API_KEY environment variable is used."
        ),
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


def get_api_key(explicit_key: Optional[str]) -> Optional[str]:
    """
    Returns the API key from argument or environment.
    """
    if explicit_key:
        return explicit_key
    return os.environ.get("GOOGLE_MAPS_API_KEY")


def validate_api_key(api_key: Optional[str]) -> str:
    """
    Validates that an API key is available, exits on failure.
    """
    if not api_key:
        logging.error("No API key provided. Use --api-key or set GOOGLE_MAPS_API_KEY.")
        sys.exit(1)
    return api_key


def validate_zoom(zoom: int) -> int:
    """
    Validates the zoom level, exits on failure.
    """
    if not (0 <= zoom <= 5):
        logging.error("Zoom level must be between 0 and 5 (inclusive).")
        sys.exit(1)
    return zoom


def build_create_session_payload() -> Dict[str, Any]:
    """
    Constructs the payload for the Map Tiles API createSession call.
    """
    return {
        "mapType": "streetview",
        "language": "en-US",
        "region": "US",
    }


def perform_post_json(
    url: str,
    payload: Dict[str, Any],
    description: str,
) -> Optional[Dict[str, Any]]:
    """
    Sends a JSON POST request and returns the parsed JSON response on success.
    """
    try:
        logging.info(f"{description}...")
        response = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if response.status_code != 200:
            logging.error(
                f"{description} failed with status {response.status_code}: "
                f"{response.text}"
            )
            return None

        try:
            data = response.json()
        except ValueError:
            logging.error(f"{description} returned invalid JSON.")
            return None

        logging.debug(f"{description} response: {data}")
        return data
    except Exception as error:
        logging.error(f"Error during {description}: {error}")
        return None


def create_session(api_key: str) -> Optional[str]:
    """
    Creates a Map Tiles API session for Street View and returns the session token.
    """
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    payload = build_create_session_payload()
    data = perform_post_json(url, payload, "Creating Street View session")
    if not data:
        return None

    session_token = data.get("session")
    if not session_token:
        logging.error("Session token not found in createSession response.")
        return None

    return session_token


def build_metadata_url(
    session_token: str,
    api_key: str,
    latitude: float,
    longitude: float,
    radius: int,
) -> str:
    """
    Builds the URL for the Street View metadata endpoint.
    """
    return (
        "https://tile.googleapis.com/v1/streetview/metadata"
        f"?session={session_token}"
        f"&key={api_key}"
        f"&lat={latitude}"
        f"&lng={longitude}"
        f"&radius={radius}"
    )


def fetch_json(
    url: str,
    description: str,
) -> Optional[Dict[str, Any]]:
    """
    Performs a GET request and returns parsed JSON on success.
    """
    try:
        logging.info(f"{description}...")
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            logging.error(
                f"{description} failed with status {response.status_code}: "
                f"{response.text}"
            )
            return None

        try:
            data = response.json()
        except ValueError:
            logging.error(f"{description} returned invalid JSON.")
            return None

        logging.debug(f"{description} response: {data}")
        return data
    except Exception as error:
        logging.error(f"Error during {description}: {error}")
        return None


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Ensures required panorama metadata fields are present.
    """
    pano_id = metadata.get("panoId")
    if not pano_id:
        logging.error("No panorama found for the given location.")
        return False

    required_keys = ("imageWidth", "imageHeight", "tileWidth", "tileHeight")
    for key in required_keys:
        if key not in metadata:
            logging.error(f"Missing '{key}' in metadata response.")
            return False

    return True


def get_panorama_metadata(
    session_token: str,
    api_key: str,
    latitude: float,
    longitude: float,
    radius: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves panorama metadata for the closest Street View panorama.
    """
    url = build_metadata_url(session_token, api_key, latitude, longitude, radius)
    metadata = fetch_json(url, "Fetching panorama metadata")
    if not metadata:
        return None
    if not validate_metadata(metadata):
        return None
    return metadata


def extract_metadata_dimensions(metadata: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    Extracts image and tile dimensions from panorama metadata.
    """
    image_width = int(metadata["imageWidth"])
    image_height = int(metadata["imageHeight"])
    tile_width = int(metadata["tileWidth"])
    tile_height = int(metadata["tileHeight"])
    return image_width, image_height, tile_width, tile_height


def calculate_tile_grid(
    zoom: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[int, int]:
    """
    Calculates the tile grid (columns, rows) for a given zoom level.

    Uses the Map Tiles API zoom semantics:
      - zoom in [0,5]
      - At zoom=5: full-resolution panorama split into tiles.
      - At lower zoom levels, fewer tiles cover the same panorama.

    Implementation:
      1. Compute tile counts at max zoom (5) from image and tile sizes.
      2. Scale down by powers of two for lower zoom levels.
    """
    if zoom < 0 or zoom > 5:
        raise ValueError("Zoom level must be between 0 and 5 (inclusive).")

    max_cols = (image_width + tile_width - 1) // tile_width
    max_rows = (image_height + tile_height - 1) // tile_height

    if zoom == 5:
        return max_cols, max_rows

    factor = 2 ** (5 - zoom)
    cols = max(1, (max_cols + factor - 1) // factor)
    rows = max(1, (max_rows + factor - 1) // factor)
    return cols, rows


def build_tile_url(
    session_token: str,
    api_key: str,
    pano_id: str,
    zoom: int,
    x: int,
    y: int,
) -> str:
    """
    Builds the URL for downloading a specific Street View tile.
    """
    return (
        "https://tile.googleapis.com/v1/streetview/tiles"
        f"/{zoom}/{x}/{y}"
        f"?session={session_token}"
        f"&key={api_key}"
        f"&panoId={pano_id}"
    )


def download_tile(
    session_token: str,
    api_key: str,
    pano_id: str,
    zoom: int,
    x: int,
    y: int,
) -> Optional[Image.Image]:
    """
    Downloads a single Street View tile via the Map Tiles API.
    """
    url = build_tile_url(session_token, api_key, pano_id, zoom, x, y)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
        logging.warning(
            f"Tile {zoom}/{x}/{y} request failed with status {response.status_code}"
        )
    except Exception as error:
        logging.warning(f"Failed to download tile {zoom}/{x}/{y}: {error}")

    return None


def create_placeholder_tile(tile_width: int, tile_height: int) -> Image.Image:
    """
    Returns a black placeholder tile image.
    """
    return Image.new("RGB", (tile_width, tile_height), (0, 0, 0))


def download_row_tiles(
    session_token: str,
    api_key: str,
    pano_id: str,
    zoom: int,
    y: int,
    cols: int,
    tile_width: int,
    tile_height: int,
) -> List[Image.Image]:
    """
    Downloads all tiles for one row of the panorama.
    """
    row_tiles: List[Image.Image] = []
    for x in range(cols):
        logging.debug(f"Downloading tile {zoom}/{x}/{y}")
        tile = download_tile(session_token, api_key, pano_id, zoom, x, y)
        if tile is None:
            tile = create_placeholder_tile(tile_width, tile_height)
        row_tiles.append(tile)
    return row_tiles


def download_all_tiles(
    session_token: str,
    api_key: str,
    pano_id: str,
    zoom: int,
    cols: int,
    rows: int,
    tile_width: int,
    tile_height: int,
) -> List[List[Image.Image]]:
    """
    Downloads all tiles required for the panorama.
    """
    tiles: List[List[Image.Image]] = []
    for y in range(rows):
        row_tiles = download_row_tiles(
            session_token,
            api_key,
            pano_id,
            zoom,
            y,
            cols,
            tile_width,
            tile_height,
        )
        tiles.append(row_tiles)
    return tiles


def stitch_tiles(
    tiles: List[List[Image.Image]],
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    """
    Stitches a 2D list of tiles into a single panorama image.
    """
    rows = len(tiles)
    columns = len(tiles[0]) if rows > 0 else 0

    panorama = Image.new("RGB", (columns * tile_width, rows * tile_height))

    for row_idx, row in enumerate(tiles):
        for col_idx, tile in enumerate(row):
            panorama.paste(tile, (col_idx * tile_width, row_idx * tile_height))

    return panorama


def _decimal_to_dms_rational(
    value: float,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Converts a decimal degree value into EXIF-compatible DMS rationals.
    """
    abs_val = abs(value)
    deg = int(abs_val)
    minutes_full = (abs_val - deg) * 60
    minute = int(minutes_full)
    seconds = (minutes_full - minute) * 60
    # Use a reasonable precision for seconds
    frac = Fraction(seconds).limit_denominator(1_000_000)
    return (deg, 1), (minute, 1), (frac.numerator, frac.denominator)


def _build_gps_exif_bytes(latitude: float, longitude: float) -> bytes:
    """
    Builds EXIF bytes containing GPS coordinates.
    """
    lat_ref = "N" if latitude >= 0 else "S"
    lon_ref = "E" if longitude >= 0 else "W"
    lat_dms = _decimal_to_dms_rational(latitude)
    lon_dms = _decimal_to_dms_rational(longitude)

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        piexif.GPSIFD.GPSLongitude: lon_dms,
    }

    exif_dict = {"0th": {}, "Exif": {}, "GPS": gps_ifd, "1st": {}, "thumbnail": None}
    return piexif.dump(exif_dict)


def save_panorama(
    panorama: Image.Image, output_path: str, latitude: float, longitude: float
) -> None:
    """
    Saves the panorama image to the specified path. When saving to JPEG,
    embeds GPS coordinates into proper EXIF GPS tags.
    """
    try:
        lower = output_path.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            exif_bytes = _build_gps_exif_bytes(latitude, longitude)
            panorama.save(output_path, exif=exif_bytes)
        else:
            logging.warning(
                "Output file is not JPEG; EXIF GPS metadata will not be embedded."
            )
            panorama.save(output_path)
        logging.info(f"Panorama saved to '{output_path}'")
    except Exception as error:
        logging.error(f"Failed to save image: {error}")
        sys.exit(1)


def log_attribution(metadata: Dict[str, Any]) -> None:
    """
    Logs copyright/attribution information from metadata.
    """
    copyright_text = metadata.get("copyright")
    if copyright_text:
        logging.info(f"Attribution: {copyright_text}")


def main() -> None:
    """
    Main function to orchestrate the panorama download and stitching.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    api_key = validate_api_key(get_api_key(args.api_key))
    zoom = validate_zoom(args.zoom)

    session_token = create_session(api_key)
    if session_token is None:
        sys.exit(1)

    metadata = get_panorama_metadata(
        session_token,
        api_key,
        args.latitude,
        args.longitude,
    )
    if metadata is None:
        sys.exit(1)

    pano_id = metadata["panoId"]
    image_width, image_height, tile_width, tile_height = extract_metadata_dimensions(
        metadata
    )

    logging.info(f"Using panoId: {pano_id}")
    logging.debug(
        f"Panorama size: {image_width}x{image_height}, "
        f"tile size: {tile_width}x{tile_height}"
    )

    try:
        cols, rows = calculate_tile_grid(
            zoom,
            image_width,
            image_height,
            tile_width,
            tile_height,
        )
    except ValueError as error:
        logging.error(str(error))
        sys.exit(1)

    logging.info(f"Tile grid size at zoom {zoom}: {cols} x {rows}")

    tiles = download_all_tiles(
        session_token,
        api_key,
        pano_id,
        zoom,
        cols,
        rows,
        tile_width,
        tile_height,
    )

    logging.info("Stitching tiles into panorama...")
    panorama = stitch_tiles(tiles, tile_width, tile_height)

    save_panorama(panorama, args.output, args.latitude, args.longitude)
    log_attribution(metadata)


if __name__ == "__main__":
    main()
