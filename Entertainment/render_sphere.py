#!/usr/bin/env python3

# -------------------------------------------------------
# Script: render_sphere.py
#
# Description:
# Renders an animated spinning 3D sphere in ASCII art.
#
# Usage:
#   ./render_sphere.py [options]
#
# Options:
#   -R, --radius RADIUS           Sphere radius (default: 15).
#   -A, --lat-segments SEGMENTS   Number of horizontal segments (default: 16).
#   -O, --lon-segments SEGMENTS   Number of vertical segments (default: 16).
#   -x, --x-speed SPEED           X rotation speed (-10 to 10, default: 3).
#   -y, --y-speed SPEED           Y rotation speed (-10 to 10, default: 1).
#   -z, --z-speed SPEED           Z rotation speed (-10 to 10, default: 1).
#   -n, --no-color                Disable colored output.
#   -v, --verbose                 Enable verbose logging (INFO level).
#   -vv, --debug                  Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - colorama (install via: pip install colorama==0.4.6)
#   - termcolor (install via: pip install termcolor==2.5.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import curses
import logging
import math
import time
from typing import Dict, List, Tuple

try:
    import colorama
    from termcolor import colored

    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Renders an animated spinning ASCII sphere in the terminal."
    )
    parser.add_argument(
        "-R",
        "--radius",
        type=int,
        default=15,
        help="Sphere radius (default: 15).",
    )
    parser.add_argument(
        "-A",
        "--lat-segments",
        type=int,
        default=16,
        help="Number of horizontal segments (default: 16).",
    )
    parser.add_argument(
        "-O",
        "--lon-segments",
        type=int,
        default=16,
        help="Number of vertical segments (default: 16).",
    )
    parser.add_argument(
        "-x",
        "--x-speed",
        type=float,
        default=3.0,
        help="X rotation speed (-10 to 10, default: 3).",
    )
    parser.add_argument(
        "-y",
        "--y-speed",
        type=float,
        default=1.0,
        help="Y rotation speed (-10 to 10, default: 1).",
    )
    parser.add_argument(
        "-z",
        "--z-speed",
        type=float,
        default=1.0,
        help="Z rotation speed (-10 to 10, default: 1).",
    )
    parser.add_argument(
        "-n",
        "--no-color",
        action="store_true",
        help="Disable colored output.",
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

    args = parser.parse_args()

    if args.radius < 1:
        parser.error("Radius must be at least 1")

    for speed_arg in ["x_speed", "y_speed", "z_speed"]:
        value = getattr(args, speed_arg)
        if not (-10 <= value <= 10):
            parser.error(f"{speed_arg} must be between -10 and 10")

    if args.lat_segments < 1:
        parser.error("lat-segments must be at least 1")

    if args.lon_segments < 1:
        parser.error("lon-segments must be at least 1")

    return args


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


def create_3d_sphere(
    radius: float, lat_segments: int, lon_segments: int
) -> List[List[Tuple[float, float, float]]]:
    """
    Creates a wireframe mesh (list of faces) for a sphere with the given radius.
    Each 'face' is a list of 3D vertex pairs or sets of points to draw lines.
    We'll group lines for easier drawing: each face is a lat-lon trapezoid.
    """
    vertices = []
    # Generate vertex grid in latitude and longitude
    for lat_i in range(lat_segments + 1):
        lat = math.pi * (float(lat_i) / lat_segments - 0.5)
        ring = []
        for lon_i in range(lon_segments + 1):
            lon = 2.0 * math.pi * float(lon_i) / lon_segments
            x = radius * math.cos(lat) * math.cos(lon)
            y = radius * math.sin(lat)
            z = radius * math.cos(lat) * math.sin(lon)
            ring.append((x, y, z))
        vertices.append(ring)

    # Build wireframe faces as lat-lon quads (split into lines)
    faces = []
    for lat_i in range(lat_segments):
        for lon_i in range(lon_segments):
            # Corner points of this lat-lon cell
            p1 = vertices[lat_i][lon_i]
            p2 = vertices[lat_i][lon_i + 1]
            p3 = vertices[lat_i + 1][lon_i + 1]
            p4 = vertices[lat_i + 1][lon_i]
            # We'll store each face as a list of edges
            faces.append([p1, p2, p3, p4])
    return faces


def rotate_point(
    point: Tuple[float, float, float], ax: float, ay: float, az: float
) -> Tuple[float, float, float]:
    """
    Rotates a 3D point around X, Y, Z axes by ax, ay, az radians.
    """
    x, y, z = point

    # Rotate around X-axis
    y_rot = y * math.cos(ax) - z * math.sin(ax)
    z_rot = y * math.sin(ax) + z * math.cos(ax)
    y, z = y_rot, z_rot

    # Rotate around Y-axis
    x_rot = x * math.cos(ay) + z * math.sin(ay)
    z_rot = -x * math.sin(ay) + z * math.cos(ay)
    x, z = x_rot, z_rot

    # Rotate around Z-axis
    x_rot = x * math.cos(az) - y * math.sin(az)
    y_rot = x * math.sin(az) + y * math.cos(az)
    x, y = x_rot, y_rot

    return (x, y, z)


def project_point(
    point: Tuple[float, float, float], cx: int, cy: int, scale: float = 1.0
) -> Tuple[int, int]:
    """
    Projects a 3D point onto 2D screen with a simple perspective effect.
    """
    x, y, z = point
    distance = 100
    factor = distance / (distance + z + 50)
    screen_x = int(cx + x * factor * scale)
    screen_y = int(cy - y * factor * scale * 0.5)
    return (screen_x, screen_y)


def draw_line(
    stdscr, x0: int, y0: int, x1: int, y1: int, char: str, color_pair: int = 0
) -> None:
    """
    Draws a line using Bresenham's algorithm on the curses screen.
    """
    height, width = stdscr.getmaxyx()

    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            try:
                if color_pair != 0:
                    stdscr.addstr(y0, x0, char, curses.color_pair(color_pair))
                else:
                    stdscr.addstr(y0, x0, char)
            except curses.error:
                pass

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def render_sphere(
    stdscr,
    faces: List[List[Tuple[float, float, float]]],
    ax: float,
    ay: float,
    az: float,
    params: Dict[str, int],
    use_color: bool,
) -> None:
    """
    Renders the sphere wireframe given rotation angles and screen geometry.
    """
    stdscr.clear()
    center_x = params["center_x"]
    center_y = params["center_y"]
    dimension = float(params["sphere_radius"])

    for i, face in enumerate(faces):
        rotated_face = [rotate_point(p, ax, ay, az) for p in face]
        # We choose a single shading character
        # Light intensity based on average Z
        avg_z = sum(p[2] for p in rotated_face) / len(rotated_face)
        shading_chars = " .:-=+*#%@"
        intensity = (avg_z + dimension) / (2 * dimension)
        intensity = max(0.0, min(1.0, intensity))
        char_idx = int(intensity * (len(shading_chars) - 1))
        char = shading_chars[char_idx]

        color_pair = 0
        if use_color and HAS_COLOR:
            # Cycle through some color pairs
            # (1=red,2=green,3=yellow,4=blue,5=magenta,6=cyan,7=white)
            color_pair = (i % 7) + 1

        # Project face edges
        proj_points = [
            project_point(p, center_x, center_y, dimension / 10.0) for p in rotated_face
        ]
        # Draw lines forming a quad: [p1->p2, p2->p3, p3->p4, p4->p1]
        for idx_pt in range(len(proj_points)):
            start = proj_points[idx_pt]
            end = proj_points[(idx_pt + 1) % len(proj_points)]
            draw_line(stdscr, start[0], start[1], end[0], end[1], char, color_pair)

    stdscr.refresh()


def animate_sphere(
    stdscr,
    radius: int,
    lat_segments: int,
    lon_segments: int,
    x_speed: float,
    y_speed: float,
    z_speed: float,
    use_color: bool,
) -> None:
    """
    Main loop to animate the rotating ASCII sphere.
    """
    curses.curs_set(0)
    stdscr.timeout(0)

    if use_color and HAS_COLOR:
        curses.start_color()
        curses.use_default_colors()
        color_map = [
            (curses.COLOR_RED, 1),
            (curses.COLOR_GREEN, 2),
            (curses.COLOR_YELLOW, 3),
            (curses.COLOR_BLUE, 4),
            (curses.COLOR_MAGENTA, 5),
            (curses.COLOR_CYAN, 6),
            (curses.COLOR_WHITE, 7),
        ]
        for color, pair_num in color_map:
            curses.init_pair(pair_num, color, -1)

    term_height, term_width = stdscr.getmaxyx()
    params = {
        "center_x": term_width // 2,
        "center_y": term_height // 2,
        "sphere_radius": radius,
    }

    faces = create_3d_sphere(radius, lat_segments, lon_segments)

    ax = 0.0
    ay = 0.0
    az = 0.0

    # Scale speeds
    ax_speed = (x_speed / 10.0) * 0.1
    ay_speed = (y_speed / 10.0) * 0.1
    az_speed = (z_speed / 10.0) * 0.1

    try:
        while True:
            key = stdscr.getch()
            if key != -1:
                break

            ax += ax_speed
            ay += ay_speed
            az += az_speed

            new_height, new_width = stdscr.getmaxyx()
            if new_height != term_height or new_width != term_width:
                term_height, term_width = new_height, new_width
                params["center_x"] = term_width // 2
                params["center_y"] = term_height // 2

            render_sphere(stdscr, faces, ax, ay, az, params, use_color)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def main() -> None:
    """
    Main function to orchestrate the sphere rendering process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if not HAS_COLOR and not args.no_color:
        logging.warning("Color requires 'colorama' and 'termcolor' packages.")
        logging.warning("Install with: pip install colorama termcolor")
    elif args.no_color:
        logging.info("Color output disabled by user")
    else:
        logging.info("Color output enabled")

    if HAS_COLOR:
        colorama.init()

    logging.info("Starting sphere animation.")
    logging.info(f"Radius: {args.radius}")
    logging.info(f"Lat segments: {args.lat_segments}")
    logging.info(f"Lon segments: {args.lon_segments}")
    logging.info(f"Speeds (x,y,z): ({args.x_speed}, {args.y_speed}, {args.z_speed})")

    try:
        curses.wrapper(
            lambda stdscr: animate_sphere(
                stdscr,
                args.radius,
                args.lat_segments,
                args.lon_segments,
                args.x_speed,
                args.y_speed,
                args.z_speed,
                use_color=HAS_COLOR and not args.no_color,
            )
        )
    except Exception as e:
        logging.error(f"Error in animation: {e}")

    if HAS_COLOR:
        colorama.deinit()

    logging.info("Animation finished.")


if __name__ == "__main__":
    main()
