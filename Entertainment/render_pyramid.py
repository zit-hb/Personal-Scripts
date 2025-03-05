#!/usr/bin/env python3

# -------------------------------------------------------
# Script: render_pyramid.py
#
# Description:
# Renders an animated spinning 3D pyramid in ASCII art.
#
# Usage:
#   ./render_pyramid.py [options]
#
# Options:
#   -W, --width WIDTH       Absolute pyramid width (default: 25).
#   -H, --height HEIGHT     Absolute pyramid height (default: 20).
#   -x, --x-speed SPEED     X rotation speed (-10 to 10, default: 3).
#   -y, --y-speed SPEED     Y rotation speed (-10 to 10, default: 1).
#   -z, --z-speed SPEED     Z rotation speed (-10 to 10, default: 1).
#   -n, --no-color          Disable colored output.
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
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
from typing import List, Tuple, Dict

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
        description="Renders an animated spinning 3D pyramid in ASCII art."
    )

    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=25,
        help="Absolute pyramid width (default: 25).",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=20,
        help="Absolute pyramid height (default: 20).",
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

    for speed_arg in ["x_speed", "y_speed", "z_speed"]:
        value = getattr(args, speed_arg)
        if not (-10 <= value <= 10):
            parser.error(f"{speed_arg} must be between -10 and 10")

    if args.width < 1:
        parser.error("Width must be at least 1")
    if args.height < 1:
        parser.error("Height must be at least 1")

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


def calculate_pyramid_parameters(
    term_width: int, term_height: int, pyramid_width: int, pyramid_height: int
) -> Dict[str, int]:
    """
    Prepares pyramid parameters, including screen center.
    """
    center_x = term_width // 2
    center_y = term_height // 2

    logging.debug(
        f"Calculated center: ({center_x}, {center_y}), "
        f"pyramid width: {pyramid_width}, pyramid height: {pyramid_height}"
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": pyramid_width,
        "height": pyramid_height,
    }


def create_3d_pyramid(
    pyramid_width: float, pyramid_height: float
) -> List[List[Tuple[float, float, float]]]:
    """
    Creates the 3D coordinates for a pyramid given absolute width and height.
    The base is a square with side = pyramid_width, apex at pyramid_height above the base.
    """
    half_w = pyramid_width / 2
    half_h = pyramid_height / 2

    # In this layout:
    # - The base is centered around y = -half_h
    # - The apex is at y = +half_h
    # - The base corners extend ± half_w in x and z
    base = [
        [-half_w, -half_h, -half_w],  # front-left
        [half_w, -half_h, -half_w],  # front-right
        [half_w, -half_h, half_w],  # back-right
        [-half_w, -half_h, half_w],  # back-left
        [0, half_h, 0],  # apex
    ]

    # Define the faces (each face is a list of vertex indices)
    faces = [
        [0, 1, 4],  # front face
        [1, 2, 4],  # right face
        [2, 3, 4],  # back face
        [3, 0, 4],  # left face
        [0, 3, 2, 1],  # base face
    ]

    # Convert to list of faces where each face is a list of 3D points
    return [[base[idx] for idx in face] for face in faces]


def rotate_point(
    point: List[float], angle_x: float, angle_y: float, angle_z: float
) -> List[float]:
    """
    Rotates a 3D point around all three axes.
    """
    x, y, z = point

    # Rotate around X-axis
    y_rot = y * math.cos(angle_x) - z * math.sin(angle_x)
    z_rot = y * math.sin(angle_x) + z * math.cos(angle_x)
    y, z = y_rot, z_rot

    # Rotate around Y-axis
    x_rot = x * math.cos(angle_y) + z * math.sin(angle_y)
    z_rot = -x * math.sin(angle_y) + z * math.cos(angle_y)
    x, z = x_rot, z_rot

    # Rotate around Z-axis
    x_rot = x * math.cos(angle_z) - y * math.sin(angle_z)
    y_rot = x * math.sin(angle_z) + y * math.cos(angle_z)
    x, y = x_rot, y_rot

    return [x, y, z]


def project_point(
    point: List[float], center_x: int, center_y: int, scale: float = 1.0
) -> Tuple[int, int]:
    """
    Projects a 3D point onto the 2D screen with a perspective effect.
    """
    x, y, z = point

    # Simple perspective projection
    distance = 100
    factor = distance / (distance + z + 50)  # Adding offset to avoid division by zero

    screen_x = int(center_x + x * factor * scale)
    screen_y = int(
        center_y - y * factor * scale * 0.5
    )  # Terminal characters are taller than wide

    return screen_x, screen_y


def render_pyramid(
    stdscr,
    pyramid: List[List[List[float]]],
    angle_x: float,
    angle_y: float,
    angle_z: float,
    params: Dict[str, int],
    use_color: bool,
) -> None:
    """
    Renders the pyramid with the given rotation angles.
    """
    center_x = params["center_x"]
    center_y = params["center_y"]
    pyramid_w = params["width"]
    pyramid_h = params["height"]

    # We'll use the average of width/height for scaling in projection
    dimension = (pyramid_w + pyramid_h) / 2.0

    # Clear the screen
    stdscr.clear()

    # Rotate the pyramid
    rotated_pyramid = []
    for face in pyramid:
        rotated_face = [
            rotate_point(point, angle_x, angle_y, angle_z) for point in face
        ]
        rotated_pyramid.append(rotated_face)

    # Calculate the average Z value for each face to determine visibility
    face_z_values = []
    for face in rotated_pyramid:
        avg_z = sum(p[2] for p in face) / len(face)
        face_z_values.append(avg_z)

    # Sort faces by average Z value for proper depth rendering (painter's algorithm)
    # We'll render from back to front
    sorted_faces = sorted(zip(rotated_pyramid, face_z_values), key=lambda f: f[1])

    # Shading characters from darkest to lightest
    shading_chars = " .:-=+*#%@"

    # Define colors if enabled
    colors = (
        ["red", "green", "yellow", "blue", "magenta"] if use_color and HAS_COLOR else []
    )

    for idx, (face, z_val) in enumerate(sorted_faces):
        # Project the face points to 2D
        projected_points = [
            project_point(p, center_x, center_y, dimension / 10.0) for p in face
        ]

        # Calculate a light intensity based on orientation
        # Simple lighting calculation, higher z values are brighter
        # 'dimension' is used as a reference scale so that the shading doesn't blow up
        light_intensity = (z_val + dimension) / (2 * dimension)
        light_intensity = max(0.1, min(0.9, light_intensity))

        char_idx = int(light_intensity * (len(shading_chars) - 1))
        char = shading_chars[char_idx]

        # Connect the face points to draw lines
        for i in range(len(projected_points)):
            start = projected_points[i]
            end = projected_points[(i + 1) % len(projected_points)]

            face_color = None
            if colors:
                face_color = colors[idx % len(colors)]

            draw_line(
                stdscr,
                start[0],
                start[1],
                end[0],
                end[1],
                char,
                use_color,
                face_color,
            )

    # Update the screen
    stdscr.refresh()


def draw_line(
    stdscr,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    char: str,
    use_color: bool,
    color: str = None,
) -> None:
    """
    Draws a line using Bresenham's algorithm.
    """
    height, width = stdscr.getmaxyx()

    # Ensure coordinates are integers
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    # Define color pairs for curses if color is enabled
    if use_color and HAS_COLOR and color:
        color_map = {
            "red": 1,
            "green": 2,
            "yellow": 3,
            "blue": 4,
            "magenta": 5,
            "cyan": 6,
            "white": 7,
        }
        color_pair = color_map.get(color, 7)  # Default to white if color not found

    while True:
        # Check if point is inside screen bounds
        if 0 <= y0 < height - 1 and 0 <= x0 < width - 1:
            try:
                if use_color and HAS_COLOR and color:
                    # Use curses color pairs
                    stdscr.addstr(y0, x0, char, curses.color_pair(color_pair))
                else:
                    stdscr.addstr(y0, x0, char)
            except curses.error:
                pass  # Ignore errors from writing to the bottom-right corner

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def animate_pyramid(
    stdscr,
    use_color: bool = True,
    x_speed: float = 0.0,
    y_speed: float = 3.0,
    z_speed: float = 0.0,
    pyramid_width: int = 40,
    pyramid_height: int = 25,
) -> None:
    """
    Animates the rotating pyramid.
    """
    # Initialize terminal
    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(0)  # Non-blocking getch()

    # Initialize colors if using color
    if use_color and HAS_COLOR:
        curses.start_color()
        curses.use_default_colors()
        # Initialize color pairs
        color_pairs = [
            (curses.COLOR_RED, 1),
            (curses.COLOR_GREEN, 2),
            (curses.COLOR_YELLOW, 3),
            (curses.COLOR_BLUE, 4),
            (curses.COLOR_MAGENTA, 5),
            (curses.COLOR_CYAN, 6),
            (curses.COLOR_WHITE, 7),
        ]
        for color, pair_num in color_pairs:
            curses.init_pair(pair_num, color, -1)  # -1 means default background

    # Get screen dimensions
    term_height, term_width = stdscr.getmaxyx()

    # Calculate parameters
    params = calculate_pyramid_parameters(
        term_width, term_height, pyramid_width, pyramid_height
    )

    # Create the 3D pyramid
    pyramid = create_3d_pyramid(params["width"], params["height"])

    # Scale rotation speeds to keep them reasonable
    x_rotation_speed = (x_speed / 10) * 0.1
    y_rotation_speed = (y_speed / 10) * 0.1
    z_rotation_speed = (z_speed / 10) * 0.1

    angle_x, angle_y, angle_z = 0, 0, 0

    try:
        while True:
            # Check for key press to exit
            key = stdscr.getch()
            if key != -1:
                break

            # Update rotation angles
            angle_x += x_rotation_speed
            angle_y += y_rotation_speed
            angle_z += z_rotation_speed

            # Render the pyramid
            render_pyramid(
                stdscr, pyramid, angle_x, angle_y, angle_z, params, use_color
            )

            # Short delay for animation
            time.sleep(0.05)

            # Check if terminal size changed
            new_height, new_width = stdscr.getmaxyx()
            if new_height != term_height or new_width != term_width:
                term_height, term_width = new_height, new_width
                params = calculate_pyramid_parameters(
                    term_width, term_height, pyramid_width, pyramid_height
                )
                pyramid = create_3d_pyramid(params["width"], params["height"])

    except KeyboardInterrupt:
        pass


def main() -> None:
    """
    Main function to orchestrate the pyramid rendering process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    # Color is enabled by default unless explicitly disabled
    use_color = HAS_COLOR and not args.no_color

    if not HAS_COLOR and not args.no_color:
        logging.warning("Color support requires 'colorama' and 'termcolor' packages.")
        logging.warning("Install with: pip install colorama termcolor")
    elif args.no_color:
        logging.info("Color output disabled by user")
    else:
        logging.info("Color output enabled")

    if HAS_COLOR:
        colorama.init()

    logging.info("Starting pyramid animation")
    logging.info(f"X rotation speed: {args.x_speed}")
    logging.info(f"Y rotation speed: {args.y_speed}")
    logging.info(f"Z rotation speed: {args.z_speed}")
    logging.info(f"Pyramid width: {args.width}, Pyramid height: {args.height}")

    try:
        # Use curses for terminal manipulation
        curses.wrapper(
            lambda stdscr: animate_pyramid(
                stdscr,
                use_color,
                args.x_speed,
                args.y_speed,
                args.z_speed,
                args.width,
                args.height,
            )
        )
    except Exception as e:
        logging.error(f"Error in animation: {e}")
        return

    if HAS_COLOR:
        colorama.deinit()

    logging.info("Animation complete")


if __name__ == "__main__":
    main()
