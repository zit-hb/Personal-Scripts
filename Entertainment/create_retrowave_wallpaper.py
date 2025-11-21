#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_retrowave_wallpaper.py
#
# Description:
# Generates a retrowave aesthetic wallpaper with perspective grid
# matching specified dimensions.
#
# Usage:
#   ./create_retrowave_wallpaper.py [options]
#
# Options:
#   -W, --width WIDTH       Width of the wallpaper (default: 1920).
#   -H, --height HEIGHT     Height of the wallpaper (default: 1080).
#   -o, --output OUTPUT     Output filename (default: wallpaper.png).
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - Pillow (install via: pip install Pillow==11.1.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import sys
from typing import Tuple

from PIL import Image, ImageDraw


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a retrowave aesthetic wallpaper with perspective grid."
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=1920,
        help="Width of the wallpaper (default: 1920).",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=1080,
        help="Height of the wallpaper (default: 1080).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="wallpaper.png",
        help="Output filename (default: wallpaper.png).",
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


def draw_sky_gradient(
    draw: ImageDraw.Draw, width: int, height: int, horizon_y: int, sky_colors: list
) -> None:
    """
    Draws the sky gradient from top to horizon.
    """
    for y in range(horizon_y):
        ratio = y / max(1, horizon_y - 1)
        idx = ratio * (len(sky_colors) - 1)
        low = int(idx)
        high = min(low + 1, len(sky_colors) - 1)
        t = idx - low
        c1, c2 = sky_colors[low], sky_colors[high]
        color = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)


def draw_ground_gradient(
    draw: ImageDraw.Draw, width: int, height: int, horizon_y: int, ground_colors: list
) -> None:
    """
    Draws the ground gradient from horizon to bottom.
    """
    for y in range(horizon_y, height):
        ratio = (y - horizon_y) / max(1, height - horizon_y - 1)
        idx = ratio * (len(ground_colors) - 1)
        low = int(idx)
        high = min(low + 1, len(ground_colors) - 1)
        t = idx - low
        c1, c2 = ground_colors[low], ground_colors[high]
        color = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)


def draw_horizontal_grid_lines(
    grid_draw: ImageDraw.Draw,
    sw: int,
    vp_y: float,
    bottom_y: float,
    scale: int,
    grid_color_bright: Tuple[int, int, int],
    grid_color_dim: Tuple[int, int, int],
) -> None:
    """
    Draws horizontal perspective grid lines with proper perspective compression.
    """
    num_horiz = 60  # Increased from 40 for denser grid
    for i in range(num_horiz):
        # Use stronger exponential curve for more perspective compression
        # Higher power = more compression near horizon
        t = (i / num_horiz) ** 2.2  # Increased from 1.8 for more dramatic perspective
        y = vp_y + t * (bottom_y - vp_y)

        # Calculate distance factor (0 at horizon, 1 at bottom)
        factor = i / num_horiz  # 0 at horizon (far), 1 at bottom (close)

        # Lines get thicker as they get closer
        width_px = max(1, int(2 * scale * (0.3 + 0.7 * factor)))

        # Use lower exponent: stays bright longer, then darkens quickly near horizon
        brightness_factor = factor**1.5  # Lower exponent = slower then faster falloff

        # Darker at horizon but not too dark
        brightness = 0.1 + 0.9 * brightness_factor
        opacity = int(255 * brightness)

        # Apply brightness to color
        color = tuple(int(grid_color_bright[j] * brightness) for j in range(3)) + (
            opacity,
        )
        grid_draw.line([(0, y), (sw, y)], fill=color, width=width_px)


def draw_vertical_grid_lines(
    grid_draw: ImageDraw.Draw,
    sw: int,
    vp_x: float,
    vp_y: float,
    bottom_y: float,
    scale: int,
    grid_color_bright: Tuple[int, int, int],
) -> int:
    """
    Draws vertical perspective grid lines converging to top center.
    Lines are drawn from bottom to horizon with gradient based on distance.
    Returns the number of lines actually drawn.
    """
    # Vanishing point is at the very top center of the image
    vanishing_point_x = sw / 2
    vanishing_point_y = 0  # Top of image

    # Grid goes from bottom to horizon
    grid_bottom_y = bottom_y
    grid_top_y = vp_y  # Horizon line

    # Number of vertical lines - matching the original aesthetic
    num_lines = 40
    line_spacing_bottom = sw / num_lines

    # Calculate how far we need to extend beyond the edges
    # Left edge calculation
    dx_left = vanishing_point_x - 0
    dy_left = vanishing_point_y - grid_top_y
    t_left = (grid_bottom_y - grid_top_y) / dy_left if dy_left != 0 else 0
    left_x_at_bottom = 0 + t_left * dx_left

    # Right edge calculation
    dx_right = vanishing_point_x - sw
    dy_right = vanishing_point_y - grid_top_y
    t_right = (grid_bottom_y - grid_top_y) / dy_right if dy_right != 0 else 0
    right_x_at_bottom = sw + t_right * dx_right

    # Now we know the range at the bottom
    x_range = right_x_at_bottom - left_x_at_bottom
    num_lines_needed = int(x_range / line_spacing_bottom) + 1

    logging.info(
        f"Drawing {num_lines_needed} vertical lines to fill plane from {left_x_at_bottom:.0f} to {right_x_at_bottom:.0f}"
    )

    lines_drawn = 0

    # Draw lines across the entire range
    for i in range(num_lines_needed + 1):
        # Position at bottom of image
        x_bottom = left_x_at_bottom + i * line_spacing_bottom

        # Calculate where this line would intersect at the horizon (grid_top_y)
        dx = vanishing_point_x - x_bottom
        dy = vanishing_point_y - grid_bottom_y

        # Find where this line intersects y = grid_top_y (horizon)
        t = (grid_top_y - grid_bottom_y) / dy if dy != 0 else 0
        x_top = x_bottom + t * dx

        # Draw the line in segments with brightness AND width based on distance
        # NO ALPHA - use opaque colors like horizontal lines
        num_segments = 50
        for seg in range(num_segments):
            y_start = grid_bottom_y + (grid_top_y - grid_bottom_y) * (
                seg / num_segments
            )
            y_end = grid_bottom_y + (grid_top_y - grid_bottom_y) * (
                (seg + 1) / num_segments
            )

            # Calculate x positions for this segment
            t_start = (y_start - grid_bottom_y) / (grid_top_y - grid_bottom_y)
            t_end = (y_end - grid_bottom_y) / (grid_top_y - grid_bottom_y)

            x_start = x_bottom + t_start * (x_top - x_bottom)
            x_end = x_bottom + t_end * (x_top - x_bottom)

            # Calculate distance from viewer (0 at bottom, 1 at horizon)
            distance = t_start

            # Use EXACT same formula as horizontal lines
            factor = 1.0 - distance  # Convert to 0 at horizon, 1 at bottom
            brightness_factor = (
                factor**1.5
            )  # Stays bright longer, darkens quickly near horizon
            brightness = 0.1 + 0.9 * brightness_factor

            # Match horizontal line width calculation exactly!
            line_width = max(1, int(2 * scale * (0.3 + 0.7 * factor)))

            # Apply brightness to color - NO ALPHA, just RGB like horizontal lines!
            color = tuple(int(grid_color_bright[j] * brightness) for j in range(3))

            grid_draw.line(
                [(x_start, y_start), (x_end, y_end)],
                fill=color,  # Solid color, no alpha!
                width=line_width,
            )

        lines_drawn += 1

    logging.info(f"Vertical lines drawn: {lines_drawn}")

    return lines_drawn


def add_horizon_glow(img: Image.Image, width: int, height: int, horizon_y: int) -> None:
    """
    Adds a subtle glow effect at the horizon.
    """
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_height = int(height * 0.15)
    for i in range(glow_height):
        alpha = int(30 * (1 - i / glow_height) ** 2)
        glow_draw.line(
            [
                (0, horizon_y - glow_height // 2 + i),
                (width, horizon_y - glow_height // 2 + i),
            ],
            fill=(255, 100, 200, alpha),
        )
    img.paste(glow, (0, 0), glow)


def add_horizon_center_gradient(
    img: Image.Image, width: int, height: int, horizon_y: int
) -> None:
    """
    Adds a black-to-transparent gradient at the horizon center for clean transition.
    """
    center_gradient = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    center_draw = ImageDraw.Draw(center_gradient)
    for i in range(3):
        # First line (i=0) is completely black (255), last line (i=2) is transparent (0)
        alpha = int(255 * (1 - i / 2))
        y = horizon_y + i
        if y < height:
            center_draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    img.paste(center_gradient, (0, 0), center_gradient)


def create_retrowave_wallpaper(width: int, height: int) -> Image.Image:
    """
    Creates a retrowave aesthetic wallpaper with perspective grid.
    """
    # Base image and draw context
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Color palettes
    sky_colors = [
        (10, 10, 40),
        (40, 10, 60),
        (80, 20, 100),
        (150, 40, 120),
        (200, 60, 140),
        (255, 100, 150),
        (255, 140, 100),
    ]
    ground_colors = [
        (20, 5, 30),
        (10, 5, 20),
        (5, 0, 10),
    ]
    grid_color_bright: Tuple[int, int, int] = (255, 0, 200)
    grid_color_dim: Tuple[int, int, int] = (100, 0, 80)

    # Horizon position
    horizon_y = int(height * 0.55)

    # Draw gradients
    draw_sky_gradient(draw, width, height, horizon_y, sky_colors)
    draw_ground_gradient(draw, width, height, horizon_y, ground_colors)

    # Create antialiased grid on a larger canvas
    scale = 2
    sw, sh = width * scale, height * scale
    grid_img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid_img)
    vp_x, vp_y = sw / 2, horizon_y * scale
    bottom_y = sh

    # Calculate line width scale factor based on image dimensions
    # Use 1920x1080 as reference resolution
    reference_width = 1920
    size_scale_factor = width / reference_width
    effective_scale = scale * size_scale_factor

    # Draw grid lines
    draw_horizontal_grid_lines(
        grid_draw,
        sw,
        vp_y,
        bottom_y,
        effective_scale,
        grid_color_bright,
        grid_color_dim,
    )
    draw_vertical_grid_lines(
        grid_draw, sw, vp_x, vp_y, bottom_y, effective_scale, grid_color_bright
    )

    # Downsample and composite
    grid_img = grid_img.resize((width, height), Image.LANCZOS)
    img.paste(grid_img, (0, 0), grid_img)

    # Add effects
    add_horizon_glow(img, width, height, horizon_y)
    add_horizon_center_gradient(img, width, height, horizon_y)

    return img


def main() -> None:
    """
    Main function to orchestrate the wallpaper generation.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    logging.info(f"Generating {args.width}x{args.height} wallpaper.")
    try:
        wallpaper = create_retrowave_wallpaper(args.width, args.height)
        wallpaper.save(args.output, "JPEG", quality=95)
        logging.info(f"Wallpaper saved to: {args.output}")
    except Exception as error:
        logging.error(f"Failed to generate wallpaper: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
