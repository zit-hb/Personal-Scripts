#!/usr/bin/env python3

# -------------------------------------------------------
# Script: show_joke.py
#
# Description:
# Shows a random joke using the JokeAPI.
#
# Usage:
#   ./show_joke.py [options]
#
# Options:
#   -c, --category CATEGORY     Joke category: 'programming', 'misc', 'pun', 'spooky', or 'christmas' (default: any)
#   -t, --type TYPE             Type of joke: 'single' or 'twopart' (default: any)
#   -v, --verbose               Enable verbose logging (INFO level)
#   -vv, --debug                Enable debug logging (DEBUG level)
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
import sys
from typing import Dict, Optional

import requests


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Show a random funny joke.")

    parser.add_argument(
        "-c",
        "--category",
        type=str,
        choices=["programming", "misc", "pun", "spooky", "christmas"],
        help="Joke category (default: any)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["single", "twopart"],
        help="Type of joke (default: any)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)",
    )
    parser.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level)",
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


def fetch_joke(category: Optional[str], joke_type: Optional[str]) -> Dict:
    """
    Fetches a joke from the JokeAPI.
    """
    base_url = "https://v2.jokeapi.dev/joke"

    category_param = category if category else "Programming,Misc,Pun,Spooky,Christmas"

    url = f"{base_url}/{category_param}"

    params = {
        "safe-mode": False,
    }

    if joke_type:
        params["type"] = joke_type

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch joke: {e}")
        sys.exit(1)


def display_joke(joke_data: Dict) -> None:
    """
    Displays the joke in a formatted way.
    """
    if joke_data["type"] == "single":
        print(f"{joke_data['joke']}")
    else:
        print(f"- {joke_data['setup']}")
        print(f"- {joke_data['delivery']}")


def main() -> None:
    """
    Main function to orchestrate the joke generation process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    logging.info("Fetching a joke...")

    joke = fetch_joke(args.category, args.type)

    if joke["error"]:
        logging.error("Failed to get joke from API")
        sys.exit(1)

    logging.debug(f"Received joke data: {joke}")
    display_joke(joke)


if __name__ == "__main__":
    main()
