#!/usr/bin/env python3

# -------------------------------------------------------
# Script: tell_joke.py
#
# Description:
# Tells a random joke using the JokeAPI.
#
# Usage:
#   ./tell_joke.py [options]
#
# Options:
#   -c, --category CATEGORY     Joke category: 'programming', 'misc', 'pun', 'spooky', or 'christmas' (default: any).
#   -t, --type TYPE             Type of joke: 'single' or 'twopart' (default: any).
#   -b, --blacklist FLAGS       Blacklist flags (comma separated): 'nsfw', 'religious', 'political', 'racist', 'sexist', 'explicit'.
#   -l, --lang LANG             Language code: 'cs', 'de', 'en', 'es', 'fr', 'pt' (default: en).
#   -i, --id-range RANGE        ID range (e.g., '0-100', '42').
#   -s, --search TEXT           Search for jokes containing this text.
#   -a, --amount NUMBER         Number of jokes to fetch (1-10).
#   -m, --safe-mode             Enable safe mode (filters out explicit jokes).
#   -v, --verbose               Enable verbose logging (INFO level).
#   -vv, --debug                Enable debug logging (DEBUG level).
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
    parser = argparse.ArgumentParser(description="Tell a random funny joke.")

    parser.add_argument(
        "-c",
        "--category",
        type=str,
        choices=["programming", "misc", "pun", "spooky", "christmas"],
        help="Joke category (default: any).",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["single", "twopart"],
        help="Type of joke (default: any).",
    )
    parser.add_argument(
        "-b",
        "--blacklist",
        type=str,
        help="Blacklist flags (comma separated): 'nsfw', 'religious', 'political', 'racist', 'sexist', 'explicit'.",
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        default="en",
        choices=["cs", "de", "en", "es", "fr", "pt"],
        help="Language code (default: en).",
    )
    parser.add_argument(
        "-i",
        "--id-range",
        type=str,
        help="ID range (e.g., '0-100', '42').",
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        help="Search for jokes containing this text.",
    )
    parser.add_argument(
        "-a",
        "--amount",
        type=int,
        choices=range(1, 11),
        default=1,
        help="Number of jokes to fetch (1-10).",
    )
    parser.add_argument(
        "-m",
        "--safe-mode",
        action="store_true",
        help="Enable safe mode (filters out explicit jokes).",
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


def fetch_joke(
    category: Optional[str],
    joke_type: Optional[str],
    blacklist: Optional[str] = None,
    lang: str = "en",
    id_range: Optional[str] = None,
    search: Optional[str] = None,
    amount: int = 1,
    safe_mode: bool = False,
) -> Dict:
    """
    Fetches a joke from the JokeAPI.
    """
    base_url = "https://v2.jokeapi.dev/joke"

    category_param = category if category else "Programming,Misc,Pun,Spooky,Christmas"

    url = f"{base_url}/{category_param}"

    params = {
        "lang": lang,
    }

    if joke_type:
        params["type"] = joke_type

    if blacklist:
        params["blacklistFlags"] = blacklist

    if id_range:
        params["idRange"] = id_range

    if search:
        params["contains"] = search

    if amount > 1:
        params["amount"] = amount

    if safe_mode:
        params["safe-mode"] = ""

    try:
        logging.debug(f"API Request URL: {url} with params: {params}")
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
    if "jokes" in joke_data:
        # Multiple jokes
        for idx, joke in enumerate(joke_data["jokes"], 1):
            print(f"Joke #{idx}:")
            if joke["type"] == "single":
                print(f"{joke['joke']}")
            else:
                print(f"- {joke['setup']}")
                print(f"- {joke['delivery']}")
            print()  # Empty line between jokes
    else:
        # Single joke
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

    joke = fetch_joke(
        args.category,
        args.type,
        args.blacklist,
        args.lang,
        args.id_range,
        args.search,
        args.amount,
        args.safe_mode,
    )

    if "error" in joke and joke["error"]:
        logging.error(
            f"Failed to get joke from API: {joke.get('message', 'Unknown error')}"
        )
        sys.exit(1)

    logging.debug(f"Received joke data: {joke}")
    display_joke(joke)


if __name__ == "__main__":
    main()
