#!/usr/bin/env python3

# -------------------------------------------------------
# Script: q.py
#
# Description:
# This script provides a convenient command-line utility
# for performing common tasks in Linux systems with as
# few keystrokes as possible.
#
# Template: ubuntu24.04
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "buchwald" / "q.json"

# Directory the personal scripts are installed into by the "u" subcommand.
SCRIPTS_DIR: Path = Path.home() / ".cache" / "buchwald" / "q" / "scripts"

# The configuration file may hold API keys, so it is kept private.
CONFIG_FILE_MODE = 0o600
CONFIG_DIR_MODE = 0o700

# Aliases are stored as (pattern, [replacement, ...]) pairs.
Alias = Tuple[str, List[str]]


def parse_alias_indices(arg: str, max_len: int) -> List[int]:
    """
    Parse an alias removal index argument, which can be a single integer or a range in the form 'start-end'.
    Returns a list of valid indices to remove.
    For range inputs, the first number must be smaller than the second.
    If an invalid range is specified, a fatal error is issued.
    """
    if "-" not in arg:
        try:
            idx = int(arg)
        except ValueError:
            logging.critical(f"Alias index must be an integer or range: {arg}")
            sys.exit(1)
        if not 0 <= idx < max_len:
            logging.critical(f"Index out of range: {idx}")
            sys.exit(1)
        return [idx]

    parts = arg.split("-")
    if len(parts) != 2:
        logging.critical(f"Invalid range format: {arg}. Expected format 'start-end'.")
        sys.exit(1)
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        logging.critical(f"Alias range bounds must be integers: {arg}")
        sys.exit(1)
    if start >= end:
        logging.critical(
            f"Invalid range: {arg}. The first number should be smaller than the second number."
        )
        sys.exit(1)
    if start < 0:
        logging.critical(f"Start index {start} is less than 0.")
        sys.exit(1)
    if end >= max_len:
        logging.warning(
            f"Range upper bound {end} is out of range. Clearing aliases up to index {max_len - 1} instead."
        )
        end = max_len - 1
    return list(range(start, end + 1))


# -------------------------------------------------------
# Alias Handling
# -------------------------------------------------------
def aliases_from_config(data: Any) -> List[Alias]:
    """
    Convert the serialized form [[pattern, replacement, ...], ...] into a list of
    (pattern, [replacement, ...]) pairs, skipping malformed entries.
    """
    if not isinstance(data, list):
        logging.warning("Invalid 'aliases' type in config. Expected list; using empty.")
        return []

    aliases: List[Alias] = []
    for item in data:
        # We expect item to be at least [pattern, ...]
        if not isinstance(item, list) or len(item) < 1:
            logging.warning(
                "Alias entry must be [pattern, replacement(s)...]. Skipping."
            )
            continue
        pattern = item[0]
        # The rest are replacements
        replacements = item[1:]
        if not isinstance(pattern, str) or not all(
            isinstance(r, str) for r in replacements
        ):
            logging.warning(
                "Alias pattern must be a string and replacements must be strings. Skipping malformed alias."
            )
            continue
        aliases.append((pattern, replacements))
    return aliases


def aliases_to_config(aliases: List[Alias]) -> List[List[str]]:
    """
    Convert (pattern, [r1, r2, ...]) pairs into the serialized form [pattern, r1, r2, ...].
    """
    return [[pattern, *repls] for (pattern, repls) in aliases]


def print_aliases(aliases: List[Alias]) -> None:
    """
    Prints the given aliases with aligned columns.
    """
    if not aliases:
        logging.info("No aliases are currently configured.")
        return

    logging.info("Listing aliases (index, pattern, replacements):")
    max_index_width = len(str(len(aliases) - 1))
    max_pattern_width = max(len(pat) for pat, _ in aliases)
    for i, (pat, repls) in enumerate(aliases):
        repls_str = " ".join(repls)
        print(
            f"{str(i).ljust(max_index_width)}  {pat.ljust(max_pattern_width)}  {repls_str}"
        )


def apply_aliases(aliases: List[Alias], args_list: List[str]) -> List[str]:
    """
    For each argument in args_list, if it fully matches the pattern of
    an alias, replace that single argument with the list of replacements.
    Returns the new list of arguments after all expansions.
    """
    result = []
    for arg in args_list:
        replaced = False
        for pattern, replacements in aliases:
            # Use a full match so that '^foo$' won't partially match
            if re.fullmatch(pattern, arg):
                result.extend(replacements)
                replaced = True
                break
        if not replaced:
            result.append(arg)
    return result


def upsert_alias(aliases: List[Alias], pattern: str, replacements: List[str]) -> None:
    """
    Add an alias, overwriting an existing alias with the same pattern.
    """
    logging.debug(f"Adding alias pattern '{pattern}' -> {replacements}.")
    for idx, (existing_pattern, _) in enumerate(aliases):
        if existing_pattern == pattern:
            aliases[idx] = (pattern, replacements)
            return
    aliases.append((pattern, replacements))


def remove_aliases(aliases: List[Alias], index_arg: str) -> bool:
    """
    Remove aliases by numeric index or range. Returns True if anything was removed.
    """
    indices = parse_alias_indices(index_arg, len(aliases))
    if not indices:
        return False
    for idx in sorted(indices, reverse=True):
        logging.debug(f"Removing alias at index {idx}.")
        del aliases[idx]
    return True


# -------------------------------------------------------
# Configuration File Handling
# -------------------------------------------------------
def config_holds_secrets(data: Dict[str, Any]) -> bool:
    """
    Return True if the configuration stores environment variables, which may be secrets.
    """
    return any(
        isinstance(section, dict) and section.get("environment")
        for section in data.values()
    )


def enforce_config_permissions(config_path: Path, data: Dict[str, Any]) -> None:
    """
    Restrict the permissions of a configuration file that holds environment variables but
    is readable by other users. Older versions of this script created it with the default
    umask, so existing files are repaired here.
    """
    if not config_holds_secrets(data):
        return
    try:
        mode = stat.S_IMODE(config_path.stat().st_mode)
    except OSError:
        return
    if not mode & 0o077:
        return
    try:
        os.chmod(config_path, CONFIG_FILE_MODE)
    except OSError as e:
        logging.warning(f"Could not restrict the permissions of '{config_path}': {e}")
        return
    logging.warning(
        f"'{config_path}' holds environment variables and was readable by other users "
        f"({mode:04o}); its permissions were restricted to {CONFIG_FILE_MODE:04o}."
    )


def read_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Read and parse the configuration file, returning an empty dict if it is missing or
    cannot be parsed.
    """
    if not config_path.exists():
        logging.debug(f"No '{config_path}' file found; using default configuration.")
        return {}

    try:
        with config_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logging.warning(f"Failed to read/parse '{config_path}': {e}")
        return {}

    if not isinstance(data, dict):
        logging.warning(f"'{config_path}' does not contain an object; ignoring it.")
        return {}

    enforce_config_permissions(config_path, data)
    return data


def write_config_file(config_path: Path, data: Dict[str, Any]) -> None:
    """
    Write the configuration file atomically and with private permissions, so that a
    failed write cannot corrupt an existing configuration and secrets stay unreadable
    for other users.
    """
    directory = config_path.parent
    directory_existed = directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    if not directory_existed:
        os.chmod(directory, CONFIG_DIR_MODE)

    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=".q.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), CONFIG_FILE_MODE)
            json.dump(data, f, indent=2)
        os.replace(tmp_path, config_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


# -------------------------------------------------------
# Base Configuration
# -------------------------------------------------------
class QConfig(ABC):
    """
    Abstract base class for subcommand-specific configurations.
    Each subcommand has a unique key in the top-level configuration file.
    Subclasses must define:
      - SUBCOMMAND_KEY (class-level, str)
      - from_dict(...) -> <subclass>
      - to_dict() -> dict
    """

    # The top-level key in the configuration file under which this subcommand's settings
    # are stored. For example, "o" for the "o" subcommand.
    SUBCOMMAND_KEY: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Make sure every concrete configuration declares the key it is stored under.
        """
        super().__init_subclass__(**kwargs)
        if not cls.SUBCOMMAND_KEY:
            raise TypeError(f"{cls.__name__} must define a non-empty SUBCOMMAND_KEY.")

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QConfig":
        """
        Construct a config instance of this subcommand from a dict of data.
        """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this subcommand's config instance to a dict, suitable
        for serialization into the configuration file.
        """

    @classmethod
    def load(cls, config_path: Path) -> "QConfig":
        """
        Load (and parse) the configuration file from the given path and
        extract this subcommand's configuration. If the file or section does
        not exist, return a default instance.
        """
        sub_data = read_config_file(config_path).get(cls.SUBCOMMAND_KEY, {})
        if not isinstance(sub_data, dict):
            logging.warning(
                f"Section '{cls.SUBCOMMAND_KEY}' in '{config_path}' is not an object; ignoring it."
            )
            sub_data = {}
        return cls.from_dict(sub_data)

    def save(self, config_path: Path) -> None:
        """
        Load the existing configuration (if any), merge in this subcommand's config,
        and write it back to disk at the given path. If the configuration file does not exist, create it.
        """
        data = read_config_file(config_path)
        data[self.SUBCOMMAND_KEY] = self.to_dict()

        try:
            write_config_file(config_path, data)
        except OSError as e:
            logging.error(f"Failed to write config to '{config_path}': {e}")
            return

        logging.info(
            f"Configuration for subcommand '{self.SUBCOMMAND_KEY}' saved to '{config_path}'."
        )


# -------------------------------------------------------
# "o" Subcommand Configuration
# -------------------------------------------------------
class OConfig(QConfig):
    """
    Configuration for the "o" subcommand.
    Stores a command (e.g., "xdg-open") and aliases
    under the "o" key in the configuration file.
    """

    SUBCOMMAND_KEY: ClassVar[str] = "o"

    def __init__(
        self,
        command: Optional[str] = None,
        aliases: Optional[List[Alias]] = None,
    ) -> None:
        self.command = command
        self.aliases = aliases if aliases is not None else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OConfig":
        return cls(
            command=data.get("command"),
            aliases=aliases_from_config(data.get("aliases", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "aliases": aliases_to_config(self.aliases),
        }


# -------------------------------------------------------
# Base Subcommand Interface
# -------------------------------------------------------
class QSubcommand(ABC):
    """
    Abstract base class for 'q' subcommands.
    Each subcommand is responsible for:
      - Registering its command-line interface.
      - Handling its own configuration (via a QConfig subclass).
      - Executing subcommand logic in run().
    """

    @abstractmethod
    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """
        Register the subcommand's parser, options, and arguments with the subparsers.
        Must call `set_defaults(subcommand_obj=self)` on the parser.
        """

    @abstractmethod
    def run(self, args: argparse.Namespace) -> None:
        """
        Execute the subcommand logic given parsed command-line arguments.
        """


# -------------------------------------------------------
# "o" Subcommand Implementation
# -------------------------------------------------------
class OSubcommand(QSubcommand):
    """
    Subcommand to open a file/path/URL (or apply regex-based alias) using a configured command.
    Defaults to "xdg-open" if no command is explicitly set.
    Aliases are stored as list of (pattern, [replacement1, replacement2, ...]) pairs.
    By default, the command is executed detached with no output.
    Use the '--foreground' (or '-F') option to run in foreground and print its output.
    """

    def __init__(self) -> None:
        self.config: Optional[OConfig] = None

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "o",
            help="Open a file/path/URL (or alias) using a configured command.",
        )
        parser.add_argument(
            "file_or_alias",
            nargs="?",
            help="File/path/URL to open. If omitted, nothing is opened.",
        )
        command_group = parser.add_mutually_exclusive_group()
        command_group.add_argument(
            "-c",
            "--command",
            type=str,
            help="Set or update the command used to open paths (e.g., 'firefox').",
        )
        command_group.add_argument(
            "-C",
            "--clear-command",
            action="store_true",
            help="Clear any custom command, so that the default ('xdg-open') will be used.",
        )
        parser.add_argument(
            "-a",
            "--add-alias",
            nargs=argparse.REMAINDER,
            help="Create an alias. Example: -a ^foo$ http://example.org",
        )
        parser.add_argument(
            "-A",
            "--remove-alias",
            metavar="INDEX",
            help="Remove an existing alias by its numeric index or range (e.g., '6' or '6-10').",
        )
        parser.add_argument(
            "-l",
            "--list-aliases",
            action="store_true",
            help="List all aliases for the 'o' subcommand with their indexes.",
        )
        parser.add_argument(
            "-F",
            "--foreground",
            action="store_true",
            help="Run the open command in the foreground and print its output.",
        )
        parser.add_argument(
            "-X",
            "--disable-aliases",
            action="store_true",
            help="Disable applying aliases for this run.",
        )

        parser.set_defaults(subcommand_obj=self)

    def run(self, args: argparse.Namespace) -> None:
        """
        Execute the "o" subcommand logic given parsed command-line arguments.
        """
        config_path = Path(args.config)
        self.config = OConfig.load(config_path)

        # Handle configuration updates
        if self._handle_config_updates(args):
            self.config.save(config_path)

        # Possibly list aliases
        if args.list_aliases:
            print_aliases(self.config.aliases)

        # Possibly open file/path/URL
        if args.file_or_alias:
            self._open_file_or_alias(
                args.file_or_alias, args.disable_aliases, args.foreground
            )
        else:
            logging.info("No path specified, not opening anything")

    def _handle_config_updates(self, args: argparse.Namespace) -> bool:
        """
        Handle additions, removals, or updates to the configuration based on CLI arguments.
        Returns True if the configuration was changed, False otherwise.
        """
        config_changed = False

        # Clear command if requested
        if args.clear_command:
            logging.debug("Clearing custom command (will use default 'xdg-open').")
            self.config.command = None
            config_changed = True

        # Set/update command if requested
        if args.command:
            logging.debug(f"Setting command to '{args.command}'.")
            self.config.command = args.command
            config_changed = True

        # Add alias if requested; if an alias with the same pattern exists, overwrite it.
        if args.add_alias:
            upsert_alias(self.config.aliases, args.add_alias[0], args.add_alias[1:])
            config_changed = True

        # Remove alias if requested (by index or range)
        if args.remove_alias is not None and remove_aliases(
            self.config.aliases, args.remove_alias
        ):
            config_changed = True

        return config_changed

    def _open_file_or_alias(
        self, file_or_alias: str, disable_aliases: bool, foreground: bool
    ) -> None:
        """
        If a file/path/URL (or alias) was provided, open it with the configured command.
        Applies alias expansions unless explicitly disabled.
        """
        # Expand the single argument into possibly multiple if an alias matches
        if disable_aliases:
            resolved_args = [file_or_alias]
        else:
            resolved_args = apply_aliases(self.config.aliases, [file_or_alias])

        cmd = self.config.command if self.config.command is not None else "xdg-open"

        if not resolved_args:
            logging.info("Alias expansion produced no argument. Nothing to open.")
            return

        # If multiple arguments result, open each in turn
        for rarg in resolved_args:
            logging.info(f"Opening '{rarg}' using '{cmd}'")
            self._open_with_command(cmd, rarg, foreground)

    @staticmethod
    def _open_with_command(command: str, target: str, foreground: bool = False) -> None:
        """
        Open the given target with the specified command.
        If foreground is True, run in the foreground capturing output.
        Otherwise, run detached with no output.
        """
        try:
            if foreground:
                result = subprocess.run(
                    [command, target],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
            else:
                subprocess.Popen(
                    [command, target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except OSError as e:
            logging.error(f"Failed to open '{target}' with '{command}': {e}")


# -------------------------------------------------------
# "s" Subcommand Configuration
# -------------------------------------------------------
class SConfig(QConfig):
    """
    Configuration for the "s" subcommand.
    Stores aliases and environment variables under the "s" key in the configuration file.
    """

    SUBCOMMAND_KEY: ClassVar[str] = "s"

    def __init__(
        self,
        aliases: Optional[List[Alias]] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> None:
        self.aliases = aliases if aliases is not None else []
        self.environment = environment if environment is not None else {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SConfig":
        env_data = data.get("environment", {})
        if not isinstance(env_data, dict):
            logging.warning(
                "Invalid 'environment' type in config. Expected dict; using empty."
            )
            env_data = {}
        return cls(
            aliases=aliases_from_config(data.get("aliases", [])),
            environment={str(k): str(v) for k, v in env_data.items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aliases": aliases_to_config(self.aliases),
            "environment": self.environment,
        }


# -------------------------------------------------------
# "s" Subcommand Implementation
# -------------------------------------------------------
class SSubcommand(QSubcommand):
    """
    Subcommand that manages personal scripts from GitHub or runs them
    with optional aliases, environment variables, and Docker usage.
    """

    def __init__(self) -> None:
        self.config: Optional[SConfig] = None

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "s",
            help="Run personal scripts with optional aliases and Docker usage.",
        )
        parser.add_argument(
            "-a",
            "--add-alias",
            nargs=argparse.REMAINDER,
            help="Create an alias. Example: -a ^foo$ /some/replacement -h",
        )
        parser.add_argument(
            "-A",
            "--remove-alias",
            metavar="INDEX",
            help="Remove an existing alias by its numeric index or range (e.g., '6' or '6-10').",
        )
        parser.add_argument(
            "-l",
            "--list-aliases",
            action="store_true",
            help="List all aliases for the 's' subcommand with their indexes.",
        )
        parser.add_argument(
            "-e",
            "--add-env",
            nargs=2,
            metavar=("KEY", "VALUE"),
            action="append",
            help="Set an environment variable for the 's' subcommand. Can be used multiple times.",
        )
        parser.add_argument(
            "-E",
            "--remove-env",
            metavar="KEY",
            action="append",
            help="Remove an environment variable from the 's' subcommand configuration. Can be used multiple times.",
        )
        parser.add_argument(
            "-L",
            "--list-envs",
            action="store_true",
            help="List all environment variables stored in the 's' subcommand configuration.",
        )
        parser.add_argument(
            "-X",
            "--disable-aliases",
            action="store_true",
            help="Disable applying aliases for this run.",
        )
        parser.add_argument(
            "-s",
            "--list-scripts",
            action="store_true",
            help="List all available scripts in SCRIPTS_DIR in a tree-like structure.",
        )
        parser.add_argument(
            "-m",
            "--execution-mode",
            choices=["docker", "venv", "direct"],
            help="Set the execution mode: 'docker', 'venv', or 'direct'. If omitted, the script attempts Docker if possible, then venv, then direct.",
        )
        parser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Script and arguments to run.",
        )
        parser.set_defaults(subcommand_obj=self)

    def run(self, args: argparse.Namespace) -> None:
        """
        Execute the "s" subcommand logic given parsed command-line arguments.
        """
        config_path = Path(args.config)
        self.config = SConfig.load(config_path)

        # Handle config updates (aliases and environment variables)
        if self._handle_config_updates(args):
            self.config.save(config_path)

        # Possibly list aliases
        if args.list_aliases:
            print_aliases(self.config.aliases)

        # Possibly list environment variables
        if args.list_envs:
            self._list_environment()

        # Possibly list scripts
        if args.list_scripts:
            self._list_scripts()
            return

        # Run script if there are arguments
        if args.args:
            self._run_script_with_arguments(args)

    def _handle_config_updates(self, args: argparse.Namespace) -> bool:
        """
        Handle alias and environment variable additions and removals from CLI arguments.
        Returns True if the configuration was changed, False otherwise.
        """
        config_changed = False

        # Add alias if requested; overwrite if an alias with the same pattern exists.
        if args.add_alias:
            upsert_alias(self.config.aliases, args.add_alias[0], args.add_alias[1:])
            config_changed = True

        # Remove alias if requested (by index or range)
        if args.remove_alias is not None and remove_aliases(
            self.config.aliases, args.remove_alias
        ):
            config_changed = True

        # Set environment variable(s) if requested; overwrite if key already exists.
        if args.add_env:
            for key, value in args.add_env:
                logging.debug(f"Setting environment variable '{key}'.")
                self.config.environment[key] = value
                config_changed = True

        # Unset environment variable(s) if requested
        if args.remove_env:
            for key in args.remove_env:
                if key in self.config.environment:
                    logging.debug(f"Removing environment variable '{key}'.")
                    del self.config.environment[key]
                    config_changed = True
                else:
                    logging.warning(
                        f"Environment variable '{key}' not found in configuration."
                    )

        return config_changed

    def _list_environment(self) -> None:
        """
        Prints the currently stored environment variables with aligned keys.
        """
        if not self.config.environment:
            logging.info("No environment variables are currently configured.")
            return

        logging.info("Listing environment variables:")
        max_key_width = max(len(key) for key in self.config.environment)
        for key, value in self.config.environment.items():
            print(f"{key.ljust(max_key_width)} = {value}")

    def _run_script_with_arguments(self, args: argparse.Namespace) -> None:
        """
        Run a script using the provided command-line arguments. Handles:
         - Checking if scripts directory exists
         - Applying aliases (unless disabled)
         - Determining the actual script vs. wrapper arguments
         - Checking for # Template
         - Determining execution mode
         - Invoking script via Docker, venv, or directly
        """
        # We need a valid scripts directory
        if not SCRIPTS_DIR.is_dir():
            logging.error(
                f"The scripts directory '{SCRIPTS_DIR}' does not exist. "
                "Please run 'q u' to update/install personal scripts."
            )
            return

        # argparse leaves the '--' that separates q's options from the wrapper's options
        # in place. Drop only that one, so that any further '--' reaches the script.
        all_args = args.args
        if all_args and all_args[0] == "--":
            all_args = all_args[1:]

        # Apply alias expansion to all arguments (unless disabled)
        if not args.disable_aliases:
            all_args = apply_aliases(self.config.aliases, all_args)

        # Parse out wrapper arguments, the script name, and script arguments
        wrapper_args, script, script_args = self._parse_script_args(all_args)

        if not script:
            logging.error("No valid Python script found in arguments. Aborting.")
            return

        script_path = Path(script)
        if not script_path.is_file():
            logging.error(f"Script file '{script_path}' not found.")
            return

        execution_mode = args.execution_mode or self._detect_execution_mode(script_path)

        # Execute according to chosen mode
        if execution_mode == "docker":
            self._run_via_wrapper(
                "docker.py", wrapper_args, str(script_path), script_args
            )
        elif execution_mode == "venv":
            self._run_via_wrapper(
                "venver.py", wrapper_args, str(script_path), script_args
            )
        else:
            self._run_local_script(str(script_path), wrapper_args, script_args)

    def _detect_execution_mode(self, script_path: Path) -> str:
        """
        Pick an execution mode: Docker when it is available and the script declares a
        template, a virtual environment when that is possible, and direct execution
        otherwise.
        """
        if shutil.which("docker") and self._script_has_template(script_path):
            return "docker"

        try:
            import venv  # noqa: F401
        except ImportError:
            return "direct"
        return "venv"

    def _parse_script_args(
        self, all_args: List[str]
    ) -> Tuple[List[str], str, List[str]]:
        """
        Determine the first argument that corresponds to a file (absolute or relative
        to SCRIPTS_DIR) which exists and whose first line contains 'python'.
        Everything before that file is treated as wrapper arguments (options for
        docker.py or venver.py), and everything after it is script arguments.

        Returns:
            (wrapper_args, script, script_args)
            script will be an empty string if no valid Python script was found.
        """
        wrapper_args = []

        for i, arg in enumerate(all_args):
            # Check if arg is a path (absolute or relative to SCRIPTS_DIR) that exists
            candidate_path = Path(arg)
            if not candidate_path.is_absolute():
                candidate_path = SCRIPTS_DIR / arg

            if not candidate_path.is_file():
                wrapper_args.append(arg)
                continue

            # Read the first line to see if it contains 'python'
            try:
                with candidate_path.open(encoding="utf-8") as f:
                    first_line = f.readline()
            except OSError:
                wrapper_args.append(arg)
                continue

            if "python" in first_line:
                return (wrapper_args, str(candidate_path), all_args[i + 1 :])
            wrapper_args.append(arg)

        # If we never found a valid Python script, just return
        return (all_args, "", [])

    def _build_environment(self) -> Dict[str, str]:
        """
        Returns the environment for a script run, extended with the configured variables.
        """
        env = os.environ.copy()
        env.update(self.config.environment)
        return env

    def _config_env_arguments(self) -> List[str]:
        """
        Returns '-e KEY' options that make the wrapper forward the configured environment
        variables into the container. Only names are passed on the command line; the values
        are taken from the wrapper's own environment, so secrets stay out of the process list.
        """
        env_args = []
        for key in sorted(self.config.environment):
            env_args += ["-e", key]
        return env_args

    def _run_via_wrapper(
        self,
        wrapper_name: str,
        wrapper_args: List[str],
        script: str,
        script_args: List[str],
    ) -> None:
        """
        Run a script through one of the wrappers in the 'Meta' directory:
          <wrapper> [wrapper_args] [script] [script_args]
        """
        wrapper_script = SCRIPTS_DIR / "Meta" / wrapper_name
        if not wrapper_script.is_file():
            logging.error(
                f"Cannot find '{wrapper_name}' in the 'Meta' directory. "
                "Please run 'q u' again or check the repository structure."
            )
            return

        cmd = [str(wrapper_script)]
        if wrapper_name == "docker.py":
            cmd += self._config_env_arguments()
        cmd += [*wrapper_args, script, *script_args]

        logging.info(f"Running wrapper command: {shlex.join(cmd)}")
        self._run_command(cmd, wrapper_name)

    def _run_local_script(
        self,
        script: str,
        wrapper_args: List[str],
        script_args: List[str],
    ) -> None:
        """
        Run a script locally (without docker.py or venver.py).
        """
        if wrapper_args:
            # These are options for docker.py/venver.py; passing them on would corrupt
            # the script's own argument list.
            logging.warning(
                "Ignoring wrapper options %s: they are not supported in direct execution mode.",
                " ".join(wrapper_args),
            )

        cmd = [script, *script_args]
        logging.info(f"Running local script command: {shlex.join(cmd)}")
        self._run_command(cmd, script)

    def _run_command(self, cmd: List[str], description: str) -> None:
        """
        Run a command with the configured environment and report a non-zero exit code.
        """
        try:
            result = subprocess.run(cmd, env=self._build_environment(), check=False)
        except OSError as e:
            logging.error(f"Error running '{description}': {e}")
            return
        if result.returncode != 0:
            logging.error(
                f"'{description}' exited with return code {result.returncode}"
            )

    @staticmethod
    def _script_has_template(script_path: Path) -> bool:
        """
        Return True if the script at script_path contains a line with '# Template:'.
        Return False otherwise (or if the file cannot be read).
        """
        try:
            with script_path.open(encoding="utf-8") as f:
                return any("# Template:" in line for line in f)
        except OSError as e:
            logging.debug(f"Failed to read script '{script_path}': {e}")
            return False

    def _list_scripts(self) -> None:
        """
        List all available Python scripts in SCRIPTS_DIR (except for anything in 'Meta')
        in a tree-like structure.
        """
        if not SCRIPTS_DIR.is_dir():
            logging.error(
                f"The scripts directory '{SCRIPTS_DIR}' does not exist. "
                "Please run 'q u' to update/install personal scripts."
            )
            return

        print(f"Available scripts in '{SCRIPTS_DIR}':")
        if not self._directory_has_python_files_excluding_meta(SCRIPTS_DIR):
            print("  No Python scripts found.")
            return

        self._print_scripts_tree(SCRIPTS_DIR, is_root=True)

    @staticmethod
    def _directory_has_python_files_excluding_meta(dirpath: Path) -> bool:
        """
        Return True if there is any .py file (directly or in subdirectories) below dirpath,
        excluding anything under 'Meta'.
        """
        for _, dirs, files in os.walk(dirpath):
            if "Meta" in dirs:
                dirs.remove("Meta")
            if any(f.endswith(".py") for f in files):
                return True
        return False

    def _print_scripts_tree(
        self,
        directory: Path,
        prefix: str = "",
        is_last: bool = True,
        is_root: bool = False,
    ) -> None:
        """
        Recursively print a tree-like structure of all .py files under 'directory',
        skipping the 'Meta' folder entirely.
        """
        if not is_root:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + directory.name + "/")
            prefix += "    " if is_last else "│   "

        entries = self._list_relevant_entries(directory)
        for i, entry in enumerate(entries):
            is_entry_last = i == len(entries) - 1
            if entry.is_dir():
                self._print_scripts_tree(entry, prefix, is_entry_last, is_root=False)
            else:
                connector = "└── " if is_entry_last else "├── "
                print(prefix + connector + entry.name)

    def _list_relevant_entries(self, dirpath: Path) -> List[Path]:
        """
        Return a sorted list of relevant subdirectories (those that contain .py files somewhere inside,
        excluding 'Meta') plus .py files in this directory (also ignoring 'Meta').
        """
        dirs_in_dir = []
        files_in_dir = []

        for item in sorted(dirpath.iterdir()):
            if item.is_dir():
                if (
                    item.name != "Meta"
                    and self._directory_has_python_files_excluding_meta(item)
                ):
                    dirs_in_dir.append(item)
            elif item.suffix == ".py":
                files_in_dir.append(item)

        return [*dirs_in_dir, *files_in_dir]


# -------------------------------------------------------
# "u" Subcommand Implementation
# -------------------------------------------------------
class USubcommand(QSubcommand):
    """
    Subcommand that auto-updates this script from a remote URL,
    and also updates personal scripts from a tarball if needed.
    Compares sha512 hashes (local vs. remote) for both q.py and master.tar.gz.
    """

    UPDATE_URL = (
        "https://raw.githubusercontent.com/"
        "zit-hb/Personal-Scripts/refs/heads/master/Meta/q.py"
    )
    TARBALL_URL = (
        "https://github.com/zit-hb/Personal-Scripts/archive/refs/heads/master.tar.gz"
    )

    TARBALL_HASH_FILE: Path = SCRIPTS_DIR / ".tarball.sha512"

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "u",
            help="Auto-update this script (q.py) and personal scripts tarball from the remote repository.",
        )
        parser.add_argument(
            "-c",
            "--only-check",
            action="store_true",
            help="Only check for new versions (compare sha512), do not replace/update anything.",
        )
        parser.add_argument(
            "-q",
            "--only-q-script",
            action="store_true",
            help="Only update the 'q' script, do not update the personal scripts tarball.",
        )
        parser.add_argument(
            "-s",
            "--only-scripts",
            action="store_true",
            help="Only update the personal scripts tarball, do not update the 'q' script.",
        )
        parser.set_defaults(subcommand_obj=self)

    def run(self, args: argparse.Namespace) -> None:
        """
        Perform update logic for both the local q.py script and the personal scripts tarball.
        """
        do_update_q = True
        do_update_scripts = True

        # If user specified only one or the other, adjust accordingly
        if args.only_q_script and not args.only_scripts:
            do_update_scripts = False
        elif args.only_scripts and not args.only_q_script:
            do_update_q = False

        # If neither is specified, we do both (already True).
        # If both are specified, we also do both.

        # Possibly update q.py
        if do_update_q:
            self._handle_q_script_update(args.only_check)

        # Possibly update personal scripts tarball
        if do_update_scripts:
            self._handle_scripts_tarball_update(args.only_check)

    def _handle_q_script_update(self, only_check: bool) -> None:
        """
        Check and possibly update q.py.
        """
        local_path = Path(__file__).resolve()
        logging.info(f"Checking for a new version of 'q' script at: {self.UPDATE_URL}")

        remote_data = self._download(self.UPDATE_URL)
        if remote_data is None:
            logging.error("Failed to download the remote 'q' script.")
            return

        local_hash = self._calculate_sha512_file(local_path)
        remote_hash = self._calculate_sha512_data(remote_data)

        if not local_hash:
            logging.error(
                "Could not read local 'q' script for hashing; skipping update check."
            )
            return

        if local_hash == remote_hash:
            print("Your 'q' script is already up to date.")
            return

        print("A new version of the 'q' script is available.")
        if only_check:
            logging.info("Not updating because --only-check was specified.")
            return

        if self._update_local_script(local_path, remote_data):
            print("The 'q' script was updated successfully.")
        else:
            logging.error("Failed to update the local 'q' script.")

    def _handle_scripts_tarball_update(self, only_check: bool) -> None:
        """
        Check and possibly update the personal scripts tarball.
        """
        logging.info(
            f"Checking for a new version of personal scripts tarball at: {self.TARBALL_URL}"
        )

        remote_tarball_data = self._download(self.TARBALL_URL)
        if not remote_tarball_data:
            logging.error("Failed to download the tarball.")
            return

        remote_hash = self._calculate_sha512_data(remote_tarball_data)
        if self._read_local_tarball_hash() == remote_hash and SCRIPTS_DIR.is_dir():
            print("Your personal scripts are already up to date.")
            return

        print("A new version of the personal scripts tarball is available.")
        if only_check:
            logging.info("Not updating because --only-check was specified.")
            return

        # Proceed with update: extract into a staging directory, swap it in, store new hash
        with tempfile.TemporaryDirectory() as tmpdir:
            tarball_path = os.path.join(tmpdir, "master.tar.gz")
            try:
                with open(tarball_path, "wb") as out_file:
                    out_file.write(remote_tarball_data)
            except OSError as e:
                logging.error(f"Failed to store the downloaded tarball: {e}")
                return

            if not self._install_scripts_tree(tarball_path):
                return

        self._write_local_tarball_hash(remote_hash)
        print("Personal scripts updated successfully.")

    def _install_scripts_tree(self, tarball_path: str) -> bool:
        """
        Extract the tarball into a staging directory next to SCRIPTS_DIR and only then swap
        it in. The previous installation is kept until the swap succeeded, so a failed
        update never leaves the user without any scripts.
        """
        logging.info(f"Updating the personal scripts in '{SCRIPTS_DIR}' now...")

        staging = SCRIPTS_DIR.with_name(SCRIPTS_DIR.name + ".new")
        backup = SCRIPTS_DIR.with_name(SCRIPTS_DIR.name + ".old")

        try:
            for leftover in (staging, backup):
                if leftover.exists():
                    shutil.rmtree(leftover)
            staging.mkdir(parents=True)
        except OSError as e:
            logging.error(f"Could not prepare the staging directory '{staging}': {e}")
            return False

        try:
            if not self._extract_tarball(tarball_path, staging):
                return False
            self._flatten_extracted_tree(staging)
            return self._swap_in_scripts_tree(staging, backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _swap_in_scripts_tree(staging: Path, backup: Path) -> bool:
        """
        Replace SCRIPTS_DIR with the staging directory, restoring the previous
        installation if the swap fails halfway through.
        """
        had_previous = SCRIPTS_DIR.exists()
        try:
            if had_previous:
                os.replace(SCRIPTS_DIR, backup)
            try:
                os.replace(staging, SCRIPTS_DIR)
            except OSError:
                if had_previous:
                    os.replace(backup, SCRIPTS_DIR)
                raise
        except OSError as e:
            logging.error(f"Failed to install the new scripts directory: {e}")
            return False

        if had_previous:
            shutil.rmtree(backup, ignore_errors=True)
        return True

    @staticmethod
    def _extract_tarball(tarball_path: str, destination: Path) -> bool:
        """
        Extract the tarball into destination, refusing members that would escape it.
        """
        try:
            with tarfile.open(tarball_path, "r:gz") as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(destination, filter="data")
                else:
                    # Older Python versions have no extraction filters, so the members
                    # have to be validated by hand.
                    USubcommand._reject_unsafe_members(tar, destination)
                    tar.extractall(destination)
            return True
        except (OSError, ValueError, tarfile.TarError) as e:
            logging.error(f"Failed to extract scripts: {e}")
            return False

    @staticmethod
    def _reject_unsafe_members(tar: tarfile.TarFile, destination: Path) -> None:
        """
        Raise a ValueError if the archive contains links or paths that point outside of
        the destination directory.
        """
        base = os.path.realpath(destination)
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing to extract link member '{member.name}'.")
            target = os.path.realpath(os.path.join(base, member.name))
            if target != base and not target.startswith(base + os.sep):
                raise ValueError(
                    f"Refusing to extract '{member.name}' outside of '{base}'."
                )

    @staticmethod
    def _flatten_extracted_tree(directory: Path) -> None:
        """
        GitHub tarballs wrap everything in a single top-level directory. Move its contents
        up so that the scripts end up directly in the destination.
        """
        entries = list(directory.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            return

        top_level_dir = entries[0]
        for item in top_level_dir.iterdir():
            shutil.move(str(item), str(directory))
        top_level_dir.rmdir()

    def _read_local_tarball_hash(self) -> str:
        """
        Read and return the stored tarball sha512 hash from TARBALL_HASH_FILE.
        If the file does not exist or can't be read, return an empty string.
        """
        try:
            return self.TARBALL_HASH_FILE.read_text(encoding="utf-8").strip()
        except OSError as e:
            logging.debug(f"Failed to read local tarball hash file: {e}")
            return ""

    def _write_local_tarball_hash(self, new_hash: str) -> None:
        """
        Write the new tarball sha512 hash to TARBALL_HASH_FILE.
        """
        try:
            self.TARBALL_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.TARBALL_HASH_FILE.write_text(new_hash, encoding="utf-8")
        except OSError as e:
            logging.debug(f"Failed to write tarball hash file: {e}")

    @staticmethod
    def _download(url: str) -> Optional[bytes]:
        """
        Download the given URL and return its bytes, or None on error.
        """
        if not url.startswith("https://"):
            logging.error(f"Refusing to download from a non-HTTPS URL: {url}")
            return None
        try:
            with urllib.request.urlopen(url) as response:
                return response.read()
        except (OSError, ValueError) as e:
            logging.error(f"Error downloading '{url}': {e}")
            return None

    @staticmethod
    def _calculate_sha512_data(data: bytes) -> str:
        """
        Return the sha512 hex digest for the given data.
        """
        return hashlib.sha512(data).hexdigest()

    @staticmethod
    def _calculate_sha512_file(path: Path) -> str:
        """
        Return the sha512 hex digest for the file at the given path.
        """
        sha = hashlib.sha512()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha.update(chunk)
        except OSError as e:
            logging.error(f"Error reading local script '{path}': {e}")
            return ""
        return sha.hexdigest()

    @staticmethod
    def _update_local_script(local_path: Path, new_data: bytes) -> bool:
        """
        Replace the local script at local_path with new_data. The new version is written to
        a temporary file next to it and then moved into place, so an interrupted update
        cannot leave a half-written (and therefore unusable) script behind.
        """
        tmp_path: Optional[Path] = None
        try:
            mode = stat.S_IMODE(local_path.stat().st_mode)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(local_path.parent), prefix=f".{local_path.name}.", suffix=".new"
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as f:
                f.write(new_data)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, mode)
            os.replace(tmp_path, local_path)
            return True
        except OSError as e:
            logging.error(f"Error writing updated script to '{local_path}': {e}")
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            return False


# -------------------------------------------------------
# Main Application Logic
# -------------------------------------------------------
def create_main_parser() -> argparse.ArgumentParser:
    """
    Create and return the main argument parser for the 'q' utility.
    """
    parser = argparse.ArgumentParser(
        description="q: The ultimate command-line utility for quick tasks."
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
    parser.add_argument(
        "-f",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configuration file (default: %(default)s)",
    )
    return parser


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure the logging level and format based on the verbose/debug flags.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main() -> None:
    """
    Main entry point for the 'q' utility, parsing arguments and dispatching subcommands.
    """
    main_parser = create_main_parser()
    subparsers = main_parser.add_subparsers(dest="subcommand", required=True)

    # Register subcommands
    subcommands = [
        OSubcommand(),
        SSubcommand(),
        USubcommand(),
    ]
    for subcmd in subcommands:
        subcmd.register_parser(subparsers)

    args = main_parser.parse_args()
    setup_logging(verbose=args.verbose, debug=args.debug)

    subcmd_obj = getattr(args, "subcommand_obj", None)
    if not subcmd_obj:
        logging.error("No valid subcommand selected.")
        sys.exit(1)

    subcmd_obj.run(args)


if __name__ == "__main__":
    main()
