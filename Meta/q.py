#!/usr/bin/env python3

# -------------------------------------------------------
# Script: q.py
#
# Description:
# This script provides a convenient command-line utility
# for performing common tasks in Linux systems with as
# few keystrokes as possible.
#
# Template: ubuntu22.04
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "buchwald" / "q.json"


# -------------------------------------------------------
# Base Configuration
# -------------------------------------------------------
class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


class QConfig(ABC):
    """
    Abstract base class for subcommand-specific configurations.
    Each subcommand has a unique key in the top-level configuration file.
    Subclasses must define:
      - SUBCOMMAND_KEY (class-level, str)
      - from_dict(...) -> <subclass>
      - to_dict() -> dict
    """

    @classproperty
    @abstractmethod
    def SUBCOMMAND_KEY(self) -> str:
        """
        The top-level key in the configuration file under which this
        subcommand's settings are stored. For example, "o" for the "o" subcommand.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QConfig":
        """
        Construct a config instance of this subcommand from a dict of data.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this subcommand's config instance to a dict, suitable
        for serialization into the configuration file.
        """
        pass

    @classmethod
    def load(cls, config_path: Path) -> "QConfig":
        """
        Load (and parse) the configuration file from the given path and
        extract this subcommand's configuration. If the file or section does
        not exist, return a default instance.
        """
        if not config_path.exists():
            logging.debug(
                f"No '{config_path}' file found; using default config for subcommand '{cls.__name__}'."
            )
            return cls.from_dict({})

        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to read/parse '{config_path}': {e}")
            return cls.from_dict({})

        sub_data = data.get(cls.SUBCOMMAND_KEY, {})
        return cls.from_dict(sub_data)

    def save(self, config_path: Path) -> None:
        """
        Load the existing configuration (if any), merge in this subcommand's config,
        and write it back to disk at the given path. If the configuration file does not exist, create it.
        """
        existing_data: Dict[str, Any] = {}

        # Load existing config if possible
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                logging.warning(
                    f"Failed to read existing config from '{config_path}': {e}"
                )

        # Update subcommand-specific section
        existing_data[self.SUBCOMMAND_KEY] = self.to_dict()

        # Write updated config
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)
            logging.info(
                f"Configuration for subcommand '{self.SUBCOMMAND_KEY}' saved to '{config_path}'."
            )
        except Exception as e:
            logging.error(f"Failed to write config to '{config_path}': {e}")


# -------------------------------------------------------
# "o" Subcommand Configuration
# -------------------------------------------------------
class OConfig(QConfig):
    """
    Configuration for the "o" subcommand.
    Stores a command (e.g., "xdg-open") and aliases
    under the "o" key in the configuration file.
    """

    @classproperty
    def SUBCOMMAND_KEY(self) -> str:
        return "o"

    def __init__(
        self,
        command: Optional[str] = None,
        aliases: Optional[List[Tuple[str, List[str]]]] = None,
    ) -> None:
        self.command = command
        # Each alias is (pattern, list_of_replacements)
        self.aliases = aliases if aliases is not None else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OConfig":
        command = data.get("command", None)
        aliases_data = data.get("aliases", [])
        if not isinstance(aliases_data, list):
            logging.warning(
                "Invalid 'aliases' type in config. Expected list; using empty."
            )
            aliases_data = []
        clean_aliases: List[Tuple[str, List[str]]] = []
        for item in aliases_data:
            # We now expect item to be at least [pattern, ...]
            if isinstance(item, list) and len(item) >= 1:
                pattern = item[0]
                # The rest are replacements
                replacements = item[1:]
                if isinstance(pattern, str) and all(
                    isinstance(r, str) for r in replacements
                ):
                    clean_aliases.append((pattern, replacements))
                else:
                    logging.warning(
                        "Alias pattern must be a string and replacements must be strings. Skipping malformed alias."
                    )
            else:
                logging.warning(
                    "Alias entry must be [pattern, replacement(s)...]. Skipping."
                )
        return cls(command=command, aliases=clean_aliases)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            # Convert (pattern, [r1, r2, ...]) -> [pattern, r1, r2, ...]
            "aliases": [[pattern] + repls for (pattern, repls) in self.aliases],
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
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace) -> None:
        """
        Execute the subcommand logic given parsed command-line arguments.
        """
        pass


# -------------------------------------------------------
# "o" Subcommand Implementation
# -------------------------------------------------------
class OSubcommand(QSubcommand):
    """
    Subcommand to open a file/path/URL (or apply regex-based alias) using a configured command.
    Defaults to "xdg-open" if no command is explicitly set.
    Aliases are stored as list of (pattern, [replacement1, replacement2, ...]) pairs.
    By default, the command is executed detached with no output.
    Use the '--foreground' (or '-F') option to run in foreground and print output.
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
            "--alias",
            nargs=argparse.REMAINDER,
            help="Create an alias. Example: -a ^foo$ http://example.org",
        )
        parser.add_argument(
            "-A",
            "--remove-alias",
            metavar="INDEX",
            help="Remove an existing alias by its numeric index (as shown by --list-aliases).",
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
        config_changed = self._handle_config_updates(args)

        # Save config if any changes were made
        if config_changed:
            self.config.save(config_path)

        # Possibly list aliases
        self._list_aliases_if_requested(args)

        # Possibly open file/path/URL
        self._open_file_or_alias_if_requested(args)

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

        # Add alias if requested
        if args.alias:
            pattern = args.alias[0]
            replacements = args.alias[1:]
            logging.debug(f"Adding alias pattern '{pattern}' -> {replacements}.")
            self.config.aliases.append((pattern, replacements))
            config_changed = True

        # Remove alias if requested (by index)
        if args.remove_alias is not None:
            try:
                idx = int(args.remove_alias)
                if 0 <= idx < len(self.config.aliases):
                    logging.debug(f"Removing alias at index {idx}.")
                    del self.config.aliases[idx]
                    config_changed = True
                else:
                    logging.warning(f"Index out of range: {idx}")
            except ValueError:
                logging.warning(f"Alias index must be an integer: {args.remove_alias}")

        return config_changed

    def _list_aliases_if_requested(self, args: argparse.Namespace) -> None:
        """
        If --list-aliases is requested, prints the currently configured aliases.
        """
        if args.list_aliases:
            if self.config.aliases:
                logging.info("Listing aliases (index, pattern, replacements):")
                for i, (pat, repls) in enumerate(self.config.aliases):
                    print(f"{i}\t{pat}\t{repls}")
            else:
                logging.info("No aliases are currently configured.")

    def _open_file_or_alias_if_requested(self, args: argparse.Namespace) -> None:
        """
        If a file/path/URL (or alias) was provided, open it with the configured command.
        Applies alias expansions unless explicitly disabled.
        """
        if not args.file_or_alias:
            logging.info("No path specified, not opening anything")
            return

        # Expand the single argument into possibly multiple if an alias matches
        if args.disable_aliases:
            resolved_args = [args.file_or_alias]
        else:
            resolved_args = self._apply_aliases_to_arglist(
                self.config.aliases, [args.file_or_alias]
            )

        cmd = self.config.command if self.config.command is not None else "xdg-open"

        if not resolved_args:
            logging.info("Alias expansion produced no argument. Nothing to open.")
            return

        # If multiple arguments result, open each in turn
        for rarg in resolved_args:
            logging.info(f"Opening '{rarg}' using '{cmd}'")
            self._open_with_command(cmd, rarg, args.foreground)

    @staticmethod
    def _apply_aliases_to_arglist(
        aliases: List[Tuple[str, List[str]]], args_list: List[str]
    ) -> List[str]:
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

    @staticmethod
    def _open_with_command(command: str, target: str, foreground: bool = False) -> None:
        """
        Open the given target with the specified command.
        If foreground is True, run in the foreground capturing output.
        Otherwise, run detached with no output.
        """
        if foreground:
            try:
                result = subprocess.run(
                    [command, target],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
            except Exception as e:
                logging.error(f"Failed to open '{target}' with '{command}': {e}")
        else:
            try:
                subprocess.Popen(
                    [command, target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logging.error(f"Failed to open '{target}' with '{command}': {e}")


# -------------------------------------------------------
# "s" Subcommand Configuration
# -------------------------------------------------------
class SConfig(QConfig):
    """
    Configuration for the "s" subcommand.
    Stores aliases under the "s" key in the configuration file.
    """

    @classproperty
    def SUBCOMMAND_KEY(self) -> str:
        return "s"

    def __init__(self, aliases: Optional[List[Tuple[str, List[str]]]] = None) -> None:
        # Each alias is (pattern, list_of_replacements)
        self.aliases = aliases if aliases is not None else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SConfig":
        aliases_data = data.get("aliases", [])
        if not isinstance(aliases_data, list):
            logging.warning(
                "Invalid 'aliases' type in config. Expected list; using empty."
            )
            aliases_data = []
        clean_aliases: List[Tuple[str, List[str]]] = []
        for item in aliases_data:
            # We now expect item to be at least [pattern, ...]
            if isinstance(item, list) and len(item) >= 1:
                pattern = item[0]
                replacements = item[1:]
                if isinstance(pattern, str) and all(
                    isinstance(r, str) for r in replacements
                ):
                    clean_aliases.append((pattern, replacements))
                else:
                    logging.warning(
                        "Alias pattern must be a string and replacements must be strings. Skipping malformed alias."
                    )
            else:
                logging.warning(
                    "Alias entry must be [pattern, replacement(s)...]. Skipping."
                )
        return cls(aliases=clean_aliases)

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Convert (pattern, [r1, r2, ...]) -> [pattern, r1, r2, ...]
            "aliases": [[pattern] + repls for (pattern, repls) in self.aliases],
        }


# -------------------------------------------------------
# "s" Subcommand Implementation
# -------------------------------------------------------
class SSubcommand(QSubcommand):
    """
    Subcommand that manages personal scripts from GitHub or runs them
    with optional aliases and Docker usage.
    """

    SCRIPTS_DIR: Path = Path.home() / ".cache" / "buchwald" / "q" / "scripts"

    def __init__(self) -> None:
        self.config: Optional[SConfig] = None

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "s",
            help=("Run personal scripts with optional aliases and Docker usage."),
        )
        parser.add_argument(
            "-a",
            "--alias",
            nargs=argparse.REMAINDER,
            help="Create an alias. Example: -a ^foo$ /some/replacement -h",
        )
        parser.add_argument(
            "-A",
            "--remove-alias",
            metavar="INDEX",
            help="Remove an existing alias by its numeric index (as shown by --list-aliases).",
        )
        parser.add_argument(
            "-l",
            "--list-aliases",
            action="store_true",
            help="List all aliases for the 's' subcommand with their indexes.",
        )
        parser.add_argument(
            "-i",
            "--install",
            action="store_true",
            help=(
                "Download the scripts tarball from GitHub and extract it to "
                f"{self.SCRIPTS_DIR}. Existing directory is removed first."
            ),
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

        # Handle config updates (aliases)
        config_changed = self._handle_config_updates(args)
        if config_changed:
            self.config.save(config_path)

        # Possibly list aliases
        self._list_aliases_if_requested(args)

        # Possibly list scripts
        if args.list_scripts:
            self._list_scripts()
            return

        # Possibly install scripts
        if args.install:
            logging.info("Installing scripts from GitHub...")
            if not self._install_scripts():
                return

        # Run script if there are arguments
        if args.args:
            self._run_script_with_arguments(args)

    def _handle_config_updates(self, args: argparse.Namespace) -> bool:
        """
        Handle alias additions and removals from CLI arguments.
        Returns True if the configuration was changed, False otherwise.
        """
        config_changed = False

        # Add alias if requested
        if args.alias:
            pattern = args.alias[0]
            replacements = args.alias[1:]
            logging.debug(f"Adding alias pattern '{pattern}' -> {replacements}.")
            self.config.aliases.append((pattern, replacements))
            config_changed = True

        # Remove alias if requested (by index)
        if args.remove_alias is not None:
            try:
                idx = int(args.remove_alias)
                if 0 <= idx < len(self.config.aliases):
                    logging.debug(f"Removing alias at index {idx}.")
                    del self.config.aliases[idx]
                    config_changed = True
                else:
                    logging.warning(f"Index out of range: {idx}")
            except ValueError:
                logging.warning(f"Alias index must be an integer: {args.remove_alias}")

        return config_changed

    def _list_aliases_if_requested(self, args: argparse.Namespace) -> None:
        """
        If --list-aliases is requested, prints the currently configured aliases.
        """
        if args.list_aliases:
            if self.config.aliases:
                logging.info("Listing aliases (index, pattern, replacements):")
                for i, (pat, repls) in enumerate(self.config.aliases):
                    print(f"{i}\t{pat}\t{repls}")
            else:
                logging.info("No aliases are currently configured.")

    def _run_script_with_arguments(self, args: argparse.Namespace) -> None:
        """
        Run a script using the provided command-line arguments. Handles:
         - Checking if scripts directory exists
         - Applying aliases (unless disabled)
         - Determining the actual script vs. docker-like arguments
         - Checking for # Template
         - Determining execution mode
         - Invoking script via Docker, venv, or directly
        """
        # We need a valid scripts directory
        if not self.SCRIPTS_DIR.is_dir():
            logging.error(
                f"The scripts directory '{self.SCRIPTS_DIR}' does not exist. "
                "Please run 'q s --install' first."
            )
            return

        # Apply alias expansion to all arguments (unless disabled)
        all_args = args.args
        all_args = [a for a in all_args if a != "--"]
        if not args.disable_aliases:
            all_args = self._apply_aliases_to_arglist(self.config.aliases, all_args)

        # Parse out docker-like arguments, the script name, and script arguments
        docker_args, script, script_args = self._parse_script_args(all_args)

        if not script:
            logging.error("No valid Python script found in arguments. Aborting.")
            return

        # Determine execution mode
        execution_mode = args.execution_mode
        can_use_docker = shutil.which("docker") is not None
        venv_available = False
        try:
            import venv  # noqa

            venv_available = True
        except ImportError:
            pass

        # Check if the script has a 'Template:' line to decide Docker usage automatically if no mode specified
        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = self.SCRIPTS_DIR / script

        if execution_mode is None:
            if can_use_docker and self._script_has_template(script_path):
                execution_mode = "docker"
            elif venv_available:
                execution_mode = "venv"
            else:
                execution_mode = "direct"

        # Execute according to chosen mode
        if execution_mode == "docker":
            docker_script = self.SCRIPTS_DIR / "Meta" / "docker.py"
            if not docker_script.is_file():
                logging.error(
                    "Cannot find 'docker.py' in the 'Meta' directory. "
                    "Please run 'q s --install' again or check the repository structure."
                )
                return
            self._run_docker_script(docker_script, docker_args, script, script_args)
        elif execution_mode == "venv":
            venver_script = self.SCRIPTS_DIR / "Meta" / "venver.py"
            if not venver_script.is_file():
                logging.error(
                    "Cannot find 'venver.py' in the 'Meta' directory. "
                    "Please run 'q s --install' again or check the repository structure."
                )
                return
            self._run_venver_script(venver_script, docker_args, script, script_args)
        else:  # "direct"
            self._run_local_script(script, docker_args, script_args)

    def _parse_script_args(
        self, all_args: List[str]
    ) -> Tuple[List[str], str, List[str]]:
        """
        Determine the first argument that corresponds to a file (absolute or relative
        to SCRIPTS_DIR) which exists and whose first line contains 'python'. Everything
        before that file is treated as docker-like arguments, and everything after it
        is script arguments.

        Returns:
            (docker_args, script, script_args)
            script will be an empty string if no valid Python script was found.
        """
        docker_args = []
        script = ""
        script_args = []
        script_found = False

        for i, arg in enumerate(all_args):
            # Check if arg is a path (absolute or relative to SCRIPTS_DIR) that exists
            candidate_path = Path(arg)
            if not candidate_path.is_absolute():
                candidate_path = self.SCRIPTS_DIR / arg

            if candidate_path.is_file():
                # Read the first line to see if it contains 'python'
                try:
                    with candidate_path.open("r", encoding="utf-8") as f:
                        first_line = f.readline()
                    if "python" in first_line:
                        script = arg
                        script_args = all_args[i + 1 :]
                        script_found = True
                        break
                    else:
                        docker_args.append(arg)
                except Exception:
                    docker_args.append(arg)
            else:
                docker_args.append(arg)

        if not script_found:
            # If we never found a valid Python script, just return
            return (all_args, "", [])

        return (docker_args, script, script_args)

    def _run_docker_script(
        self,
        docker_script: Path,
        docker_args: List[str],
        script: str,
        script_args: List[str],
    ) -> None:
        """
        Run docker.py with the combined arguments:
          docker.py [docker_args] [script] [script_args]
        """
        try:
            os.chdir(self.SCRIPTS_DIR)
        except Exception as e:
            logging.error(f"Failed to change directory to '{self.SCRIPTS_DIR}': {e}")
            return

        cmd = [str(docker_script)] + docker_args + [script] + script_args
        logging.info(f"Running Docker script command: {cmd}")
        try:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logging.error(
                    f"'docker.py' exited with return code {result.returncode}"
                )
        except Exception as e:
            logging.error(f"Error running '{docker_script.name}': {e}")

    def _run_venver_script(
        self,
        venver_script: Path,
        venver_args: List[str],
        script: str,
        script_args: List[str],
    ) -> None:
        """
        Run venver.py with the combined arguments:
          venver.py [venver_args] [script] [script_args]
        """
        try:
            os.chdir(self.SCRIPTS_DIR)
        except Exception as e:
            logging.error(f"Failed to change directory to '{self.SCRIPTS_DIR}': {e}")
            return

        cmd = [str(venver_script)] + venver_args + [script] + script_args
        logging.info(f"Running Venver script command: {cmd}")
        try:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logging.error(
                    f"'venver.py' exited with return code {result.returncode}"
                )
        except Exception as e:
            logging.error(f"Error running '{venver_script.name}': {e}")

    def _run_local_script(
        self,
        script: str,
        docker_args: List[str],
        script_args: List[str],
    ) -> None:
        """
        Run a script locally (without docker.py or venver.py). We still change to SCRIPTS_DIR,
        then execute the script with all arguments appended.
        """
        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = self.SCRIPTS_DIR / script

        if not script_path.is_file():
            logging.error(f"Script file '{script_path}' not found.")
            return

        try:
            os.chdir(self.SCRIPTS_DIR)
        except Exception as e:
            logging.error(f"Failed to change directory to '{self.SCRIPTS_DIR}': {e}")
            return

        cmd = [script] + docker_args + script_args
        logging.info(f"Running local script command: {cmd}")
        try:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logging.error(
                    f"Local script exited with return code {result.returncode}"
                )
        except Exception as e:
            logging.error(f"Error running local script '{script}': {e}")

    def _install_scripts(self) -> bool:
        """
        Download and extract the personal scripts from GitHub into SCRIPTS_DIR.
        Overwrite any existing directory. Return True if succeeded, False otherwise.
        """
        url = "https://github.com/zit-hb/Personal-Scripts/archive/refs/heads/master.tar.gz"

        # Remove existing directory if it exists
        if self.SCRIPTS_DIR.exists():
            logging.debug(f"Removing existing scripts directory: {self.SCRIPTS_DIR}")
            try:
                shutil.rmtree(self.SCRIPTS_DIR)
            except Exception as e:
                logging.error(f"Failed to remove existing scripts directory: {e}")
                return False

        # Attempt to create the scripts directory
        try:
            self.SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logging.error(
                f"Permission error while creating scripts directory '{self.SCRIPTS_DIR}': {e}"
            )
            return False
        except Exception as e:
            logging.error(
                f"Could not create scripts directory '{self.SCRIPTS_DIR}': {e}"
            )
            return False

        # Create a temporary file for the downloaded tarball
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".tar.gz")
            os.close(temp_fd)
        except Exception as e:
            logging.error(f"Failed to create a temporary file for download: {e}")
            return False

        # Download the tarball
        logging.debug(f"Downloading scripts from {url} to {temp_path}")
        try:
            with (
                urllib.request.urlopen(url) as response,
                open(temp_path, "wb") as out_file,
            ):
                out_file.write(response.read())
        except Exception as e:
            logging.error(f"Failed to download scripts from {url}: {e}")
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return False

        # Extract the tarball
        try:
            logging.debug(f"Extracting tarball {temp_path} to {self.SCRIPTS_DIR}")
            with tarfile.open(temp_path, "r:gz") as tar:
                tar.extractall(self.SCRIPTS_DIR)

            # The tar archive creates "Personal-Scripts-master"
            top_level_dir = self.SCRIPTS_DIR / "Personal-Scripts-master"
            if top_level_dir.is_dir():
                for item in top_level_dir.iterdir():
                    shutil.move(str(item), str(self.SCRIPTS_DIR))
                shutil.rmtree(top_level_dir)
        except Exception as e:
            logging.error(f"Failed to extract scripts: {e}")
            return False
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        print("Scripts installed successfully.")
        return True

    @staticmethod
    def _apply_aliases_to_arglist(
        aliases: List[Tuple[str, List[str]]], args_list: List[str]
    ) -> List[str]:
        """
        For each argument in args_list, if it fully matches the pattern of
        an alias, replace that single argument with the list of replacements.
        Returns the new list of arguments after all expansions.
        """
        result = []
        for arg in args_list:
            replaced = False
            for pattern, replacements in aliases:
                if re.fullmatch(pattern, arg):
                    result.extend(replacements)
                    replaced = True
                    break
            if not replaced:
                result.append(arg)
        return result

    @staticmethod
    def _script_has_template(script_path: Path) -> bool:
        """
        Return True if the script at script_path contains a line with '# Template:'.
        Return False otherwise (or if the file cannot be read).
        """
        if not script_path.is_file():
            return False
        try:
            with script_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if "# Template:" in line:
                        return True
        except Exception as e:
            logging.debug(f"Failed to read script '{script_path}': {e}")
        return False

    def _list_scripts(self) -> None:
        """
        List all available Python scripts in SCRIPTS_DIR (except for anything in 'Meta').
        in a tree-like structure.
        """
        if not self.SCRIPTS_DIR.is_dir():
            logging.error(
                f"The scripts directory '{self.SCRIPTS_DIR}' does not exist. "
                "Please run 'q s --install' first."
            )
            return

        print(f"Available scripts in '{self.SCRIPTS_DIR}':")
        if not self._directory_has_python_files_excluding_meta(self.SCRIPTS_DIR):
            print("  No Python scripts found.")
            return

        self._print_scripts_tree(self.SCRIPTS_DIR, is_root=True)

    def _directory_has_python_files_excluding_meta(self, dirpath: Path) -> bool:
        """
        Return True if there is any .py file (directly or in subdirectories) below dirpath,
        excluding anything under 'Meta'.
        """
        for root, dirs, files in os.walk(dirpath):
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
        if is_root:
            # For the root, don't print its name, just print its children
            entries = self._list_relevant_entries(directory)
        else:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + directory.name + "/")
            prefix += "    " if is_last else "│   "
            entries = self._list_relevant_entries(directory)

        for i, entry in enumerate(entries):
            is_entry_last = i == len(entries) - 1
            if entry.is_dir():
                self._print_scripts_tree(entry, prefix, is_entry_last, is_root=False)
            else:
                # It's a file
                connector = "└── " if is_entry_last else "├── "
                print(prefix + connector + entry.name)

    def _list_relevant_entries(self, dirpath: Path) -> List[Path]:
        """
        Return a sorted list of relevant subdirectories (those that contain .py files somewhere inside,
        excluding 'Meta') plus .py files in this directory (also ignoring 'Meta').
        """
        entries = []
        dirs_in_dir = []
        files_in_dir = []

        for item in sorted(dirpath.iterdir()):
            if item.is_dir():
                if item.name == "Meta":
                    continue
                if self._directory_has_python_files_excluding_meta(item):
                    dirs_in_dir.append(item)
            else:
                if item.suffix == ".py":
                    files_in_dir.append(item)

        entries.extend(dirs_in_dir)
        entries.extend(files_in_dir)
        return entries


# -------------------------------------------------------
# "u" Subcommand Implementation
# -------------------------------------------------------
class USubcommand(QSubcommand):
    """
    Subcommand that auto-updates this script from a remote URL.
    Compares sha512 hashes of the remote and local scripts.
    If they differ, optionally replaces the local script with the remote version.
    """

    UPDATE_URL = (
        "https://raw.githubusercontent.com/"
        "zit-hb/Personal-Scripts/refs/heads/master/Meta/q.py"
    )

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "u",
            help=("Auto-update this script from the remote repository."),
        )
        parser.add_argument(
            "-c",
            "--only-check",
            action="store_true",
            help="Only check for a new version (compare sha512), do not replace the local script.",
        )
        parser.set_defaults(subcommand_obj=self)

    def run(self, args: argparse.Namespace) -> None:
        """
        Perform the update logic: compare local and remote sha512;
        if different and not only-check, replace the local script.
        """
        local_path = Path(__file__).resolve()
        logging.debug(f"Local script path: {local_path}")

        remote_data = self._download_remote_script()
        if remote_data is None:
            logging.error("Failed to download the remote script.")
            return

        local_hash = self._calculate_sha512_file(local_path)
        remote_hash = self._calculate_sha512_data(remote_data)
        logging.debug(f"Local sha512:  {local_hash}")
        logging.debug(f"Remote sha512: {remote_hash}")

        self._compare_and_update_if_needed(
            args, local_path, remote_data, local_hash, remote_hash
        )

    def _compare_and_update_if_needed(
        self,
        args: argparse.Namespace,
        local_path: Path,
        remote_data: bytes,
        local_hash: str,
        remote_hash: str,
    ) -> None:
        """
        Compare the local and remote hashes. If they differ, handle update
        logic depending on the --only-check argument.
        """
        if local_hash == remote_hash:
            print("Your script is already up to date.")
        else:
            print("A new version is available.")
            if args.only_check:
                return
            if self._update_local_script(local_path, remote_data):
                print("Script updated successfully.")
            else:
                logging.error("Failed to update the local script.")

    @staticmethod
    def _download_remote_script() -> Optional[bytes]:
        """
        Download the remote script and return its bytes, or None on error.
        """
        try:
            with urllib.request.urlopen(USubcommand.UPDATE_URL) as response:
                return response.read()
        except Exception as e:
            logging.error(f"Error downloading remote script: {e}")
            return None

    @staticmethod
    def _calculate_sha512_data(data: bytes) -> str:
        """
        Return the sha512 hex digest for the given data.
        """
        sha = hashlib.sha512()
        sha.update(data)
        return sha.hexdigest()

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
        except Exception as e:
            logging.error(f"Error reading local script '{path}': {e}")
            return ""
        return sha.hexdigest()

    @staticmethod
    def _update_local_script(local_path: Path, new_data: bytes) -> bool:
        """
        Replace the local script at local_path with new_data. Return True if successful, False otherwise.
        """
        try:
            with local_path.open("wb") as f:
                f.write(new_data)
        except Exception as e:
            logging.error(f"Error writing updated script to '{local_path}': {e}")
            return False

        return True


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
