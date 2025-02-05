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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self, command: Optional[str] = None, aliases: Optional[Dict[str, str]] = None
    ) -> None:
        self.command = command
        self.aliases = aliases if aliases is not None else {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OConfig":
        command = data.get("command", None)
        aliases = data.get("aliases", {})
        if not isinstance(aliases, dict):
            logging.warning(
                "Invalid 'aliases' type in config. Expected dict; using empty."
            )
            aliases = {}
        return cls(command=command, aliases=aliases)

    def to_dict(self) -> Dict[str, Any]:
        return {"command": self.command, "aliases": self.aliases}


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
    Subcommand to open a file/path/URL (or alias) using a configured command.
    Defaults to "xdg-open" if no command is explicitly set.
    Supports aliases, plus setting or clearing the command.
    By default, the command is executed detached with no output.
    Use the '--foreground' (or '-F') option to run in foreground and print output.
    """

    def __init__(self) -> None:
        self.config: Optional[OConfig] = None

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "o", help="Open a file/path/URL (or alias) using a configured command."
        )
        parser.add_argument(
            "file_or_alias",
            nargs="?",
            help="File/path/URL or alias to open. If omitted, no file is opened.",
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
            nargs=2,
            metavar=("ALIAS", "PATH"),
            help="Create or update an alias. Example: -a google https://google.com",
        )
        parser.add_argument(
            "-A",
            "--remove-alias",
            metavar="ALIAS",
            help="Remove an existing alias by name.",
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List all aliases for the 'o' subcommand.",
        )
        parser.add_argument(
            "-F",
            "--foreground",
            action="store_true",
            help="Run the open command in the foreground and print its output.",
        )

        parser.set_defaults(subcommand_obj=self)

    def run(self, args: argparse.Namespace) -> None:
        config_path = Path(args.config)
        self.config = OConfig.load(config_path)
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

        # Add/update alias if requested
        if args.alias:
            alias_key, alias_path = args.alias
            logging.debug(f"Setting alias '{alias_key}' to '{alias_path}'.")
            self.config.aliases[alias_key] = alias_path
            config_changed = True

        # Remove alias if requested
        if args.remove_alias:
            if args.remove_alias in self.config.aliases:
                logging.debug(f"Removing alias '{args.remove_alias}'.")
                del self.config.aliases[args.remove_alias]
                config_changed = True
            else:
                logging.warning(
                    f"Alias '{args.remove_alias}' not found in configuration."
                )

        # Save config if any changes were made
        if config_changed:
            self.config.save(config_path)

        # List aliases if requested
        if args.list:
            if self.config.aliases:
                logging.info("Listing aliases:")
                for key, path_val in sorted(self.config.aliases.items()):
                    print(f"{key}\t{path_val}")
            else:
                logging.info("No aliases are currently configured.")

        # Open the file/path/alias if given
        if args.file_or_alias:
            resolved_path = self.config.aliases.get(
                args.file_or_alias, args.file_or_alias
            )
            cmd = self.config.command if self.config.command is not None else "xdg-open"
            logging.info(f"Opening '{resolved_path}' using '{cmd}'")
            self._open_with_command(cmd, resolved_path, args.foreground)
        else:
            logging.info("No path specified, not opening anything")

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

    def __init__(self, aliases: Optional[Dict[str, str]] = None) -> None:
        self.aliases = aliases if aliases is not None else {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SConfig":
        aliases = data.get("aliases", {})
        if not isinstance(aliases, dict):
            logging.warning(
                "Invalid 'aliases' type in config. Expected dict; using empty."
            )
            aliases = {}
        return cls(aliases=aliases)

    def to_dict(self) -> Dict[str, Any]:
        return {"aliases": self.aliases}


# -------------------------------------------------------
# "s" Subcommand Implementation
# -------------------------------------------------------
class SSubcommand(QSubcommand):
    """
    Subcommand that manages personal scripts from GitHub or runs them.

    --install/-i:
      - Downloads https://github.com/zit-hb/Personal-Scripts/archive/refs/heads/master.tar.gz
      - Unpacks it to ~/.cache/buchwald/q/scripts/. If that directory already exists, it is removed first.

    If one or more unnamed arguments are provided (and --install is not given),
    the subcommand:
      - Checks if ~/.cache/buchwald/q/scripts exists. If not, error out.
      - Changes to that directory
      - Splits out Docker arguments (leading '-' or '--'), then the script name (alias resolved),
        then script-specific arguments. If '--' is present, everything after it is treated
        as script arguments, ignoring whether they begin with '-'.
      - If Docker is available and not disabled, runs docker.py with the parsed arguments.
      - Otherwise, runs the script directly.
    """

    SCRIPTS_DIR: Path = Path.home() / ".cache" / "buchwald" / "q" / "scripts"

    def __init__(self) -> None:
        self.config: Optional[SConfig] = None

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "s",
            help=(
                "Run personal scripts with optional aliases and additional arguments."
            ),
        )
        parser.add_argument(
            "-a",
            "--alias",
            nargs=2,
            metavar=("ALIAS", "RESOLVED_VALUE"),
            help="Create or update an alias. Example: -a foobar /some/thing",
        )
        parser.add_argument(
            "-A",
            "--remove-alias",
            metavar="ALIAS",
            help="Remove an existing alias by name.",
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List all aliases for the 's' subcommand.",
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
            "-D",
            "--disable-docker",
            action="store_true",
            help="Disable Docker usage and run the script directly without docker.py.",
        )
        parser.add_argument(
            "args",
            nargs="*",
            help=(
                "Optional arguments. Leading '-' or '--' arguments are considered Docker args "
                "unless Docker is disabled. The first non-dash argument is treated as "
                "the script name (or alias). The remainder are passed as script arguments. "
                "If you include a '--' anywhere, everything after it goes to the script arguments."
            ),
        )
        parser.set_defaults(subcommand_obj=self)

    def run(self, args: argparse.Namespace) -> None:
        config_path = Path(args.config)
        self.config = SConfig.load(config_path)
        config_changed = False

        # Add/update alias if requested
        if args.alias:
            alias_key, alias_value = args.alias
            logging.debug(f"Setting alias '{alias_key}' to '{alias_value}'.")
            self.config.aliases[alias_key] = alias_value
            config_changed = True

        # Remove alias if requested
        if args.remove_alias:
            if args.remove_alias in self.config.aliases:
                logging.debug(f"Removing alias '{args.remove_alias}'.")
                del self.config.aliases[args.remove_alias]
                config_changed = True
            else:
                logging.warning(
                    f"Alias '{args.remove_alias}' not found in configuration."
                )

        # Save config if any changes were made
        if config_changed:
            self.config.save(config_path)

        # List aliases if requested
        if args.list:
            if self.config.aliases:
                logging.info("Listing aliases:")
                for key, val in sorted(self.config.aliases.items()):
                    print(f"{key}\t{val}")
            else:
                logging.info("No aliases are currently configured.")

        # Handle install
        if args.install:
            logging.info("Installing scripts from GitHub...")
            if not self._install_scripts():
                # If installation failed, bail out
                return

        # If there are arguments, handle running scripts
        if args.args:
            # We need a valid scripts directory
            if not self.SCRIPTS_DIR.is_dir():
                logging.error(
                    f"The scripts directory '{self.SCRIPTS_DIR}' does not exist. "
                    "Please run 'q s --install' first."
                )
                return

            # Parse out docker-like arguments, the script name (alias), and script arguments
            docker_args, script, script_args = self._parse_script_args(args.args)

            if not script:
                logging.error("No script specified. Provide a script name or alias.")
                return

            # If we're using Docker, we need docker.py
            can_use_docker = (not args.disable_docker) and (
                shutil.which("docker") is not None
            )

            if can_use_docker:
                docker_script = self.SCRIPTS_DIR / "docker.py"
                if not docker_script.is_file():
                    logging.error(
                        "Cannot find 'docker.py' in the scripts directory. "
                        "Please run 'q s --install' again or check the repository structure."
                    )
                    return
                self._run_docker_script(docker_script, docker_args, script, script_args)
            else:
                self._run_local_script(script, docker_args, script_args)

    def _parse_script_args(self, all_args: List[str]) -> (List[str], str, List[str]):
        """
        Split the provided arguments into:
          - docker_args (leading items, including any immediate value if it directly follows a dash-arg)
          - script (the first true 'command' after the above, or from after '--' if present)
          - script_args (everything else, or everything after the script name or a '--').

        Example usage:
          q s -p 8080 -v myscript arg1 arg2
          => docker_args = ['-p', '8080', '-v'], script = 'myscript', script_args = ['arg1', 'arg2']

          q s -p 8080 -v -- myscript -x -y
          => docker_args = ['-p', '8080', '-v'], script = 'myscript', script_args = ['-x', '-y']
        """
        docker_args = []
        script_arg = ""
        script_args = []

        # Check if there's a '--' marker
        double_dash_index = None
        if "--" in all_args:
            double_dash_index = all_args.index("--")

        # Split into parts: the portion before '--' (docker_part) and after '--' (post_dd)
        if double_dash_index is not None:
            docker_part = all_args[:double_dash_index]
            post_dd = all_args[double_dash_index + 1 :]
        else:
            docker_part = all_args
            post_dd = []

        i = 0
        # Collect "docker-like" args, pairing a possible immediate non-dash value
        while i < len(docker_part):
            if docker_part[i].startswith("-"):
                docker_args.append(docker_part[i])
                i += 1
                # If next item doesn't start with '-', treat it as a parameter to the dash arg
                if i < len(docker_part) and not docker_part[i].startswith("-"):
                    docker_args.append(docker_part[i])
                    i += 1
            else:
                # The first non-dash in docker_part is the script name
                script_arg = docker_part[i]
                i += 1
                # If we have no '--', the remainder of docker_part becomes script_args
                if double_dash_index is None:
                    script_args = docker_part[i:]
                break

        # If we never found a script in docker_part, then the first arg after '--' is script
        if not script_arg and post_dd:
            script_arg = post_dd[0]
            script_args = post_dd[1:]
        else:
            # If we found a script or no post_dd, we still append any post_dd to script_args
            script_args += post_dd

        # Resolve alias
        script_arg = self.config.aliases.get(script_arg, script_arg)

        return docker_args, script_arg, script_args

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

    def _run_local_script(
        self,
        script: str,
        docker_args: List[str],
        script_args: List[str],
    ) -> None:
        """
        Run a script locally (without docker.py). We still change to SCRIPTS_DIR,
        then execute the script with all arguments appended.
        """
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

        # Attempt to create the scripts directory (with parents)
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

        logging.info("Scripts installed successfully.")
        return True


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
        "zit-hb/Personal-Scripts/refs/heads/master/Tools/q.py"
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

        # Download remote script data
        remote_data = self._download_remote_script()
        if remote_data is None:
            logging.error("Failed to download the remote script.")
            return

        # Calculate hashes
        local_hash = self._calculate_sha512_file(local_path)
        remote_hash = self._calculate_sha512_data(remote_data)

        logging.debug(f"Local sha512:  {local_hash}")
        logging.debug(f"Remote sha512: {remote_hash}")

        # Compare
        if local_hash == remote_hash:
            print("Your script is already up to date.")
        else:
            print("A new version is available.")
            if args.only_check:
                return

            # Perform the update
            if not self._update_local_script(local_path, remote_data):
                logging.error("Failed to update the local script.")
            else:
                print("Script updated successfully.")

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
            # Write the new script data to the same file
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
