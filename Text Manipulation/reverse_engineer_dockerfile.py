#!/usr/bin/env python3

# -------------------------------------------------------
# Script: reverse_engineer_dockerfile.py
#
# Description:
# Turns Docker images into Dockerfiles by analyzing the image
# metadata and layer history, then using an LLM to generate
# a coherent Dockerfile that could reproduce the image.
#
# Usage:
#   ./reverse_engineer_dockerfile.py [options] IMAGE_REFERENCE
#
# Arguments:
#   - IMAGE_REFERENCE: Docker image reference (name:tag, ID, or digest)
#
# Options:
#   -o, --output OUTPUT         Output file path for the Dockerfile (default: stdout)
#   -p, --provider PROVIDER     LLM provider to use: openai or anthropic (default: openai)
#   -m, --model MODEL           Model to use (default: auto → provider default)
#   -k, --api-key API_KEY       Your API key (or set via OPENAI_API_KEY/ANTHROPIC_API_KEY env var)
#   -l, --local                 Use local image instead of remote registry
#   -b, --base-images           Include likely base images in the output
#   -d, --detailed              Include detailed layer analysis in prompt to LLM
#   -i, --include-instructions  Include helper instructions in the Dockerfile as comments
#   -s, --skip-llm              Skip LLM and output only raw analysis
#   -v, --verbose               Enable verbose logging (INFO level)
#   -vv, --debug                Enable debug logging (DEBUG level)
#
# Template: ubuntu24.04
#
# Requirements:
#   - docker (command-line tool must be installed)
#   - openai (install via: pip install openai==1.64.0)
#   - anthropic (install via: pip install anthropic==0.48.0)
#   - rich (install via: pip install rich==13.9.4)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.syntax import Syntax

console = Console()

PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-5",
    "anthropic": "claude-opus-4-1-20250805",
}


def resolve_model(provider: str, model: Optional[str]) -> str:
    """
    Resolves the effective model name based on provider and user input.

    Rules:
      - If model is None, empty, or 'auto' (case-insensitive), return the provider's default.
      - Otherwise, return the provided model unchanged.
    """
    if not model or model.strip().lower() == "auto":
        return PROVIDER_DEFAULT_MODELS.get(provider.lower(), "gpt-5")
    return model


@dataclass
class Message:
    """
    A single message within a conversation.
    """

    role: str
    content: str


@dataclass
class Conversation:
    """
    A conversation is a sequence of messages for the LLM.
    """

    messages: List[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """
        Appends a new Message to the conversation.
        """
        self.messages.append(Message(role=role, content=content))


class IProvider:
    """
    Provider interface to send a conversation to an LLM and get a response.
    """

    def generate_response(self, conversation: Conversation, model: str) -> str:
        """
        Sends the conversation to the model and returns the model's response text.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class OpenAIProvider(IProvider):
    """
    Implementation of IProvider that uses the OpenAI client.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initializes the OpenAI provider with the given API key.
        """
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, conversation: Conversation, model: str) -> str:
        """
        Sends the conversation to the specified OpenAI model.
        """
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in conversation.messages
        ]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error during OpenAI call: {e}")
            return "Error: could not retrieve a response from the model."


class AnthropicProvider(IProvider):
    """
    Implementation of IProvider that uses the Anthropic client.
    """

    def __init__(self, api_key: str, max_tokens: int = 8192) -> None:
        """
        Initializes the Anthropic provider with the given API key.
        """
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def generate_response(self, conversation: Conversation, model: str) -> str:
        """
        Sends the conversation to the specified Anthropic model.
        """
        anthropic_messages = [
            {"role": msg.role, "content": msg.content} for msg in conversation.messages
        ]

        try:
            response = self.client.messages.create(
                model=model,
                messages=anthropic_messages,
                max_tokens=self.max_tokens,
            )
            full_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    full_text += block.text
            return full_text.strip()
        except Exception as e:
            logging.error(f"Error during Anthropic call: {e}")
            return "Error: could not retrieve a response from the model."


class DockerImageAnalyzer:
    """
    Analyzes Docker images to extract information needed for Dockerfile generation.
    """

    def __init__(
        self, image_ref: str, local: bool = False, detailed: bool = False
    ) -> None:
        """
        Initializes the analyzer with an image reference.
        """
        self.image_ref = image_ref
        self.local = local
        self.detailed = detailed
        self.inspection_data = None
        self.history_data = None

    def analyze(self) -> Dict[str, Union[Dict, List, str]]:
        """
        Performs full analysis of the Docker image.
        """
        self.inspect_image()
        self.get_history()

        analysis = {
            "image_ref": self.image_ref,
            "config": self._extract_config(),
            "layers": self._extract_layers(),
            "env_vars": self._extract_env_vars(),
            "volumes": self._extract_volumes(),
            "labels": self._extract_labels(),
            "exposed_ports": self._extract_exposed_ports(),
            "entrypoint": self._extract_entrypoint(),
            "cmd": self._extract_cmd(),
            "user": self._extract_user(),
            "workdir": self._extract_workdir(),
            "base_image_candidates": self._find_base_image_candidates(),
        }

        return analysis

    def inspect_image(self) -> None:
        """
        Runs docker inspect on the image.
        """
        cmd = ["docker", "inspect", self.image_ref]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.inspection_data = json.loads(result.stdout)[0]
            logging.info(f"Successfully inspected image {self.image_ref}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to inspect image: {e}")
            if not self.local:
                logging.error(
                    "If using a remote image, ensure you have permission to access it"
                )
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error("Failed to parse docker inspect output")
            sys.exit(1)

    def get_history(self) -> None:
        """
        Gets the history of the image to analyze layer commands.
        """
        cmd = [
            "docker",
            "history",
            "--no-trunc",
            "--format",
            "{{json .}}",
            self.image_ref,
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Split output into lines and parse each as JSON
            self.history_data = [
                json.loads(line) for line in result.stdout.strip().split("\n")
            ]
            logging.info(f"Successfully retrieved history for image {self.image_ref}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to get image history: {e}")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error("Failed to parse docker history output")
            sys.exit(1)

    def _extract_config(self) -> Dict:
        """
        Extracts the image configuration.
        """
        if not self.inspection_data:
            return {}
        return self.inspection_data.get("Config", {})

    def _extract_layers(self) -> List[Dict]:
        """
        Extracts layer information from history.
        """
        if not self.history_data:
            return []

        # Filter out empty layers if not in detailed mode
        if not self.detailed:
            return [
                {
                    "created_by": layer.get("CreatedBy", ""),
                    "size": layer.get("Size", 0),
                    "comment": layer.get("Comment", ""),
                }
                for layer in self.history_data
                if layer.get("CreatedBy")
                and not layer.get("CreatedBy").startswith("#(nop)")
            ]

        return [
            {
                "created_by": layer.get("CreatedBy", ""),
                "size": layer.get("Size", 0),
                "comment": layer.get("Comment", ""),
            }
            for layer in self.history_data
        ]

    def _extract_env_vars(self) -> Dict[str, str]:
        """
        Extracts environment variables.
        """
        if not self.inspection_data:
            return {}

        env_vars = {}
        for env_string in self.inspection_data.get("Config", {}).get("Env", []):
            if "=" in env_string:
                key, value = env_string.split("=", 1)
                env_vars[key] = value
        return env_vars

    def _extract_volumes(self) -> List[str]:
        """
        Extracts configured volumes.
        """
        if not self.inspection_data:
            return []

        volumes = self.inspection_data.get("Config", {}).get("Volumes", {})
        return list(volumes.keys()) if volumes else []

    def _extract_labels(self) -> Dict[str, str]:
        """
        Extracts image labels.
        """
        if not self.inspection_data:
            return {}

        return self.inspection_data.get("Config", {}).get("Labels", {}) or {}

    def _extract_exposed_ports(self) -> List[str]:
        """
        Extracts exposed ports.
        """
        if not self.inspection_data:
            return []

        exposed_ports = self.inspection_data.get("Config", {}).get("ExposedPorts", {})
        return list(exposed_ports.keys()) if exposed_ports else []

    def _extract_entrypoint(self) -> List[str]:
        """
        Extracts the entrypoint.
        """
        if not self.inspection_data:
            return []

        return self.inspection_data.get("Config", {}).get("Entrypoint", [])

    def _extract_cmd(self) -> List[str]:
        """
        Extracts the default command.
        """
        if not self.inspection_data:
            return []

        return self.inspection_data.get("Config", {}).get("Cmd", [])

    def _extract_user(self) -> str:
        """
        Extracts the user.
        """
        if not self.inspection_data:
            return ""

        return self.inspection_data.get("Config", {}).get("User", "")

    def _extract_workdir(self) -> str:
        """
        Extracts the working directory.
        """
        if not self.inspection_data:
            return ""

        return self.inspection_data.get("Config", {}).get("WorkingDir", "")

    def _find_base_image_candidates(self) -> List[str]:
        """
        Attempts to identify likely base images.
        """
        if not self.history_data:
            return []

        # Base images are often the first layer with FROM commands
        base_candidates = []
        for layer in reversed(self.history_data):
            cmd = layer.get("CreatedBy", "")
            if "/bin/sh -c #(nop) ADD file:" in cmd or "FROM " in cmd:
                # This is a potential base image indicator
                # Extract potential image names
                if "FROM " in cmd:
                    parts = cmd.split("FROM ")[1].split()
                    if parts:
                        base_candidates.append(parts[0])

        # Add official base images if they appear in any commands
        common_bases = [
            "alpine",
            "ubuntu",
            "debian",
            "centos",
            "fedora",
            "python",
            "node",
            "php",
            "nginx",
        ]
        for layer in self.history_data:
            cmd = layer.get("CreatedBy", "").lower()
            for base in common_bases:
                if f" {base}:" in cmd:
                    parts = cmd.split(f" {base}:")[1].split()
                    if parts:
                        base_candidates.append(f"{base}:{parts[0]}")

        return list(set(base_candidates))  # Remove duplicates


def create_prompt(analysis: Dict, include_instructions: bool) -> str:
    """
    Creates a detailed prompt for the LLM to generate a Dockerfile.
    """
    prompt = "# Dockerfile Generation Task\n\n"
    prompt += "I need you to create a Dockerfile that could be used to reproduce this Docker image.\n"
    prompt += (
        "Below is the analysis of the Docker image's configuration and history.\n\n"
    )

    # Image information
    prompt += f"## Image Reference\n`{analysis['image_ref']}`\n\n"

    # Base image candidates (if available)
    if analysis["base_image_candidates"]:
        prompt += "## Possible Base Images\n"
        for base in analysis["base_image_candidates"]:
            prompt += f"- `{base}`\n"
        prompt += "\n"

    # Commands from layers
    prompt += "## Layer History (commands used to build the image)\n"
    for layer in analysis["layers"]:
        cmd = layer["created_by"]
        if cmd:
            if cmd.startswith("/bin/sh -c #(nop)"):
                # This is a Dockerfile instruction
                cmd = cmd.replace("/bin/sh -c #(nop) ", "")
            elif cmd.startswith("/bin/sh -c"):
                # This is a RUN command
                cmd = "RUN " + cmd.replace("/bin/sh -c ", "")
            prompt += f"- `{cmd}`\n"
    prompt += "\n"

    # Environment variables
    if analysis["env_vars"]:
        prompt += "## Environment Variables\n"
        for key, value in analysis["env_vars"].items():
            prompt += f"- `{key}={value}`\n"
        prompt += "\n"

    # Exposed ports
    if analysis["exposed_ports"]:
        prompt += "## Exposed Ports\n"
        for port in analysis["exposed_ports"]:
            prompt += f"- `{port}`\n"
        prompt += "\n"

    # Volumes
    if analysis["volumes"]:
        prompt += "## Volumes\n"
        for volume in analysis["volumes"]:
            prompt += f"- `{volume}`\n"
        prompt += "\n"

    # Entrypoint and CMD
    if analysis["entrypoint"]:
        prompt += f"## Entrypoint\n`{analysis['entrypoint']}`\n\n"

    if analysis["cmd"]:
        prompt += f"## Command\n`{analysis['cmd']}`\n\n"

    # User and workdir
    if analysis["user"]:
        prompt += f"## User\n`{analysis['user']}`\n\n"

    if analysis["workdir"]:
        prompt += f"## Working Directory\n`{analysis['workdir']}`\n\n"

    # Labels
    if analysis["labels"]:
        prompt += "## Labels\n"
        for key, value in analysis["labels"].items():
            prompt += f"- `{key}={value}`\n"
        prompt += "\n"

    # Instructions
    prompt += "## Requirements\n"
    prompt += "1. Create a valid Dockerfile that reproduces the functionality of this image.\n"
    prompt += "2. Use standard Dockerfile instructions (FROM, RUN, COPY, etc.).\n"
    prompt += "3. Organize commands logically and efficiently.\n"
    prompt += (
        "4. Include all necessary environment variables, exposed ports, volumes, etc.\n"
    )

    if include_instructions:
        prompt += "5. Include helpful comments explaining what each section does.\n"

    prompt += "\nPlease output ONLY the Dockerfile content - no markdown formatting or explanations.\n"

    return prompt


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Turns Docker images into Dockerfiles by analyzing the image and using an LLM."
    )

    parser.add_argument(
        "image_reference",
        type=str,
        help="Docker image reference (name:tag, ID, or digest)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path for the Dockerfile (default: stdout)",
    )

    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use: openai or anthropic (default: openai)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="auto",
        help="Model to use (default: auto → provider default)",
    )

    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="Your API key (or set via OPENAI_API_KEY/ANTHROPIC_API_KEY env var)",
    )

    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="Use local image instead of remote registry",
    )

    parser.add_argument(
        "-b",
        "--base-images",
        action="store_true",
        help="Include likely base images in the output",
    )

    parser.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help="Include detailed layer analysis in prompt to LLM",
    )

    parser.add_argument(
        "-i",
        "--include-instructions",
        action="store_true",
        help="Include helper instructions in the Dockerfile as comments",
    )

    parser.add_argument(
        "-s",
        "--skip-llm",
        action="store_true",
        help="Skip LLM and output only raw analysis",
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


def get_api_key(provided_key: Optional[str], provider: str) -> str:
    """
    Returns the appropriate API key for the chosen provider.
    """
    if provider.lower() == "anthropic":
        return provided_key or os.getenv("ANTHROPIC_API_KEY") or ""
    elif provider.lower() == "openai":
        return provided_key or os.getenv("OPENAI_API_KEY") or ""
    else:
        raise ValueError(f"Invalid provider: {provider}")


def print_syntax(content: str, syntax_type: str = "dockerfile") -> None:
    """
    Prints content with syntax highlighting.
    """
    syntax = Syntax(content, syntax_type, theme="monokai", line_numbers=True)
    console.print(syntax)


def output_raw_analysis(analysis: Dict) -> None:
    """
    Outputs raw analysis information in a readable format.
    """
    # Format the analysis data as a string
    output = "# Docker Image Analysis\n\n"

    # Image reference
    output += f"## Image: {analysis['image_ref']}\n\n"

    # Base image candidates
    if analysis["base_image_candidates"]:
        output += "## Possible Base Images\n"
        for base in analysis["base_image_candidates"]:
            output += f"- {base}\n"
        output += "\n"

    # Layer commands
    output += "## Layer Commands\n"
    for i, layer in enumerate(analysis["layers"]):
        output += f"{i + 1}. {layer['created_by']}\n"
    output += "\n"

    # Environment variables
    if analysis["env_vars"]:
        output += "## Environment Variables\n"
        for key, value in analysis["env_vars"].items():
            output += f"- {key}={value}\n"
        output += "\n"

    # Ports, volumes, entrypoint, cmd
    if analysis["exposed_ports"]:
        output += "## Exposed Ports\n"
        for port in analysis["exposed_ports"]:
            output += f"- {port}\n"
        output += "\n"

    if analysis["volumes"]:
        output += "## Volumes\n"
        for volume in analysis["volumes"]:
            output += f"- {volume}\n"
        output += "\n"

    if analysis["entrypoint"]:
        output += "## Entrypoint\n"
        output += f"{analysis['entrypoint']}\n\n"

    if analysis["cmd"]:
        output += "## Command\n"
        output += f"{analysis['cmd']}\n\n"

    if analysis["user"]:
        output += f"## User: {analysis['user']}\n\n"

    if analysis["workdir"]:
        output += f"## Working Directory: {analysis['workdir']}\n\n"

    # Labels
    if analysis["labels"]:
        output += "## Labels\n"
        for key, value in analysis["labels"].items():
            output += f"- {key}={value}\n"
        output += "\n"

    # Print the formatted output
    print_syntax(output, "markdown")


def main() -> None:
    """
    Main function to orchestrate the Dockerfile generation process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    # Validate Docker is installed
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error(
            "Docker CLI is not installed or not in PATH. Please install Docker."
        )
        sys.exit(1)

    # Analyze the Docker image
    logging.info(f"Analyzing Docker image: {args.image_reference}")
    analyzer = DockerImageAnalyzer(
        args.image_reference, local=args.local, detailed=args.detailed
    )

    analysis = analyzer.analyze()
    logging.info("Analysis complete")

    # If skip-llm flag is set, just output the raw analysis
    if args.skip_llm:
        logging.info("Skipping LLM processing, outputting raw analysis")
        output_raw_analysis(analysis)
        return

    # Set up LLM provider
    api_key = get_api_key(args.api_key, args.provider)
    if not api_key:
        logging.error(
            f"No API key provided for {args.provider}. Use -k/--api-key or set the "
            f"{'ANTHROPIC_API_KEY' if args.provider == 'anthropic' else 'OPENAI_API_KEY'} "
            "environment variable."
        )
        sys.exit(1)

    # Resolve effective model
    model = resolve_model(args.provider, args.model)

    if args.provider == "anthropic":
        provider = AnthropicProvider(api_key=api_key)
    else:  # openai
        provider = OpenAIProvider(api_key=api_key)

    # Create prompt for the LLM
    prompt = create_prompt(analysis, args.include_instructions)
    logging.debug(f"Generated prompt:\n{prompt}")

    # Create conversation and send to LLM
    conversation = Conversation()
    conversation.add_message("user", prompt)

    logging.info(f"Sending request to {args.provider} model: {model}")
    dockerfile_content = provider.generate_response(conversation, model)

    # Clean up the response if needed (remove markdown formatting if present)
    if dockerfile_content.startswith("```"):
        # Split by lines and strip possible code fences
        lines = dockerfile_content.split("\n")
        # Remove leading fence
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove trailing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        dockerfile_content = "\n".join(lines).strip()

    # Write Dockerfile to file if specified, otherwise print to stdout
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(dockerfile_content.rstrip() + "\n")
            logging.info(f"Dockerfile has been saved to {args.output}")
        except Exception as e:
            logging.error(f"Failed to write Dockerfile to {args.output}: {e}")
            sys.exit(1)
    else:
        print_syntax(dockerfile_content, "dockerfile")


if __name__ == "__main__":
    main()
