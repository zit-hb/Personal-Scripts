#!/usr/bin/env python3

# -------------------------------------------------------
# Script: create_script.py
#
# Description:
# This script leverages an LLM to either:
#   1) Create a new Python script from scratch
#   2) Modify an existing Python script
#
# Usage:
#   ./create_script.py <command> [options]
#
# Commands:
#   new       Create a new script from scratch.
#   modify    Modify an existing script.
#
# Options:
#   -k, --api-key API_KEY               Your OpenAI or Anthropic API key (or set via OPENAI_API_KEY / ANTHROPIC_API_KEY).
#   -m, --model MODEL                   Model to use (default: "o1-mini").
#   -P, --provider PROVIDER             Which LLM provider to use: openai or anthropic (default: openai).
#   -T, --max-tokens MAX_TOKENS         Maximum tokens to request from the LLM (default: 4096).
#   -S, --example-script FILE           Paths to example scripts to reference (can be specified multiple times).
#   -I, --instruction-set NAME          Names of instruction sets to include (can be specified multiple times).
#   -o, --output OUTPUT                 Path to file where the generated code is written.
#   -R, --disable-ruff                  Disable ruff formatting.
#   -v, --verbose                       Enable verbose logging (INFO level).
#   -vv, --debug                        Enable debug logging (DEBUG level).
#
# New-specific Options:
#   -p, --prompt PROMPT                 The user instructions (if not provided, read from stdin).
#
# Modify-specific Options:
#   -p, --prompt PROMPT                 The modification instructions (if not provided, read from stdin).
#   -i, --input INPUT                   The path to the script to modify (required).
#
# Template: ubuntu24.04
#
# Requirements:
#   - openai (install via: pip install openai==1.64.0)
#   - rich (install via: pip install rich==13.9.4)
#   - ruff (install via: pip install ruff==0.9.7)
#   - anthropic (install via: pip install anthropic==0.48.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import re
import sys
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.syntax import Syntax

console = Console() if Console else None

CURRENT_YEAR = datetime.now().year

DOC_HEADER = f"""#!/usr/bin/env python3

# -------------------------------------------------------
# Script: script_name.py
#
# Description:
# Brief description of what the script does.
# Can span multiple lines with proper indentation.
#
# Usage:
#   ./script_name.py [options] argument
#
# Commands:
#   some_command       Description of the sub-command.
#
# Arguments:
#   - [argument]: Description of the argument.
#
# Options:
#   -o, --option OPTION     Description of the option.
#                           Additional indented details if needed.
#
# some_command Options:
#   -f, --foo               Description of the some_command-specific option.
#
# Template: ubuntu24.04
#
# Requirements:
#   - apt package (install via: apt-get install -y package)
#   - pypi package (install via: pip install package==version)
#
# -------------------------------------------------------
# © {CURRENT_YEAR} Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------
"""

MAIN_FUNCTIONS = """def parse_arguments() -> argparse.Namespace:
    \"\"\"
    Parses command-line arguments.
    \"\"\"
    # Implementation


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    \"\"\"
    Sets up the logging configuration.
    \"\"\"
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main() -> None:
    \"\"\"
    Main function to orchestrate the [specific task] process.
    \"\"\"
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.foo:
        handle_task_1()
    elif something():
        handle_task_2()
"""

EXAMPLE_SCRIPT = (
    """#!/usr/bin/env python3

# -------------------------------------------------------
# Script: process_file.py
#
# Description:
# Example of a Python script following the style guidelines.
#
# Usage:
#   ./process_file.py [options] input_file
#
# Arguments:
#   - input_file: Path to the input file.
#
# Options:
#   -o, --output OUTPUT     Output file path. (default: output.txt)
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - some-package (install via: pip install some-package==1.0.0)
#
# -------------------------------------------------------
# © """
    + str(CURRENT_YEAR)
    + """ Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from typing import List, Optional

from third_party_lib import some_function


def parse_arguments() -> argparse.Namespace:
    \"\"\"
    Parses command-line arguments.
    \"\"\"
    parser = argparse.ArgumentParser(
        description=\"Process a file to demonstrate the style guidelines.\"
    )
    parser.add_argument(
        \"input_file\",
        type=str,
        help=\"Path to the input file.\",
    )
    parser.add_argument(
        \"-o\",
        \"--output\",
        type=str,
        default=\"output.txt\",
        help=\"Output file path. (default: output.txt)\",
    )
    parser.add_argument(
        \"-v\",
        \"--verbose\",
        action=\"store_true\",
        help=\"Enable verbose logging (INFO level).\",
    )
    parser.add_argument(
        \"-vv\",
        \"--debug\",
        action=\"store_true\",
        help=\"Enable debug logging (DEBUG level).\",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    \"\"\"
    Sets up the logging configuration.
    \"\"\"
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format=\"%(levelname)s: %(message)s\")


def process_file(input_path: str, output_path: str) -> bool:
    \"\"\"
    Processes the input file and writes results to the output file.
    \"\"\"
    try:
        with open(input_path, \"r\") as in_file:
            content = in_file.read()

        processed_content = content.upper()

        with open(output_path, \"w\") as out_file:
            out_file.write(processed_content)

        logging.info(f\"Successfully processed '{input_path}' to '{output_path}'\")
        return True
    except FileNotFoundError:
        logging.error(f\"Input file '{input_path}' not found.\")
        return False
    except Exception as e:
        logging.error(f\"Unexpected error processing file: {e}\")
        return False


def main() -> None:
    \"\"\"
    Main function to orchestrate the file processing workflow.
    \"\"\"
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    logging.info(\"Starting file processing.\")

    if not os.path.isfile(args.input_file):
        logging.error(f\"Input path '{args.input_file}' is not a file.\")
        sys.exit(1)

    success = process_file(args.input_file, args.output)
    if not success:
        sys.exit(1)

    logging.info(\"File processing completed successfully.\")


if __name__ == \"__main__\":
    main()
"""
)


@dataclass
class Message:
    """
    A single message within a conversation.
    role: "system", "user", or "assistant".
    content: The text content of the message.
    """

    role: str
    content: str


@dataclass
class Conversation:
    """
    A conversation is a sequence of messages, typically in the form:
    - system or user instructions,
    - assistant response(s), etc.
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
    Subclasses must implement generate_response().
    """

    def generate_response(self, conversation: Conversation, model: str) -> str:
        """
        Sends the conversation to the model and returns the model's response text.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class OpenAIProvider(IProvider):
    """
    Implementation of IProvider that uses the 'OpenAI' client for chat completions.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initializes the OpenAI provider with the given API key.
        """
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, conversation: Conversation, model: str) -> str:
        """
        Sends the conversation to the specified model using the openai interface
        and returns the assistant's message content.
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
            logging.error(f"Error during model call: {e}")
            return "Error: could not retrieve a response from the model."


class AnthropicProvider(IProvider):
    """
    Implementation of IProvider that uses the 'Anthropic' client for chat completions.
    """

    def __init__(self, api_key: str, max_tokens: int = 8192) -> None:
        """
        Initializes the Anthropic provider with the given API key and max tokens.
        """
        self.client = Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def generate_response(self, conversation: Conversation, model: str) -> str:
        """
        Sends the conversation to the specified model using the anthropic interface
        and returns the assistant's message content by concatenating text blocks.
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
                if block.type == "text":
                    full_text += block.text
            return full_text.strip()
        except Exception as e:
            logging.error(f"Error during Anthropic call: {e}")
            return "Error: could not retrieve a response from the model."


class PromptGenerator:
    """
    A class to build "general instructions" by combining multiple instruction sets
    and any example script references. The exact texts for each instruction set can
    be stored or fetched within this class.
    """

    INSTRUCTION_SET_TEXTS = {
        "coding_style": [
            "Use Pythonic conventions (PEP 8).",
            "Make sure the code is valid and modern Python.",
            "Ensure all functions are strictly typed. Avoid using 'Any' unless absolutely necessary.",
            "Use meaningful variable and function names. Avoid abbreviations.",
            "Functions should be short and focused on a single task.",
            "The name of the script should follow the format 'verb_object.py', e.g., `summarize_text.py` or `find_best_images.py`.",
            f"Add the following doc header:\n```\n{DOC_HEADER}\n```\nDo not add 'Commands', 'Arguments', 'Options', and 'Requirements' sections if not needed.",
            f"Define the following functions:\n```\n{MAIN_FUNCTIONS}\n```",
            "All command line options need to have a reasonable and valid shortcut."
            "Do not add 'Args' and 'Returns' in function doc-strings. It is too verbose for a single-file script.",
            f"Here is an example script to demonstrate the expected coding style:\n```\n{EXAMPLE_SCRIPT}\n```",
        ],
        "minimal_modification": [
            "You are modifying an existing script. Only alter the necessary parts to fulfill my requests. Keep the rest of the code intact.",
        ],
        "major_modification": [
            "You are modifying an existing script. You are allowed to do significant changes to the script to fulfill my requests.",
            "Change whatever needs to be changed to get the best results.",
        ],
    }

    def __init__(self, instruction_sets: List[str], example_scripts: List[str]) -> None:
        """
        :param instruction_sets: A list of keys specifying which instruction sets to include.
        :param example_scripts: A list of file paths to example scripts that should be referenced.
        """
        self.instruction_sets = instruction_sets
        self.example_scripts = example_scripts

    def build_general_instructions_for_new(self) -> str:
        """
        Builds a string containing the general instructions for creating a new script.
        """
        lines: List[str] = ["# General Instructions for Creating a New Script #\n"]

        self._build_header(lines)
        self._add_instruction_sets(lines)
        self._add_example_scripts(lines)
        self._build_footer_new(lines)

        return "\n".join(lines)

    def build_general_instructions_for_modify(self) -> str:
        """
        Builds a string containing the general instructions for modifying an existing script.
        """
        lines: List[str] = [
            "# General Instructions for Modifying an Existing Script #\n"
        ]

        self._build_header(lines)
        self._add_instruction_sets(lines)
        self._add_example_scripts(lines)
        self._build_footer_modify(lines)

        return "\n".join(lines)

    def _build_header(self, lines: List[str]) -> None:
        """
        Common header instructions.
        """
        lines.append(
            "You will produce a single-file Python script. "
            "It must be enclosed in triple backticks. "
            "The response must contain only the code in those triple backticks (no extra text). "
            "Output the full code of the script; no placeholders. "
        )
        lines.append("Below are additional requirements:\n")

    def _add_instruction_sets(self, lines: List[str]) -> None:
        """
        Appends relevant instruction sets to the instructions.
        """
        for s in self.instruction_sets:
            instructions = self.INSTRUCTION_SET_TEXTS.get(s, [])
            for instruction in instructions:
                lines.append(f"- {instruction}")

    def _add_example_scripts(self, lines: List[str]) -> None:
        """
        Appends example script references (if any) to the instructions.
        """
        if not self.example_scripts:
            return

        lines.append(
            "\nYou should use the following script(s) as examples or references:"
        )
        for script_path in self.example_scripts:
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    content = f.read()
                lines.append(
                    f"\nExample script from {script_path}:\n```python\n{content}\n```\n"
                )
            except Exception as e:
                logging.warning(f"Could not read example script '{script_path}': {e}")

    def _build_footer_new(self, lines: List[str]) -> None:
        """
        Footer for new script instructions.
        """
        lines.append(
            "\nIn my next prompt, I will provide you instructions on what to implement."
        )

    def _build_footer_modify(self, lines: List[str]) -> None:
        """
        Footer for modify script instructions.
        """
        lines.append(
            "\nIn my next prompt, I will provide you with the script that needs modification."
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return them as a namespace.
    """
    parser = argparse.ArgumentParser(
        description="Create or modify Python scripts using an LLM."
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="Your OpenAI or Anthropic API key (or set via OPENAI_API_KEY or ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="o1-mini",
        help='Model to use (default: "o1-mini").',
    )
    parser.add_argument(
        "-P",
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="Which LLM provider to use (default: openai).",
    )
    parser.add_argument(
        "-T",
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to request from the LLM (default: 4096).",
    )
    parser.add_argument(
        "-S",
        "--example-script",
        type=str,
        action="append",
        default=[],
        help="Paths to example scripts to reference (can be specified multiple times).",
    )
    parser.add_argument(
        "-I",
        "--instruction-set",
        type=str,
        action="append",
        default=None,
        help="Keys of instruction sets to include (can be specified multiple times).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path for the resulting code.",
    )
    parser.add_argument(
        "-R",
        "--disable-ruff",
        action="store_true",
        help="Disable ruff formatting.",
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

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command to execute."
    )

    # Sub-command: new
    parser_new = subparsers.add_parser("new", help="Create a new script from scratch.")
    parser_new.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="The user instructions (if not provided, read from stdin).",
    )

    # Sub-command: modify
    parser_modify = subparsers.add_parser("modify", help="Modify an existing script.")
    parser_modify.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="The modification instructions (if not provided, read from stdin).",
    )
    parser_modify.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the script to be modified (required).",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration based on verbosity flags.
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


def gather_user_multiline_input() -> str:
    """
    Gathers multiline user input from stdin until EOF is encountered.
    """
    logging.info("Waiting for user input (press Ctrl+D to finish AFTER new line):")
    lines: List[str] = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def extract_code_from_response(response: str) -> str:
    """
    Extracts Python code enclosed in triple backticks from the LLM response.
    If no code block is found, returns the entire response as fallback.
    """
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if not code_blocks:
        logging.warning("No triple-backtick code block found in the response.")
        return response.strip() + "\n"
    return code_blocks[0].strip() + "\n"


def run_ruff_on_code(code: str) -> str:
    """
    Attempts to format the code with ruff, if available.
    Returns the formatted code, or the original if ruff is not installed or fails.
    """
    if shutil.which("ruff") is None:
        logging.warning("Ruff not found; skipping formatting.")
        return code

    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp.flush()
        try:
            subprocess.run(
                ["ruff", "format", tmp.name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as ex:
            logging.warning(f"Ruff formatting failed: {ex}")
            return code
        # Rewind and read back the potentially formatted code
        tmp.seek(0)
        return tmp.read().strip() + "\n"


def print_code(code: str) -> None:
    """
    Prints code to stdout with syntax highlighting if available.
    """
    if console:
        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
        console.print(syntax)
    else:
        print(code)


def handle_new(
    args: argparse.Namespace, provider: IProvider, default_instruction_sets: List[str]
) -> None:
    """
    Handles the 'new' sub-command to create a new script from scratch.
    """
    # Gather user instructions
    if args.prompt:
        user_instructions = args.prompt
    else:
        print("Please provide the user instructions for creating the new script:")
        user_instructions = gather_user_multiline_input()
        if not user_instructions:
            logging.error("No user instructions provided. Exiting.")
            sys.exit(1)
        print("...")

    # Determine instruction sets
    if args.instruction_set:
        used_instruction_sets = args.instruction_set
    else:
        used_instruction_sets = default_instruction_sets

    # Build the general instructions
    prompt_gen = PromptGenerator(
        instruction_sets=used_instruction_sets,
        example_scripts=args.example_script,
    )
    general_instructions = prompt_gen.build_general_instructions_for_new()

    # Create conversation
    conversation = Conversation()
    # 1) Add user message with general instructions
    conversation.add_message("user", general_instructions)
    # 2) Hard-code the assistant's first response
    conversation.add_message(
        "assistant", "Understood. I am waiting for instructions on what to implement."
    )
    # 3) Add user instructions
    conversation.add_message("user", user_instructions)
    # 4) Generate response from the LLM
    response_text = provider.generate_response(conversation, model=args.model)

    # Extract code from the LLM response
    code = extract_code_from_response(response_text)

    # Optionally format with ruff (if installed and not disabled)
    if not args.disable_ruff:
        code = run_ruff_on_code(code)

    # Print code to stdout
    print_code(code)

    # If output file is specified, write code to that file
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            logging.error(f"Error writing output file '{args.output}': {e}")


def handle_modify(
    args: argparse.Namespace, provider: IProvider, default_instruction_sets: List[str]
) -> None:
    """
    Handles the 'modify' sub-command to modify an existing script.
    """
    # Read the existing script to modify
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            script_content = f.read()
    except Exception as e:
        logging.error(f"Could not read input script '{args.input}': {e}")
        sys.exit(1)

    # Gather user instructions
    if args.prompt:
        modification_instructions = args.prompt
    else:
        print("Please provide the modification instructions:")
        modification_instructions = gather_user_multiline_input()
        if not modification_instructions:
            logging.error("No modification instructions provided. Exiting.")
            sys.exit(1)
        print("...")

    # Determine instruction sets
    if args.instruction_set:
        used_instruction_sets = args.instruction_set
    else:
        used_instruction_sets = default_instruction_sets

    # Build the general instructions
    prompt_gen = PromptGenerator(
        instruction_sets=used_instruction_sets,
        example_scripts=args.example_script,
    )
    general_instructions = prompt_gen.build_general_instructions_for_modify()

    # Create conversation
    conversation = Conversation()
    # 1) Add user message with general instructions
    conversation.add_message("user", general_instructions)
    # 2) Hard-code the assistant's first response
    conversation.add_message(
        "assistant",
        "Understood. I will wait for your next message that will contain the full code of the script.",
    )
    # 3) Add user message with the script content in triple backticks
    code_block = f"```python\n{script_content}\n```"
    conversation.add_message("user", code_block)
    # 4) Hard-code the assistant's second response
    conversation.add_message(
        "assistant",
        "Thank you. I have received the code that should be modified. "
        "I am waiting for instructions on what to implement.",
    )
    # 5) Add user message with the modification instructions
    conversation.add_message("user", modification_instructions)
    # 6) Generate response from the LLM
    response_text = provider.generate_response(conversation, model=args.model)

    # Extract code from the LLM response
    code = extract_code_from_response(response_text)

    # Optionally format with ruff (if installed and not disabled)
    if not args.disable_ruff:
        code = run_ruff_on_code(code)

    # Print code to stdout
    print_code(code)

    # If output file is specified, write code to that file
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            logging.error(f"Error writing output file '{args.output}': {e}")


def main() -> None:
    """
    Main entry point for the create_script CLI.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    api_key = get_api_key(args.api_key, args.provider)
    if not api_key:
        logging.error(
            "No API key provided. Use -k/--api-key or set the appropriate environment variable."
        )
        sys.exit(1)

    if args.provider.lower() == "anthropic":
        provider = AnthropicProvider(api_key=api_key, max_tokens=args.max_tokens)
    elif args.provider.lower() == "openai":
        provider = OpenAIProvider(api_key=api_key)
    else:
        logging.error(f"Invalid provider: {args.provider}")
        sys.exit(1)

    if args.command == "new":
        handle_new(args, provider, ["coding_style"])
    elif args.command == "modify":
        handle_modify(args, provider, ["coding_style", "minimal_modification"])
    else:
        logging.error("Invalid sub-command.")


if __name__ == "__main__":
    main()
