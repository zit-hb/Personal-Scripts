#!/usr/bin/env python3

# -------------------------------------------------------
# Script: call_llm.py
#
# Description:
# This script provides two primary functionalities:
# 1. Generate an OpenAI function definition schema based on a user prompt.
# 2. Use an existing function definition to extract information from a user prompt.
#
# Usage:
# ./call_llm.py [command] [options]
#
# Commands:
#   - gen-def (g)               Generate a function definition schema.
#   - use-def (u)               Use an existing function definition to process a prompt.
#
# Global Options:
#   -v, --verbose               Enable verbose logging (INFO level).
#   -vv, --debug                Enable debug logging (DEBUG level).
#
# Options for gen-def:
#   -p, --prompt PROMPT         A single prompt for generating the function definition (required).
#   -o, --output-file FILE      Store the generated function definition in a JSON file.
#   -k, --api-key API_KEY       Your OpenAI API key (can also be set via .env or OPENAI_API_KEY).
#   -m, --model MODEL           OpenAI model to use (default: "gpt-4o").
#
# Options for use-def:
#   -i, --functions-file FILE   Path to the JSON file containing the function definition (required).
#   -p, --prompt PROMPT         The prompt to process using the function definition.
#   -P, --prompt-file FILE      Path to a file containing additional prompt content to append.
#   -o, --output-file FILE      Store the assistant's response in a JSON file.
#   -k, --api-key API_KEY       Your OpenAI API key (can also be set via .env or OPENAI_API_KEY).
#   -m, --model MODEL           OpenAI model to use (default: "gpt-4o").
#
# Template: ubuntu22.04
#
# Requirements:
# - openai (install via: pip install openai==1.55.3)
# - jsonschema (install via: pip install jsonschema==4.23.0)
# - rich (optional, for enhanced terminal outputs) (install via: pip install rich==13.9.4)
# - python-dotenv (install via: pip install python-dotenv==1.0.1)
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("The 'openai' module is not installed. Install it using 'pip install openai'.")
    sys.exit(1)

from jsonschema import validate, ValidationError

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Attempt to import rich for enhanced terminal outputs
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Base schema for validating function definitions
FUNCTION_DEFINITION_BASE_SCHEMA = {
    "type": "object",
    "required": ["name", "description", "parameters"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {
            "type": "object",
            "required": ["type", "properties"],
            "properties": {
                "type": {"type": "string", "enum": ["object"]},
                "properties": {"type": "object"},
                "required": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True
        },
    },
    "additionalProperties": True
}

# Definition generation data
SYSTEM_MESSAGE_GENERATE_DEFINITION = (
    "You are an assistant that generates OpenAI function calling schemas based on user requirements. Given a description "
    "of the data to extract, output only the JSON schema defining the function, including 'name', 'description', and "
    "'parameters' with their types and descriptions. You create a schema that defines what information will be extracted "
    "from the prompt. So focus on output and not input. Do not include any additional text, comments, or formatting."
)

EXAMPLE_PROMPT_1 = (
    "Input will be code. I want a list of all modules and libraries. And the language."
)

EXAMPLE_DEFINITION_1 = (
    """
    {
        "name": "analyze_code_output",
        "description": "Analyze the provided code to determine its programming language and extract a list of all imported modules and libraries.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "The programming language of the provided code (e.g., Python, JavaScript, Java)."
                },
                "modules": {
                    "type": "array",
                    "description": "A list of all modules and libraries imported or used in the code.",
                    "items": {
                        "type": "string",
                        "description": "Name of the module or library."
                    }
                }
            },
            "required": ["language", "modules"],
            "additionalProperties": false
        }
    }
    """
)

EXAMPLE_PROMPT_2 = (
    "I want to extract the mood, style, and language of a text."
)

EXAMPLE_DEFINITION_2 = (
    """
    {
        "name": "analyze_text_output",
        "description": "Analyze the provided text to determine its mood, style, and language.",
        "parameters": {
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "description": "The emotional tone or mood of the text (e.g., happy, sad, angry, neutral)."
                },
                "style": {
                    "type": "string",
                    "description": "The writing style of the text (e.g., formal, informal, academic, conversational)."
                },
                "language": {
                    "type": "string",
                    "description": "The natural language in which the text is written (e.g., English, Spanish, French)."
                }
            },
            "required": ["mood", "style", "language"],
            "additionalProperties": false
        }
    }
    """
)

MAX_RETRIES = 3  # Number of retries for generating valid function definitions


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments with subcommands and global options.
    """
    parser = argparse.ArgumentParser(
        description="OpenAI Function Definition Manager",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Global options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (INFO level).'
    )
    parser.add_argument(
        '-vv', '--debug',
        action='store_true',
        help='Enable debug logging (DEBUG level).'
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # Subparser for generating definitions
    gen_parser = subparsers.add_parser('gen-def', aliases=['g'], help='Generate a function definition schema.')
    gen_parser.add_argument(
        '-p', '--prompt',
        type=str,
        required=True,
        help='A prompt for generating the function definition.'
    )
    gen_parser.add_argument(
        '-o', '--output-file',
        type=str,
        help='Store the generated function definition in a JSON file.'
    )
    gen_parser.add_argument(
        '-k', '--api-key',
        type=str,
        help='Your OpenAI API key (can also be set via .env or OPENAI_API_KEY).'
    )
    gen_parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: "gpt-4o").'
    )

    # Subparser for using definitions
    use_parser = subparsers.add_parser('use-def', aliases=['u'], help='Use a function definition to process a prompt.')
    use_parser.add_argument(
        '-i', '--functions-file',
        type=str,
        required=True,
        help='Path to the JSON file containing the function definition.'
    )
    use_parser.add_argument(
        '-p', '--prompt',
        type=str,
        help='The prompt to process using the function definition.'
    )
    use_parser.add_argument(
        '-P', '--prompt-file',
        type=str,
        help='Path to a file containing additional prompt content to append.'
    )
    use_parser.add_argument(
        '-o', '--output-file',
        type=str,
        metavar='FILE',
        help='Store the assistant\'s response in a JSON file.'
    )
    use_parser.add_argument(
        '-k', '--api-key',
        type=str,
        help='Your OpenAI API key (can also be set via .env or OPENAI_API_KEY).'
    )
    use_parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: "gpt-4o").'
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
        level = logging.ERROR

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def initialize_console() -> Optional[Console]:
    """
    Initializes a Rich console if available.
    """
    if RICH_AVAILABLE:
        return Console()
    return None


def generate_function_definition(client: OpenAI, prompt: str, model: str) -> Optional[Dict[str, Any]]:
    """
    Generates a function definition based on the provided prompt.
    Ensures that the definition is valid according to the base schema.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.debug(f"Generating function definition for prompt: {prompt} (Attempt {attempt})")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE_GENERATE_DEFINITION},
                    {"role": "user", "content": EXAMPLE_PROMPT_1},
                    {"role": "assistant", "content": EXAMPLE_DEFINITION_1},
                    {"role": "user", "content": EXAMPLE_PROMPT_2},
                    {"role": "assistant", "content": EXAMPLE_DEFINITION_2},
                    {"role": "user", "content": prompt}
                ]
            )

            message = response.choices[0].message
            function_definition = message.content.strip()

            # Attempt to parse JSON
            func_def = json.loads(function_definition)

            # Validate the function definition
            if validate_function_definition(func_def, verbose=False):
                logging.info("Successfully generated function definition for prompt.")
                return func_def
            else:
                logging.warning(
                    f"Validation failed for generated function definition on attempt {attempt}. Retrying...")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed on attempt {attempt}: {e}")
        except Exception as e:
            logging.error(f"Error generating function definition on attempt {attempt}: {e}")

    logging.error(f"Failed to generate a valid function definition for prompt after {MAX_RETRIES} attempts.")
    return None


def validate_function_definition(function: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Validates the function definition against the base schema.
    """
    try:
        validate(instance=function, schema=FUNCTION_DEFINITION_BASE_SCHEMA)
        if function["parameters"]["type"] != "object":
            logging.error(
                f"Function '{function['name']}' has invalid 'parameters.type'. Expected 'object', got '{function['parameters']['type']}'")
            return False
        if verbose:
            logging.info("Function definition is valid.")
        return True
    except ValidationError as ve:
        logging.error(f"Validation error: {ve.message}")
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
    return False


def save_definition(function: Dict[str, Any], filename: str) -> bool:
    """
    Saves the function definition to a JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(function, f, indent=4)
        logging.info(f"Function definition saved to '{filename}'.")
        return True
    except Exception as e:
        logging.error(f"Error saving function definition: {e}")
        return False


def display_definition(function: Dict[str, Any], console: Optional[Console] = None) -> None:
    """
    Displays the function definition with name and description, focusing on properties.
    """
    if not function:
        logging.warning("No function definition to display.")
        return

    name = function.get("name", "N/A")
    description = function.get("description", "N/A")
    parameters = function.get("parameters", {})

    if RICH_AVAILABLE and console:
        # Print name and description
        console.print(f"{name}", style="bold underline")
        console.print(f"{description}\n", style="italic")

        # Prepare table for parameters
        table = Table(title="Parameters", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta", no_wrap=True)
        table.add_column("Description", style="green")

        props = parameters.get("properties", {})
        for prop_name, prop_details in props.items():
            prop_type = prop_details.get("type", "N/A")
            prop_desc = prop_details.get("description", "N/A")
            table.add_row(prop_name, prop_type, prop_desc)

        console.print(table)
    else:
        print(f"\nFunction Name   : {name}")
        print(f"Description     : {description}\n")
        print("Parameters:")
        for prop_name, prop_details in parameters.get("properties", {}).items():
            prop_type = prop_details.get("type", "N/A")
            prop_desc = prop_details.get("description", "N/A")
            print(f"  - {prop_name} ({prop_type}): {prop_desc}")
        print("\n" + "-" * 50)


def load_definition(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """
    Loads the function definition from a JSON file and ensures it's a list of dictionaries.
    """
    if not os.path.exists(filepath):
        logging.error(f"Function definition file '{filepath}' does not exist.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            definition = json.load(f)
        if isinstance(definition, dict):
            definitions = [definition]
        elif isinstance(definition, list):
            definitions = definition
        else:
            logging.error("Function definition file must contain a JSON object or a list of objects.")
            return None
        logging.info(f"Function definition loaded from '{filepath}'.")
        return definitions
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed for '{filepath}': {e}")
    except Exception as e:
        logging.error(f"Error loading function definition from '{filepath}': {e}")
    return None


def use_function_definition(client: OpenAI, functions: List[Dict[str, Any]], prompt: str, model: str,
                            output_file: Optional[str] = None, console: Optional[Console] = None) -> None:
    """
    Uses the provided function definition to process the user prompt and extract information.
    """
    system_prompt = (
        "You are an assistant that extracts information from any text input by utilizing provided function definitions. "
        "Use the functions to extract the necessary information as specified."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Validate that 'functions' is a list of dictionaries
    if not isinstance(functions, list):
        logging.error("'functions' should be a list of dictionaries.")
        sys.exit(1)

    for idx, func in enumerate(functions):
        if not isinstance(func, dict):
            logging.error(f"Function at index {idx} is not a dictionary.")
            sys.exit(1)
        required_keys = {"name", "description", "parameters"}
        if not required_keys.issubset(func.keys()):
            logging.error(f"Function at index {idx} is missing required keys: {required_keys - func.keys()}")
            sys.exit(1)
        if not isinstance(func["name"], str) or not isinstance(func["description"], str):
            logging.error(f"Function at index {idx} has invalid 'name' or 'description' types. Both should be strings.")
            sys.exit(1)
        if not isinstance(func["parameters"], dict):
            logging.error(f"Function at index {idx} has invalid 'parameters' type. It should be a dictionary.")
            sys.exit(1)

    try:
        logging.debug("Sending user prompt to OpenAI for processing.")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call="auto"
        )

        message = response.choices[0].message

        # Process and display the assistant's response
        process_assistant_response(message, console)

        # Optionally save the assistant's response as JSON
        if output_file:
            save_assistant_response(message, output_file)

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        sys.exit(1)


def process_assistant_response(message: Any, console: Optional[Console] = None) -> None:
    """
    Processes the assistant's response to extract valuable information.
    """
    if not message:
        logging.warning("No response message to process.")
        return

    if hasattr(message, 'function_call') and message.function_call:
        arguments = message.function_call.arguments
        logging.debug(f"Processing function call arguments: {arguments}")
        try:
            arguments_json = json.loads(arguments)
            display_extracted_information(arguments_json, console)
        except json.JSONDecodeError:
            logging.error("Failed to parse function call arguments.")
    else:
        logging.warning("Assistant did not provide a function call. Displaying the content.")
        display_assistant_response(message, console)


def display_extracted_information(extracted_info: Dict[str, Any], console: Optional[Console] = None) -> None:
    """
    Displays the extracted valuable information from the assistant's function call.
    """
    if not extracted_info:
        logging.warning("No extracted information to display.")
        return

    if RICH_AVAILABLE and console:
        table = Table(title="Extracted Information", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Attribute", style="green", no_wrap=True)
        table.add_column("Value", style="magenta")

        for key, value in extracted_info.items():
            if isinstance(value, list):
                # Check if the list contains dictionaries
                if all(isinstance(item, dict) for item in value):
                    # Convert each dictionary to a JSON-formatted string
                    value_str = "\n".join([json.dumps(item, indent=2) for item in value])
                else:
                    # Assume the list contains strings or other serializable types
                    value_str = "\n".join(map(str, value))
            elif isinstance(value, dict):
                # If the value itself is a dictionary, serialize it to JSON
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            table.add_row(key.capitalize(), value_str)
        console.print(table)
    else:
        print("\nExtracted Information:")
        for key, value in extracted_info.items():
            if isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    value_str = "\n".join([json.dumps(item, indent=2) for item in value])
                else:
                    value_str = "\n".join(map(str, value))
            elif isinstance(value, dict):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            print(f"  {key.capitalize():<15}: {value_str}")


def display_assistant_response(message: Any, console: Optional[Console] = None) -> None:
    """
    Displays the assistant's response in a formatted manner.
    """
    if not message:
        logging.warning("No response message to display.")
        return

    if RICH_AVAILABLE and console:
        table = Table(title="Assistant's Response", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Content", style="magenta")
        table.add_row(message.content or "")
        console.print(table)
    else:
        print("\nAssistant's Response:")
        print(message.content or "")


def save_assistant_response(response: Any, filepath: str) -> None:
    """
    Saves the assistant's response to a JSON file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Convert the message object to a dictionary for serialization
            response_dict = {
                "content": response.content,
                "function_call": {
                    "name": response.function_call.name,
                    "arguments": response.function_call.arguments
                } if hasattr(response, 'function_call') and response.function_call else None,
                "role": response.role
            }
            json.dump(response_dict, f, indent=4)
        logging.info(f"Assistant's response saved to '{filepath}'.")
    except Exception as e:
        logging.error(f"Error saving assistant's response: {e}")


def handle_gen_def_command(args: argparse.Namespace, console: Optional[Console]) -> None:
    """
    Handles the 'gen-def' command.
    """
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=args.api_key)
        logging.debug("Initialized OpenAI client successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # Generate function definition
    function_def = generate_function_definition(client, args.prompt, model=args.model)
    if function_def:
        # If output-file is specified, save to file
        if args.output_file and save_definition(function_def, args.output_file):
            logging.info(f"Function definition saved to '{args.output_file}'.")
        # Always display the definition
        display_definition(function_def, console)
    else:
        logging.error("No valid function definition was generated.")
        sys.exit(1)


def handle_use_def_command(args: argparse.Namespace, console: Optional[Console]) -> None:
    """
    Handles the 'use-def' command.
    """
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=args.api_key)
        logging.debug("Initialized OpenAI client successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # Load function definition
    functions = load_definition(args.functions_file)
    if not functions:
        logging.error("Failed to load function definition. Exiting.")
        sys.exit(1)

    # Display definition if verbose
    if args.verbose:
        for func in functions:
            display_definition(func, console)

    # Prepare the prompt
    final_prompt = args.prompt if args.prompt else ""
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            logging.error(f"Prompt file '{args.prompt_file}' does not exist.")
            sys.exit(1)
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                additional_prompt = f.read().strip()
            if final_prompt:
                final_prompt += "\n\n"
            final_prompt += additional_prompt
            logging.info(f"Appended content from prompt file '{args.prompt_file}' to the prompt.")
        except Exception as e:
            logging.error(f"Error reading prompt file '{args.prompt_file}': {e}")
            sys.exit(1)

    if not final_prompt:
        logging.error("No prompt provided. Use '-p/--prompt' or '-P/--prompt-file' to provide a prompt.")
        sys.exit(1)

    # Use function definition to process the prompt
    use_function_definition(
        client=client,
        functions=functions,
        prompt=final_prompt,
        model=args.model,
        output_file=args.output_file,
        console=console
    )


def main() -> None:
    """
    Main function to orchestrate the CLI commands.
    """
    args = parse_arguments()
    setup_logging(
        verbose=args.verbose,
        debug=args.debug
    )
    console = initialize_console()

    # Retrieve the API key from arguments or environment variable
    args.api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not args.api_key:
        print("Error: API key not provided via command-line argument or environment variable OPENAI_API_KEY.")
        sys.exit(1)

    if args.command in ['gen-def', 'g']:
        handle_gen_def_command(args, console)

    elif args.command in ['use-def', 'u']:
        handle_use_def_command(args, console)

    else:
        logging.error("Unknown command.")
        sys.exit(1)

    logging.info("Process completed successfully.")


if __name__ == "__main__":
    main()
