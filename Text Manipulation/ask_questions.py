#!/usr/bin/env python3

# -------------------------------------------------------
# Script: ask_questions.py
#
# Description:
# This script takes a user-provided prompt about a topic and interacts
# with OpenAI to refine details by asking a series of questions. Each
# new question considers all previous questions and answers. Finally,
# it generates a detailed text about the topic.
#
# Usage:
#   ./ask_questions.py [options]
#
# Options:
#   -p, --prompt PROMPT         The main user topic or request.
#                               If not provided, you'll be asked interactively.
#   -k, --api-key API_KEY       Your OpenAI API key (or set via OPENAI_API_KEY).
#   -n, --num-questions N       Number of questions to ask (default: 5).
#   -m, --model MODEL           Model to use (default: "o4-mini").
#   -o, --output OUTPUT         Path to a JSON file for saving the session.
#   -i, --input INPUT           Path to a JSON file with an existing session to continue.
#   -v, --verbose               Enable verbose logging (INFO level).
#   -vv, --debug                Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - openai (install via: pip install openai==1.64.0)
#   - rich (install via: pip install rich==13.9.4)
#   - prompt_toolkit (install via: pip install prompt_toolkit==3.0.50)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Tuple

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession

console = Console() if Console else None


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Interactively ask questions about a user-provided prompt."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="The main user topic or request.",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="Your OpenAI API key (or set via OPENAI_API_KEY).",
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to ask the model (default: 5).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="o4-mini",
        help='Model to use (default: "o4-mini").',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to a JSON file for saving the session.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to a JSON file with an existing session to continue from.",
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


def get_api_key(provided_key: str) -> str:
    """
    Returns the API key, either from arguments or environment variables.
    """
    return provided_key or os.getenv("OPENAI_API_KEY") or ""


def build_initial_conversation(
    prompt: str, total_questions: int
) -> List[Dict[str, str]]:
    """
    Builds the initial conversation messages list,
    indicating the total number of questions (existing + new).
    o1-mini does not have a system role, so we use the user role for the first message.
    In the future, this might change to a system role.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Your task is to help a user to refine any given topic by asking relevant questions. "
                f"You should ask {total_questions} relevant questions about the topic, one by one. "
                f"When you are creating a new question, consider all previous questions and answers as well. "
                f"Focus on the most crucial aspects first. Once all questions are asked and answered, "
                f"you will receive a final prompt to stitch everything together into a detailed summary. "
                f"\nThe user has provided the following instructions:\n"
                f"```\n{prompt}\n```\n\n"
                f"Now, please start by asking the first question."
            ),
        }
    ]


def chat_with_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """
    Sends a chat request to the specified model with the given messages.
    Returns the model's response as a string.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error during model call: {e}")
        return "Error: could not retrieve a response from the model."


def gather_user_multiline_input() -> str:
    """
    Gathers multiline user input using prompt_toolkit with full editing capabilities.
    Supports arrow keys, deletion, syntax highlighting, and other terminal features.
    """
    session = PromptSession(
        multiline=True,
        enable_history_search=True,
        complete_while_typing=True,
    )

    logging.info("Enter your text (press ESC followed by Enter to submit):")

    try:
        text = session.prompt(
            "> ",
            default="",
        )
        return text.strip()
    except KeyboardInterrupt:
        logging.warning("\nInput cancelled by user.")
        sys.exit(1)
    except EOFError:
        logging.warning("\nEOF detected.")
        sys.exit(1)


def print_markdown(text: str) -> None:
    """
    Prints text, rendering it as markdown.
    """
    console.print(Markdown(text))


def ask_questions(
    client: OpenAI, model: str, conversation: List[Dict[str, str]], num_questions: int
) -> None:
    """
    Interactively asks questions from the model and gathers user answers.
    Modifies the conversation in place, for the specified number of new questions.
    """
    # Get the first question from the model
    print("...")
    question_text = chat_with_model(client, model, conversation)
    conversation.append({"role": "assistant", "content": question_text})

    # Loop through the required number of new questions
    for i in range(1, num_questions + 1):
        print_markdown(f"### Question {i}/{num_questions}\n{question_text}")
        print()
        print_markdown("**Answer:**")

        # Prompt until user provides a non-empty answer
        user_answer = gather_user_multiline_input().strip()

        # Record the user's answer
        conversation.append({"role": "user", "content": user_answer})

        # If more questions remain, get the next question from the model
        if i < num_questions:
            print("...")
            question_text = chat_with_model(client, model, conversation)
            conversation.append({"role": "assistant", "content": question_text})
        else:
            question_text = None


def generate_final_text(
    client: OpenAI, model: str, conversation: List[Dict[str, str]]
) -> str:
    """
    Asks the model for a final structured text output based on all Q&As.
    """
    print("...")

    conversation.append(
        {
            "role": "user",
            "content": (
                "Now that we have all questions and answers, please provide a detailed, "
                "structured text that addresses the main prompt comprehensively."
            ),
        }
    )

    final_text = chat_with_model(client, model, conversation)
    return final_text


def parse_qa_pairs(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Parses the conversation to extract (question, answer) pairs, skipping the initial instructions.
    """
    qas = []
    # The first message is the initial 'user' instructions, so start from index 1.
    i = 1
    while i < len(conversation) - 1:
        # We look for an assistant question followed by a user answer
        if (
            conversation[i]["role"] == "assistant"
            and i + 1 < len(conversation)
            and conversation[i + 1]["role"] == "user"
        ):
            qas.append(
                {
                    "question": conversation[i]["content"],
                    "answer": conversation[i + 1]["content"],
                }
            )
            i += 2
        else:
            i += 1
    return qas


def write_output_json(
    prompt: str, qas: List[Dict[str, str]], final_text: str, output_path: str
) -> None:
    """
    Writes the Q&A pairs and final text to a JSON file.
    """
    data = {
        "prompt": prompt,
        "qas": qas,
        "final_text": final_text,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Conversation successfully written to '{output_path}'.")
    except Exception as e:
        logging.error(f"Error writing output JSON file: {e}")


def load_existing_session(file_path: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Loads an existing session from a JSON file. Returns the prompt and Q&A pairs.
    Ignores the final_text from the JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompt = data.get("prompt", "")
        qas = data.get("qas", [])
        if not prompt:
            logging.warning("No prompt found in the input JSON. Prompt will be empty.")
        return prompt, qas
    except Exception as e:
        logging.error(f"Error reading input JSON file '{file_path}': {e}")
        return "", []


def print_existing_conversation(existing_qas: List[Dict[str, str]]) -> None:
    """
    Prints the existing conversation in markdown format if available.
    """
    if not existing_qas:
        return
    print_markdown("### Previous conversation:")
    for i, qa in enumerate(existing_qas, start=1):
        print_markdown(f"**Question {i}:** {qa['question']}")
        print_markdown(f"**Answer {i}:** {qa['answer']}")
    print()


def handle_prompt_and_session(
    args: argparse.Namespace,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Handles the logic for retrieving the prompt and any existing Q&A session data.
    """
    if args.input:
        # If --input is specified, read the session from that file
        prompt, existing_qas = load_existing_session(args.input)
        if not prompt:
            logging.error("No valid prompt found in the input JSON. Exiting.")
            sys.exit(1)
    else:
        # If no input file given, check for prompt in CLI or ask interactively
        if not args.prompt:
            logging.info(
                "No prompt provided. Please enter it below (press Ctrl+D to finish AFTER new line):"
            )
            print_markdown("**Prompt:**")
            user_prompt = gather_user_multiline_input().strip()
            if not user_prompt:
                logging.error("No prompt provided. Exiting.")
                sys.exit(1)
            prompt = user_prompt
            existing_qas = []
        else:
            prompt = args.prompt
            existing_qas = []

    return prompt, existing_qas


def orchestrate_conversation(
    client: OpenAI,
    model: str,
    prompt: str,
    existing_qas: List[Dict[str, str]],
    num_new_questions: int,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Orchestrates the conversation flow:
      - Builds an initial conversation
      - Integrates existing Q&A
      - Asks any new questions
      - Generates final text
    Returns the final conversation and final text.
    """
    existing_qas_count = len(existing_qas)
    total_questions = existing_qas_count + num_new_questions

    # Build initial conversation, referencing the total number of questions
    conversation = build_initial_conversation(prompt, total_questions)

    # Prepend existing Q&A pairs if continuing from a previous session
    if existing_qas_count > 0:
        for qa in existing_qas:
            conversation.append({"role": "assistant", "content": qa["question"]})
            conversation.append({"role": "user", "content": qa["answer"]})

    # Ask only the *new* questions
    ask_questions(
        client=client,
        model=model,
        conversation=conversation,
        num_questions=num_new_questions,
    )

    # Generate final text using all Q&As
    final_text = generate_final_text(
        client=client,
        model=model,
        conversation=conversation,
    )

    return conversation, final_text


def main() -> None:
    """
    Main function to orchestrate the question-asking process and final summary generation.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    api_key = get_api_key(args.api_key)
    if not api_key:
        logging.error(
            "No API key provided. Use -k or set OPENAI_API_KEY environment variable."
        )
        sys.exit(1)

    prompt, existing_qas = handle_prompt_and_session(args)
    print_existing_conversation(existing_qas)

    # Create the OpenAI client
    client = OpenAI(api_key=api_key)

    # Run the conversation
    conversation, final_text = orchestrate_conversation(
        client=client,
        model=args.model,
        prompt=prompt,
        existing_qas=existing_qas,
        num_new_questions=args.num_questions,
    )

    print_markdown("### Final Text\n")
    print_markdown(final_text)

    # If output path is specified, write prompt, Q&A pairs and final text to JSON
    if args.output:
        qas = parse_qa_pairs(conversation)
        write_output_json(prompt, qas, final_text, args.output)

    logging.info("Conversation completed successfully.")


if __name__ == "__main__":
    main()
