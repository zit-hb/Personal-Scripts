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
#   -n, --num-questions N       Number of questions to ask the model (default: 5).
#   -m, --model MODEL           Model to use (default: "o1-mini").
#   -o, --output OUTPUT         Path to a JSON file for saving Q&A pairs and final text.
#   -v, --verbose               Enable verbose logging (INFO level).
#   -vv, --debug                Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - openai (install via: pip install openai==1.64.0)
#   - rich (install via: pip install rich==13.9.4)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import sys
from typing import List, Dict
from rich.console import Console
from rich.markdown import Markdown
from openai import OpenAI

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
        default="o1-mini",
        help='Model to use (default: "o1-mini").',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to a JSON file for saving Q&A pairs and final text.",
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
        level = logging.ERROR

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_api_key(provided_key: str) -> str:
    """
    Returns the API key, either from arguments or environment variables.
    """
    return provided_key or os.getenv("OPENAI_API_KEY") or ""


def build_initial_conversation(prompt: str, num_questions: int) -> List[Dict[str, str]]:
    """
    Builds the initial conversation messages list (no system role for o1-mini).
    """
    return [
        {
            "role": "user",
            "content": (
                f"You are an AI that helps brainstorm any given topic. "
                f"The user has requested help with the following topic:\n"
                f"'{prompt}'\n\n"
                f"You should ask {num_questions} relevant questions, one by one. "
                f"Each time, consider all previous questions and answers. "
                f"Focus on the most crucial aspects first. Once all questions are asked and answered, "
                f"you will receive a final prompt to stitch everything together into a detailed text. "
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
    Gathers multiline user input from stdin until EOF is encountered.
    """
    logging.info("Waiting for user input (press Ctrl+D to finish AFTER new line):")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


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
    Modifies the conversation in place.
    """
    # Get the first question from the model
    print("...")
    question_text = chat_with_model(client, model, conversation)
    conversation.append({"role": "assistant", "content": question_text})

    # Loop through the required number of questions
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

    if not args.prompt:
        logging.info(
            "No prompt provided. Please enter it below (press Ctrl+D to finish AFTER new line):"
        )
        print_markdown("**Prompt:**")
        args.prompt = gather_user_multiline_input().strip()
        if not args.prompt:
            logging.error("No prompt provided. Exiting.")
            sys.exit(1)

    # Create the OpenAI client
    client = OpenAI(api_key=api_key)

    # Build initial conversation
    conversation = build_initial_conversation(args.prompt, args.num_questions)

    # Ask questions and get user answers
    ask_questions(
        client=client,
        model=args.model,
        conversation=conversation,
        num_questions=args.num_questions,
    )

    # Generate final text using all Q&As
    final_text = generate_final_text(
        client=client,
        model=args.model,
        conversation=conversation,
    )

    print_markdown("### Final Text\n")
    print_markdown(final_text)

    # If output path is specified, write prompt, Q&A pairs and final text to JSON
    if args.output:
        qas = parse_qa_pairs(conversation)
        write_output_json(args.prompt, qas, final_text, args.output)

    logging.info("Conversation completed successfully.")


if __name__ == "__main__":
    main()
