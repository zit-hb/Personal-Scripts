#!/usr/bin/env python3

# -------------------------------------------------------
# Script: modify_image_text_files.py
#
# Description:
# This script scans a directory of images (`images_directory`) and checks for corresponding
# text files in another directory (`texts_directory`) with the same base name and a ".txt" extension.
# For each matching text file, it can optionally prepend and/or append a string to the first line.
#
# Usage:
# ./modify_image_text_files.py images_directory texts_directory [options]
#
# - images_directory: The path to the directory containing images.
# - texts_directory: The path to the directory containing text files.
#
# Options:
# --prepend PREPEND_STRING
#                       String to prepend to the first line of the text files.
# --append APPEND_STRING
#                       String to append to the first line of the text files.
#
# -------------------------------------------------------
# © 2024 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Modify first lines of text files corresponding to images in a directory.'
    )
    parser.add_argument(
        'images_directory',
        type=str,
        help='The path to the directory containing images.'
    )
    parser.add_argument(
        'texts_directory',
        type=str,
        help='The path to the directory containing text files.'
    )
    parser.add_argument(
        '--prepend',
        type=str,
        help='String to prepend to the first line of the text files.'
    )
    parser.add_argument(
        '--append',
        type=str,
        help='String to append to the first line of the text files.'
    )
    args = parser.parse_args()

    return args

def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def modify_text_file(text_file_path, prepend_str=None, append_str=None):
    """
    Modifies the first line of the text file by prepending and/or appending strings.

    Args:
        text_file_path (str): Path to the text file.
        prepend_str (str): String to prepend to the first line.
        append_str (str): String to append to the first line.
    """
    try:
        with open(text_file_path, 'r+') as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].rstrip('\n')
                if prepend_str:
                    first_line = prepend_str + first_line
                if append_str:
                    first_line = first_line + append_str
                lines[0] = first_line + '\n'
                f.seek(0)
                f.writelines(lines)
                f.truncate()
            else:
                # File is empty, handle as needed
                logging.warning(f"Text file '{text_file_path}' is empty.")
    except Exception as e:
        logging.error(f"Failed to modify text file '{text_file_path}': {e}")

def main():
    args = parse_arguments()
    setup_logging()

    images_directory = args.images_directory
    texts_directory = args.texts_directory
    prepend_str = args.prepend
    append_str = args.append

    if not os.path.isdir(images_directory):
        logging.error(f"Directory '{images_directory}' does not exist or is not a directory.")
        sys.exit(1)

    if not os.path.isdir(texts_directory):
        logging.error(f"Directory '{texts_directory}' does not exist or is not a directory.")
        sys.exit(1)

    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    files_in_images_dir = os.listdir(images_directory)

    for filename in files_in_images_dir:
        if filename.lower().endswith(image_extensions):
            base_name = os.path.splitext(filename)[0]
            text_file_name = base_name + '.txt'
            text_file_path = os.path.join(texts_directory, text_file_name)
            if os.path.isfile(text_file_path):
                logging.info(f"Modifying text file '{text_file_path}'.")
                modify_text_file(text_file_path, prepend_str, append_str)
            else:
                logging.info(f"No corresponding text file for image '{filename}'.")
        else:
            logging.debug(f"File '{filename}' is not an image. Skipping.")

if __name__ == '__main__':
    main()
