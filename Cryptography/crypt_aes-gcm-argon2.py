#!/usr/bin/env python3
# -------------------------------------------------------
# Script: crypt_aes-gcm-argon2.py
#
# Description:
#   This script provides two sub-commands, `encrypt` and `decrypt`, for
#   encrypting or decrypting data using AES in GCM mode. It derives a 256-bit
#   key from an arbitrary passphrase via Argon2id.
#   The script stores all relevant parameters in a proprietary JSON format.
#
# Usage:
#   ./crypt_aes-gcm-argon2.py [options] [command]
#
# Commands:
#   encrypt       Encrypt data
#   decrypt       Decrypt data
#
# Options:
#   -i, --input PATH             Path to input file. Use "-" for stdin. (default: "-")
#   -o, --output PATH            Output file path. Use "-" for stdout. (default: "-")
#   -p, --passphrase TEXT        Passphrase (any length). If omitted in a TTY,
#                                the script will prompt for it securely.
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#
# Encryption-specific Options:
#   -z, --no-gz                  Disable gzip compression for encryption.
#                                (default: compression enabled)
#   -S, --salt-size N            Number of random bytes for salt (default: 16).
#   -N, --nonce-size N           Number of random bytes for AES-GCM nonce (default: 12).
#   -T, --argon2-time-cost T     Argon2 time cost (default: 3).
#   -M, --argon2-memory-cost M   Argon2 memory cost in KiB (default: 131072 = 128MB).
#   -R, --argon2-parallelism P   Argon2 parallelism (default: 2).
#   -L, --argon2-hash-len L      Argon2 output length in bytes (default: 32).
#
# Template: ubuntu24.04
#
# Requirements:
#   - cryptography (install via: pip install cryptography==44.0.1)
#   - argon2-cffi (install via: pip install argon2-cffi==23.1.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import base64
import getpass
import gzip
import json
import logging
import secrets
import sys
from typing import Optional

from argon2.low_level import hash_secret_raw, Type, ARGON2_VERSION
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.exceptions import InvalidTag


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Encrypt or decrypt data using AES-GCM with a passphrase-derived key "
            "via Argon2id, storing all relevant parameters in a JSON blob."
        )
    )
    # Global arguments (common to both encrypt and decrypt)
    parser.add_argument(
        "-i",
        "--input",
        default="-",
        help="Path to input file or '-' for stdin. (default: '-')",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file path. Use '-' for stdout. (default: '-')",
    )
    parser.add_argument(
        "-p",
        "--passphrase",
        type=str,
        help="Passphrase for Argon2-based key derivation.",
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encrypt sub-command parser (with encryption-specific options)
    encrypt_parser = subparsers.add_parser(
        "encrypt",
        description="Encrypt data using AES-GCM with Argon2id-derived key, output JSON.",
        help="Encrypt data.",
    )
    encrypt_parser.add_argument(
        "-z",
        "--no-gz",
        action="store_true",
        help="Disable gzip compression for encryption. (default: compression enabled)",
    )
    encrypt_parser.add_argument(
        "-S",
        "--salt-size",
        type=int,
        default=16,
        help="Salt size (in bytes) for Argon2. (default: 16)",
    )
    encrypt_parser.add_argument(
        "-N",
        "--nonce-size",
        type=int,
        default=12,
        help="Nonce size (in bytes) for AES-GCM. (default: 12)",
    )
    encrypt_parser.add_argument(
        "-T",
        "--argon2-time-cost",
        type=int,
        default=3,
        help="Argon2 time cost. (default: 3)",
    )
    encrypt_parser.add_argument(
        "-M",
        "--argon2-memory-cost",
        type=int,
        default=131072,
        help="Argon2 memory cost in KiB (default: 131072 = 128MB)",
    )
    encrypt_parser.add_argument(
        "-R",
        "--argon2-parallelism",
        type=int,
        default=2,
        help="Argon2 parallelism. (default: 2)",
    )
    encrypt_parser.add_argument(
        "-L",
        "--argon2-hash-len",
        type=int,
        default=32,
        help="Argon2 output length in bytes. (default: 32)",
    )

    # Decrypt sub-command parser (no extra options)
    subparsers.add_parser(
        "decrypt",
        description="Decrypt JSON output produced by 'encrypt'.",
        help="Decrypt data.",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up logging level based on user arguments.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_passphrase(passphrase_arg: Optional[str]) -> str:
    """
    If a passphrase is provided as an argument, return it. Otherwise,
    prompt the user to input it securely (if in a TTY). If not in a TTY, exit.
    """
    if passphrase_arg is not None:
        logging.debug("Passphrase obtained from command-line argument.")
        return passphrase_arg

    if sys.stdin.isatty():
        logging.debug("Prompting passphrase securely in a TTY.")
        return getpass.getpass("Enter passphrase: ")
    else:
        logging.error("No passphrase provided and not in an interactive terminal.")
        sys.exit(1)


def derive_key_argon2(
    passphrase: str,
    salt: bytes,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
) -> bytes:
    """
    Derives a key from passphrase + salt using Argon2id, returning `hash_len` bytes.
    """
    logging.debug("Deriving key with Argon2id.")
    key = hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=hash_len,
        type=Type.ID,
        version=ARGON2_VERSION,
    )
    return key


def read_input_data(input_path: str) -> bytes:
    """
    Reads all data from a file or stdin (binary-safe).
    """
    if input_path == "-":
        logging.debug("Reading input data from stdin.")
        data = sys.stdin.buffer.read()
    else:
        logging.debug(f"Reading input file: {input_path}")
        try:
            with open(input_path, "rb") as f:
                data = f.read()
        except Exception as e:
            logging.error(f"Failed to read file '{input_path}': {e}")
            sys.exit(1)
    return data


def write_output_data(output_path: str, data: bytes) -> None:
    """
    Writes data to a file or stdout (binary-safe).
    """
    if output_path == "-":
        logging.debug("Writing output data to stdout.")
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
        except Exception as e:
            logging.error(f"Failed to write data to stdout: {e}")
            sys.exit(1)
    else:
        logging.debug(f"Writing output file: {output_path}")
        try:
            with open(output_path, "wb") as f:
                f.write(data)
        except Exception as e:
            logging.error(f"Failed to write file '{output_path}': {e}")
            sys.exit(1)


def encrypt_data(
    plaintext: bytes,
    passphrase: str,
    salt_size: int,
    nonce_size: int,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    hash_len: int,
    compress: bool,
) -> bytes:
    """
    Encrypts plaintext with AES-GCM using a key derived from passphrase + random salt.
    Optionally compresses the plaintext with gzip before encryption.
    """
    # 0. Optionally compress the plaintext
    if compress:
        logging.debug("Compressing plaintext with gzip.")
        plaintext = gzip.compress(plaintext)

    # 1. Generate random salt
    salt = secrets.token_bytes(salt_size)
    logging.debug(f"Generated {salt_size}-byte random salt.")

    # 2. Derive key
    key = derive_key_argon2(
        passphrase=passphrase,
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=hash_len,
    )
    if len(key) < 16:
        logging.error("Derived key is too short. Must be at least 128 bits.")
        sys.exit(1)

    # 3. Generate random nonce
    nonce = secrets.token_bytes(nonce_size)
    logging.debug(f"Generated {nonce_size}-byte random nonce for AES-GCM.")

    # 4. Encrypt
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    tag = encryptor.tag
    logging.debug("AES-GCM encryption complete.")

    # 5. Construct JSON
    output_dict = {
        "a2": {
            "slt": base64.b64encode(salt).decode("utf-8"),
            "tme": time_cost,
            "mem": memory_cost,
            "par": parallelism,
            "len": hash_len,
        },
        "non": base64.b64encode(nonce).decode("utf-8"),
        "cip": base64.b64encode(ciphertext).decode("utf-8"),
        "tag": base64.b64encode(tag).decode("utf-8"),
        "gz": int(compress),
    }

    json_data = json.dumps(output_dict, separators=(",", ":")).encode("utf-8")
    return json_data


def decrypt_data(enc_data: bytes, passphrase: str) -> bytes:
    """
    Decrypts ciphertext from the JSON structure produced by encrypt_data.
    Also decompresses the plaintext if gzip was enabled during encryption.
    """
    # 1. Parse JSON
    try:
        data_dict = json.loads(enc_data.decode("utf-8"))
    except Exception as e:
        logging.error(f"Failed to parse JSON from input: {e}")
        sys.exit(1)

    # 2. Extract Argon2 parameters and other fields
    try:
        argon2_params = data_dict["a2"]
        salt_b64 = argon2_params["slt"]
        time_cost = argon2_params["tme"]
        memory_cost = argon2_params["mem"]
        parallelism = argon2_params["par"]
        hash_len = argon2_params["len"]

        # Base64 decode the salt
        salt = base64.b64decode(salt_b64)

        # Nonce, ciphertext, tag
        nonce = base64.b64decode(data_dict["non"])
        ciphertext = base64.b64decode(data_dict["cip"])
        tag = base64.b64decode(data_dict["tag"])
    except KeyError as e:
        logging.error(f"Missing expected field in JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to decode fields from JSON: {e}")
        sys.exit(1)

    # 3. Derive key using these parameters
    key = derive_key_argon2(
        passphrase=passphrase,
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=hash_len,
    )
    if len(key) < 16:
        logging.error("Derived key is too short. Must be at least 128 bits.")
        sys.exit(1)

    # 4. Decrypt using AES-GCM
    cipher = Cipher(
        algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()
    )
    decryptor = cipher.decryptor()

    try:
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    except InvalidTag:
        logging.error(
            "Authentication failed: invalid tag. Wrong passphrase or corrupted data."
        )
        sys.exit(1)

    # 5. Decompress the plaintext if gzip was used
    if data_dict.get("gz", False):
        logging.debug("Decompressing plaintext with gzip.")
        try:
            plaintext = gzip.decompress(plaintext)
        except Exception as e:
            logging.error(f"Failed to decompress gzip data: {e}")
            sys.exit(1)

    logging.debug("AES-GCM decryption successful.")
    return plaintext


def main() -> None:
    """
    Main function to orchestrate encryption or decryption using subcommands,
    storing salt, Argon2 parameters, nonce, ciphertext, tag, and gzip state in a JSON structure.
    """
    args = parse_arguments()

    setup_logging(verbose=args.verbose, debug=args.debug)
    logging.info(
        "Starting encryption/decryption script with Argon2-based key derivation."
    )

    passphrase = get_passphrase(args.passphrase)
    input_data = read_input_data(args.input)

    if args.command == "encrypt":
        logging.info("Encrypting data (AES-GCM).")
        output_data = encrypt_data(
            plaintext=input_data,
            passphrase=passphrase,
            salt_size=args.salt_size,
            nonce_size=args.nonce_size,
            time_cost=args.argon2_time_cost,
            memory_cost=args.argon2_memory_cost,
            parallelism=args.argon2_parallelism,
            hash_len=args.argon2_hash_len,
            compress=not args.no_gz,
        )
        logging.debug("Encoding JSON blob in base64 for final output.")
        output_data = base64.b64encode(output_data)

    elif args.command == "decrypt":
        logging.info("Decrypting data (AES-GCM).")
        logging.debug("Decoding input from base64 before parsing JSON.")
        try:
            input_data = base64.b64decode(input_data)
        except Exception as e:
            logging.error(f"Failed to decode base64 input: {e}")
            sys.exit(1)

        output_data = decrypt_data(enc_data=input_data, passphrase=passphrase)
    else:
        logging.error(f"Unknown subcommand '{args.command}'.")
        sys.exit(1)

    # Write output
    write_output_data(args.output, output_data)
    logging.info("Operation completed successfully.")


if __name__ == "__main__":
    main()
