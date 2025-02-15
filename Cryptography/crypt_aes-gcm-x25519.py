#!/usr/bin/env python3
# -------------------------------------------------------
# Script: crypt_aes-gcm-x25519.py
#
# Description:
#   This script implements hybrid encryption using elliptic-curve
#   cryptography (X25519) to securely encrypt a randomly generated
#   AES-GCM key. The AES key encrypts the cleartext (with optional
#   gzip compression), and the AES key itself is encrypted with the
#   recipient's public key using an ephemeral ECDH key exchange and
#   AES-GCM. The full message is stored in a compact JSON format,
#   which is then base64 encoded.
#
# Usage:
#   ./crypt_aes-gcm-x25519.py [options] [command]
#
# Commands:
#   genkey        Generate a new X25519 key pair
#   encrypt       Encrypt data for a recipient using their public key
#   decrypt       Decrypt a message using your private key
#
# Options:
#   -i, --input PATH             Input file path ('-' for stdin). (default: "-")
#   -o, --output PATH            Output file path ('-' for stdout). (default: "-")
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#
# Genkey-specific Options:
#   -s, --priv KEY_PATH          File to write the private key (PEM).
#                                Defaults to "private.pem".
#   -u, --pub KEY_PATH           File to write the public key (PEM).
#                                Defaults to "public.pem".
#   -p, --passphrase TEXT        If provided, the private key will be encrypted.
#
# Encryption-specific Options:
#   -P, --pubkey PATH            Path to recipient's public key (PEM).
#                                (default: "public.pem")
#   -z, --no-gz                  Disable gzip compression (default: compression enabled).
#   -N, --nonce-size N           Nonce size (in bytes) for AES-GCM on data (default: 12).
#   -K, --key-size N             AES key length in bytes (default: 32 = AES-256).
#
# Decryption-specific Options:
#   -P, --privkey PATH           Path to your private key (PEM).
#                                (default: "private.pem")
#   -p, --passphrase TEXT        Passphrase for your private key (if encrypted).
#
# Template: ubuntu22.04
#
# Requirements:
#   - cryptography (install via: pip install cryptography==44.0.1)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import base64
import gzip
import json
import logging
import os
import secrets
import sys
from typing import Optional, Tuple

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid encryption using X25519 and AES-GCM. "
            "Generate key pairs, encrypt data (with an ephemeral AES key wrapped via ECC), "
            "and decrypt messages using your private key."
        )
    )
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

    # genkey sub-command parser
    genkey_parser = subparsers.add_parser(
        "genkey",
        description="Generate a new X25519 key pair (private and public keys).",
        help="Generate key pair.",
    )
    genkey_parser.add_argument(
        "-s",
        "--priv",
        default="private.pem",
        help="Output path for the private key (PEM). Defaults to 'private.pem'. ",
    )
    genkey_parser.add_argument(
        "-u",
        "--pub",
        default="public.pem",
        help="Output path for the public key (PEM). Defaults to 'public.pem'.",
    )
    genkey_parser.add_argument(
        "-p",
        "--passphrase",
        type=str,
        help="If provided, the private key will be encrypted.",
    )

    # encrypt sub-command parser
    encrypt_parser = subparsers.add_parser(
        "encrypt",
        description="Encrypt data using recipient's public key. Uses hybrid encryption.",
        help="Encrypt data.",
    )
    encrypt_parser.add_argument(
        "-P",
        "--pubkey",
        default="public.pem",
        help='Path to recipient\'s public key (PEM). (default: "public.pem")',
    )
    encrypt_parser.add_argument(
        "-z",
        "--no-gz",
        action="store_true",
        help="Disable gzip compression for encryption. (default: compression enabled)",
    )
    encrypt_parser.add_argument(
        "-N",
        "--nonce-size",
        type=int,
        default=12,
        help="Nonce size (in bytes) for AES-GCM on data (default: 12).",
    )
    encrypt_parser.add_argument(
        "-K",
        "--key-size",
        type=int,
        default=32,
        help="AES key length in bytes (default: 32 = AES-256).",
    )

    # decrypt sub-command parser
    decrypt_parser = subparsers.add_parser(
        "decrypt",
        description="Decrypt data encrypted with the 'encrypt' command using your private key.",
        help="Decrypt data.",
    )
    decrypt_parser.add_argument(
        "-P",
        "--privkey",
        default="private.pem",
        help='Path to your private key (PEM). (default: "private.pem")',
    )
    decrypt_parser.add_argument(
        "-p",
        "--passphrase",
        type=str,
        help="Passphrase for your private key (if encrypted).",
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


def scrub_passphrase_in_argv() -> None:
    """
    Overwrites any passphrase in sys.argv to reduce leakage.
    """
    for i, arg in enumerate(sys.argv):
        if arg in ("-p", "--passphrase"):
            if i + 1 < len(sys.argv):
                sys.argv[i + 1] = "********"


def read_input_data(input_path: str) -> bytes:
    """
    Reads data from a file or stdin (binary-safe).
    """
    if input_path == "-":
        logging.debug("Reading input data from stdin.")
        return sys.stdin.buffer.read()
    try:
        logging.debug(f"Reading input file: {input_path}")
        with open(input_path, "rb") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read file '{input_path}': {e}")
        sys.exit(1)


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
            logging.error(f"Failed to write to stdout: {e}")
            sys.exit(1)
    else:
        logging.debug(f"Writing output file: {output_path}")
        try:
            with open(output_path, "wb") as f:
                f.write(data)
        except Exception as e:
            logging.error(f"Failed to write file '{output_path}': {e}")
            sys.exit(1)


def generate_key_pair(passphrase: Optional[str]) -> Tuple[bytes, bytes]:
    """
    Generates an X25519 key pair and returns the private and public key bytes.
    The private key is encrypted if a passphrase is provided.
    """
    logging.info("Generating X25519 key pair.")
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()

    if passphrase:
        enc_algo = serialization.BestAvailableEncryption(passphrase.encode("utf-8"))
        logging.debug("Encrypting private key with provided passphrase.")
    else:
        enc_algo = serialization.NoEncryption()

    priv_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=enc_algo,
    )
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_bytes, pub_bytes


def load_public_key(pub_path: str) -> X25519PublicKey:
    """
    Loads an X25519 public key from a PEM file.
    """
    data = read_input_data(pub_path)
    try:
        key = serialization.load_pem_public_key(data, backend=default_backend())
    except Exception as e:
        logging.error(f"Failed to load public key: {e}")
        sys.exit(1)
    if not isinstance(key, X25519PublicKey):
        logging.error("The provided public key is not a valid X25519 key.")
        sys.exit(1)
    logging.debug("Recipient public key loaded successfully.")
    return key


def load_private_key(priv_path: str, passphrase: Optional[str]) -> X25519PrivateKey:
    """
    Loads an X25519 private key from a PEM file.
    """
    data = read_input_data(priv_path)
    password = passphrase.encode("utf-8") if passphrase else None
    try:
        key = serialization.load_pem_private_key(
            data, password=password, backend=default_backend()
        )
    except Exception as e:
        logging.error(f"Failed to load private key: {e}")
        sys.exit(1)
    if not isinstance(key, X25519PrivateKey):
        logging.error("The provided private key is not a valid X25519 key.")
        sys.exit(1)
    logging.debug("Private key loaded successfully.")
    return key


def encrypt_data(
    plaintext: bytes,
    recipient_pub: X25519PublicKey,
    compress: bool,
    aes_key_size: int,
    data_nonce_size: int,
) -> bytes:
    """
    Encrypts the plaintext using a hybrid scheme:
      1. Optionally compress the plaintext.
      2. Generate a random AES key.
      3. Encrypt the plaintext with AES-GCM.
      4. Generate an ephemeral X25519 key pair and perform ECDH with the recipient's public key.
      5. Derive a key encryption key (KEK) using HKDF.
      6. Encrypt the AES key with AES-GCM using the KEK.
      7. Package both encrypted blobs into a JSON structure and base64 encode it.
    """
    if compress:
        logging.debug("Compressing plaintext with gzip.")
        plaintext = gzip.compress(plaintext)
    else:
        logging.debug("Gzip compression disabled.")

    aes_key = secrets.token_bytes(aes_key_size)
    logging.debug(f"Generated {aes_key_size}-byte AES key for data encryption.")

    data_nonce = secrets.token_bytes(data_nonce_size)
    cipher = Cipher(
        algorithms.AES(aes_key), modes.GCM(data_nonce), backend=default_backend()
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    data_tag = encryptor.tag
    logging.debug("Data encryption with AES-GCM complete.")

    eph_private = X25519PrivateKey.generate()
    eph_public = eph_private.public_key()
    shared_secret = eph_private.exchange(recipient_pub)
    logging.debug("Derived shared secret via ECDH with ephemeral key.")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"ecies key encryption",
        backend=default_backend(),
    )
    kek = hkdf.derive(shared_secret)
    logging.debug("Derived KEK via HKDF.")

    key_nonce = secrets.token_bytes(12)
    key_cipher = Cipher(
        algorithms.AES(kek), modes.GCM(key_nonce), backend=default_backend()
    )
    key_encryptor = key_cipher.encryptor()
    key_ciphertext = key_encryptor.update(aes_key) + key_encryptor.finalize()
    key_tag = key_encryptor.tag
    logging.debug("AES key encrypted with KEK using AES-GCM.")

    output = {
        "d": {
            "n": base64.b64encode(data_nonce).decode("utf-8"),
            "c": base64.b64encode(ciphertext).decode("utf-8"),
            "t": base64.b64encode(data_tag).decode("utf-8"),
            "gz": int(compress),
        },
        "ek": {
            "eph": base64.b64encode(
                eph_public.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            ).decode("utf-8"),
            "n": base64.b64encode(key_nonce).decode("utf-8"),
            "c": base64.b64encode(key_ciphertext).decode("utf-8"),
            "t": base64.b64encode(key_tag).decode("utf-8"),
        },
    }
    json_data = json.dumps(output, separators=(",", ":")).encode("utf-8")
    logging.debug("Hybrid encryption JSON constructed.")
    return base64.b64encode(json_data)


def decrypt_data(enc_data: bytes, recipient_priv: X25519PrivateKey) -> bytes:
    """
    Decrypts the hybrid encrypted message:
      1. Base64-decode and parse the JSON.
      2. From "ek", load the ephemeral public key and decrypt the AES key.
      3. Use the AES key to decrypt the data (from "d").
      4. Decompress if gzip was used.
    """
    try:
        json_data = base64.b64decode(enc_data)
        data_dict = json.loads(json_data.decode("utf-8"))
    except Exception as e:
        logging.error(f"Failed to decode or parse JSON: {e}")
        sys.exit(1)

    try:
        d = data_dict["d"]
        data_nonce = base64.b64decode(d["n"])
        ciphertext = base64.b64decode(d["c"])
        data_tag = base64.b64decode(d["t"])
        gz_flag = int(d.get("gz", 0))
        ek = data_dict["ek"]
        eph_pub_bytes = base64.b64decode(ek["eph"])
        key_nonce = base64.b64decode(ek["n"])
        key_ciphertext = base64.b64decode(ek["c"])
        key_tag = base64.b64decode(ek["t"])
    except KeyError as e:
        logging.error(f"Missing expected field in JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to decode fields from JSON: {e}")
        sys.exit(1)

    try:
        eph_public = X25519PublicKey.from_public_bytes(eph_pub_bytes)
    except Exception as e:
        logging.error(f"Failed to load ephemeral public key: {e}")
        sys.exit(1)

    shared_secret = recipient_priv.exchange(eph_public)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"ecies key encryption",
        backend=default_backend(),
    )
    kek = hkdf.derive(shared_secret)
    logging.debug("Derived KEK via HKDF for key decryption.")

    key_cipher = Cipher(
        algorithms.AES(kek), modes.GCM(key_nonce, key_tag), backend=default_backend()
    )
    key_decryptor = key_cipher.decryptor()
    try:
        aes_key = key_decryptor.update(key_ciphertext) + key_decryptor.finalize()
    except InvalidTag:
        logging.error(
            "Failed to decrypt AES key: authentication error. Wrong private key?"
        )
        sys.exit(1)
    logging.debug("AES key decrypted successfully.")

    data_cipher = Cipher(
        algorithms.AES(aes_key),
        modes.GCM(data_nonce, data_tag),
        backend=default_backend(),
    )
    data_decryptor = data_cipher.decryptor()
    try:
        plaintext = data_decryptor.update(ciphertext) + data_decryptor.finalize()
    except InvalidTag:
        logging.error(
            "Data decryption failed: invalid tag. Corrupted data or wrong key?"
        )
        sys.exit(1)

    if gz_flag:
        logging.debug("Decompressing plaintext with gzip.")
        try:
            plaintext = gzip.decompress(plaintext)
        except Exception as e:
            logging.error(f"Failed to decompress gzip data: {e}")
            sys.exit(1)

    logging.debug("Data decrypted successfully.")
    return plaintext


def main() -> None:
    """
    Main function to handle subcommands: key generation, encryption, or decryption.
    """
    args = parse_arguments()
    scrub_passphrase_in_argv()
    setup_logging(verbose=args.verbose, debug=args.debug)

    if args.command == "genkey":
        # Ensure that the output files do not already exist.
        if os.path.exists(args.priv):
            logging.error(f"File '{args.priv}' already exists. Will not override.")
            sys.exit(1)
        if os.path.exists(args.pub):
            logging.error(f"File '{args.pub}' already exists. Will not override.")
            sys.exit(1)

        priv_bytes, pub_bytes = generate_key_pair(args.passphrase)
        write_output_data(args.priv, priv_bytes)
        try:
            os.chmod(args.priv, 0o600)
            logging.debug(f"Private key file '{args.priv}' permissions set to 0o600.")
        except Exception as e:
            logging.error(f"Failed to set secure permissions on '{args.priv}': {e}")
            sys.exit(1)
        write_output_data(args.pub, pub_bytes)
        logging.info("Key pair generated and saved successfully.")

    elif args.command == "encrypt":
        logging.info("Starting encryption.")
        recipient_pub = load_public_key(args.pubkey)
        input_data = read_input_data(args.input)
        output_data = encrypt_data(
            plaintext=input_data,
            recipient_pub=recipient_pub,
            compress=not args.no_gz,
            aes_key_size=args.key_size,
            data_nonce_size=args.nonce_size,
        )
        write_output_data(args.output, output_data)
        logging.info("Encryption completed successfully.")

    elif args.command == "decrypt":
        logging.info("Starting decryption.")
        recipient_priv = load_private_key(args.privkey, args.passphrase)
        input_data = read_input_data(args.input)
        output_data = decrypt_data(input_data, recipient_priv)
        write_output_data(args.output, output_data)
        logging.info("Decryption completed successfully.")
    else:
        logging.error(f"Unknown subcommand '{args.command}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
