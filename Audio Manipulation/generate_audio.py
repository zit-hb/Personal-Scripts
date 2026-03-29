#!/usr/bin/env python3

# -------------------------------------------------------
# Script: generate_audio.py
#
# Description:
# Generates audio from text prompts using Stability AI's Stable Audio Open model
# (stabilityai/stable-audio-open-1.0) via the stable-audio-tools library.
#
# Supports single or multiple prompts, configurable diffusion/sampler settings,
# output normalization, and optional deterministic seeding.
#
# Usage:
#   ./generate_audio.py [options]
#
# Options:
#   -p, --prompt TEXT            Text prompt to generate audio from. Can be used
#                                multiple times.
#   -P, --prompt-file PATH       Path to a UTF-8 text file with prompts (one per
#                                line). Empty lines and lines starting with '#'
#                                are ignored.
#   -o, --output PATH            Output WAV path. (default: output.wav)
#   -m, --model-id MODEL         Hugging Face model id. (default:
#                                stabilityai/stable-audio-open-1.0)
#   -d, --device DEVICE          Device: auto, cpu, cuda, mps. (default: auto)
#   -s, --seconds SECONDS        Total duration in seconds. (default: 10.0)
#   -b, --batch-size N           Number of variations to generate. (default: 1)
#   -S, --seed SEED              Random seed for reproducibility.
#   -n, --steps N                Number of diffusion steps. (default: 100)
#   -c, --cfg-scale SCALE        Classifier-free guidance scale. (default: 7.0)
#   -t, --sampler TYPE           Sampler type. (default: dpmpp-3m-sde)
#   --sigma-min VALUE            Minimum sigma for sampler. (default: 0.3)
#   --sigma-max VALUE            Maximum sigma for sampler. (default: 500.0)
#   --start-seconds SECONDS      Start time in seconds (conditioning). (default:
#                                0.0)
#   --normalize MODE             Normalization: peak, rms, none. (default: peak)
#   --rms-target VALUE           Target RMS for 'rms' normalization in [0, 1].
#                                (default: 0.1)
#   --no-clip                    Do not hard-clip to [-1, 1] before saving.
#   --dtype DTYPE                Model compute dtype: auto, fp16, bf16, fp32.
#                                (default: auto)
#   --sample-rate HZ             Override output sample rate (Hz). Default uses
#                                model's sample rate.
#   --channels N                 Override output channels (1=mono, 2=stereo).
#                                Default uses model output channels.
#   --hf-token TOKEN             Hugging Face access token for gated models.
#                                Can also be set via the HF_TOKEN environment
#                                variable.
#   --overwrite                  Overwrite output file if it exists.
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#
# Template: cuda12.4.1-ubuntu22.04
#
# Requirements:
#   - soundfile (install via: pip install soundfile==0.13.1)
#   - stable-audio-tools (install via: pip install stable-audio-tools==0.0.19 torch==2.5.1 torchaudio==2.5.1)
#
# -------------------------------------------------------
# © 2026 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

# Stabilize numerics — must be set before any CUDA ops.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torchaudio
from einops import rearrange
from huggingface_hub import login as hf_login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


@dataclass(frozen=True)
class ConditioningItem:
    """
    Represents a single conditioning segment for Stable Audio Open generation.
    """

    prompt: str
    seconds_start: float
    seconds_total: float


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate audio from text prompts using Stable Audio Open "
            "(stabilityai/stable-audio-open-1.0)."
        )
    )

    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
        default=[],
        type=str,
        help=(
            "Text prompt to generate audio from. Can be used multiple times. "
            "If multiple prompts are provided, they are combined into conditioning "
            "segments sharing the same timing unless overridden via multiple runs."
        ),
    )
    parser.add_argument(
        "-P",
        "--prompt-file",
        type=str,
        default=None,
        help=(
            "Path to a UTF-8 text file with prompts (one per line). Empty lines and "
            "lines starting with '#' are ignored."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV path. (default: output.wav)",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="stabilityai/stable-audio-open-1.0",
        help="Hugging Face model id. (default: stabilityai/stable-audio-open-1.0)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device: auto, cpu, cuda, mps. (default: auto)",
    )
    parser.add_argument(
        "-s",
        "--seconds",
        type=float,
        default=10.0,
        help="Total duration in seconds. (default: 10.0)",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Start time in seconds (conditioning). (default: 0.0)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Number of variations to generate. (default: 1)",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        default=100,
        help="Number of diffusion steps. (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--cfg-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale. (default: 7.0)",
    )
    parser.add_argument(
        "-t",
        "--sampler",
        type=str,
        default="dpmpp-3m-sde",
        help="Sampler type. (default: dpmpp-3m-sde)",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.3,
        help="Minimum sigma for sampler. (default: 0.3)",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=500.0,
        help="Maximum sigma for sampler. (default: 500.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model compute dtype: auto, fp16, bf16, fp32. (default: auto)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override output sample rate (Hz). Default uses model's sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 2],
        default=None,
        help="Override output channels (1=mono, 2=stereo). Default uses model output.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["peak", "rms", "none"],
        default="peak",
        help="Normalization: peak, rms, none. (default: peak)",
    )
    parser.add_argument(
        "--rms-target",
        type=float,
        default=0.1,
        help="Target RMS for 'rms' normalization in [0, 1]. (default: 0.1)",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not hard-clip to [-1, 1] before saving.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token for gated models. "
            "Can also be set via the HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
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

    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be > 0")

    if args.start_seconds < 0:
        parser.error("--start-seconds must be >= 0")

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")

    if args.steps <= 0:
        parser.error("--steps must be > 0")

    if args.cfg_scale < 0:
        parser.error("--cfg-scale must be >= 0")

    if args.sigma_min <= 0 or args.sigma_max <= 0:
        parser.error("--sigma-min and --sigma-max must be > 0")

    if args.sigma_min >= args.sigma_max:
        parser.error("--sigma-min must be < --sigma-max")

    if not (0.0 < args.rms_target <= 1.0):
        parser.error("--rms-target must be in (0, 1]")

    if args.sample_rate is not None and args.sample_rate <= 0:
        parser.error("--sample-rate must be > 0")

    return args


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


def read_prompts(prompt_list: Sequence[str], prompt_file: Optional[str]) -> List[str]:
    """
    Reads prompts from --prompt and optionally from --prompt-file.
    """
    prompts: List[str] = [p.strip() for p in prompt_list if p.strip()]

    if prompt_file is None:
        return prompts

    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prompts.append(stripped)

    return prompts


def resolve_device(device_argument: str) -> str:
    """
    Resolves the user-provided device argument to a concrete device string.
    """
    if device_argument == "cpu":
        return "cpu"
    if device_argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device 'cuda' requested, but CUDA is not available.")
        return "cuda"
    if device_argument == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("Device 'mps' requested, but MPS is not available.")
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_argument: str, device: str) -> torch.dtype:
    """
    Resolves dtype choice for inference.
    """
    if dtype_argument == "fp32":
        return torch.float32
    if dtype_argument == "fp16":
        return torch.float16
    if dtype_argument == "bf16":
        return torch.bfloat16

    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def set_deterministic_seed(seed: int, device: str) -> None:
    """
    Sets random seeds for reproducible generation (best-effort).
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def authenticate_huggingface(token: Optional[str]) -> None:
    """
    Authenticates with Hugging Face using the provided token or HF_TOKEN env var.
    Raises RuntimeError if no token is available.
    """
    resolved_token = token or os.environ.get("HF_TOKEN")
    if not resolved_token:
        raise RuntimeError(
            "A Hugging Face access token is required to download gated models. "
            "Provide it via --hf-token TOKEN or the HF_TOKEN environment variable. "
            "You can create a token at https://huggingface.co/settings/tokens"
        )
    hf_login(token=resolved_token, add_to_git_credential=False)
    logging.info("Authenticated with Hugging Face.")


def build_conditioning(
    prompts: Sequence[str],
    seconds_start: float,
    seconds_total: float,
) -> List[dict]:
    """
    Builds Stable Audio Open conditioning list.
    """
    conditioning_items: List[dict] = []
    for prompt in prompts:
        conditioning_items.append(
            {
                "prompt": prompt,
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
            }
        )
    return conditioning_items


def ensure_output_path(output_path: str, overwrite: bool) -> Path:
    """
    Validates and prepares the output path.
    """
    path = Path(output_path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(f"Output directory is not a directory: {parent}")

    return path


def generate_audio_tensor(
    model: torch.nn.Module,
    sample_size: int,
    device: str,
    steps: int,
    cfg_scale: float,
    conditioning: List[dict],
    sigma_min: float,
    sigma_max: float,
    sampler_type: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Runs diffusion generation and returns audio tensor of shape [batch, channels, samples].

    Uses torch.no_grad() + disabled AMP (matching the stable reference implementation)
    instead of torch.inference_mode(), which can interfere with the diffusion sampler's
    internal control flow and produce silent or corrupted output.
    """
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device if device != "cpu" else "cpu", enabled=False
        ):
            audio: torch.Tensor = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=device,
                batch_size=batch_size,
            )
    return audio


def join_batch(audio_batch: torch.Tensor) -> torch.Tensor:
    """
    Rearranges [batch, channels, samples] into [channels, batch * samples].
    """
    if audio_batch.ndim != 3:
        raise ValueError(
            f"Expected a 3D tensor [b, c, n], got shape {tuple(audio_batch.shape)}"
        )
    return rearrange(audio_batch, "b c n -> c (b n)")


def convert_channels(audio: torch.Tensor, channels: Optional[int]) -> torch.Tensor:
    """
    Converts channel count to mono or stereo if requested.
    """
    if channels is None:
        return audio

    if audio.ndim != 2:
        raise ValueError("Audio tensor must be 2D [channels, samples].")

    current_channels = audio.shape[0]
    if current_channels == channels:
        return audio

    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    if channels == 2:
        if current_channels == 1:
            return audio.repeat(2, 1)
        if current_channels > 2:
            return audio[:2, :]
    raise ValueError(
        f"Unsupported channel conversion: {current_channels} -> {channels}"
    )


def resample_audio(
    audio: torch.Tensor, original_rate: int, target_rate: Optional[int]
) -> Tuple[torch.Tensor, int]:
    """
    Resamples audio to target_rate if provided.
    """
    if target_rate is None or target_rate == original_rate:
        return audio, original_rate

    if audio.ndim != 2:
        raise ValueError("Audio tensor must be 2D [channels, samples].")

    resampler = torchaudio.transforms.Resample(
        orig_freq=original_rate, new_freq=target_rate
    )
    resampled = resampler(audio.to(torch.float32))
    return resampled, target_rate


def peak_normalize(audio: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Peak-normalizes audio so max abs value becomes 1.0 (if non-silent).
    """
    peak = torch.max(torch.abs(audio)).clamp_min(epsilon)
    return audio / peak


def rms_normalize(
    audio: torch.Tensor, target_rms: float, epsilon: float = 1e-12
) -> torch.Tensor:
    """
    RMS-normalizes audio to a target RMS in (0, 1].
    """
    rms = torch.sqrt(torch.mean(audio**2)).clamp_min(epsilon)
    gain = float(target_rms) / float(rms.item())
    return audio * gain


def finalize_audio_for_wav(
    audio: torch.Tensor,
    normalize_mode: str,
    rms_target: float,
    clip: bool,
) -> Tuple[np.ndarray, None]:
    """
    Normalizes/clips and converts to a float32 numpy array of shape
    [samples, channels] suitable for soundfile.write.

    Saves as float32 (PCM_32) rather than int16 to avoid quantization
    artefacts from the int16 conversion path.
    """
    audio_fp32 = audio.to(torch.float32)

    if normalize_mode == "peak":
        audio_fp32 = peak_normalize(audio_fp32)
    elif normalize_mode == "rms":
        audio_fp32 = rms_normalize(audio_fp32, target_rms=rms_target)
    elif normalize_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {normalize_mode}")

    if clip:
        audio_fp32 = audio_fp32.clamp(-1.0, 1.0)

    # soundfile expects [samples, channels]
    audio_numpy = audio_fp32.cpu().numpy()  # [channels, samples]
    audio_numpy = np.transpose(audio_numpy, (1, 0))  # [samples, channels]
    return audio_numpy


def format_conditioning_summary(conditioning: Sequence[dict]) -> str:
    """
    Formats conditioning list for logs.
    """
    lines: List[str] = []
    for index, item in enumerate(conditioning, start=1):
        prompt = str(item.get("prompt", ""))
        seconds_start = float(item.get("seconds_start", 0.0))
        seconds_total = float(item.get("seconds_total", 0.0))
        lines.append(
            f"{index}. start={seconds_start:.3f}s total={seconds_total:.3f}s prompt={prompt!r}"
        )
    return "\n".join(lines)


def main() -> None:
    """
    Main function to orchestrate the audio generation process.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    try:
        output_path = ensure_output_path(args.output, overwrite=args.overwrite)
        prompts = read_prompts(args.prompt, args.prompt_file)
        if not prompts:
            raise ValueError(
                "No prompts provided. Use --prompt and/or --prompt-file to specify at least one prompt."
            )

        authenticate_huggingface(args.hf_token)

        device = resolve_device(args.device)
        dtype = resolve_dtype(args.dtype, device=device)

        if args.seed is not None:
            set_deterministic_seed(args.seed, device=device)

        logging.info(f"Using device: {device}")
        logging.info(f"Using dtype: {dtype}")
        logging.info(f"Loading model: {args.model_id}")

        model, model_config = get_pretrained_model(args.model_id)
        # eval() is required — BatchNorm/Dropout layers behave differently in
        # training mode and can produce corrupted output during generation.
        model = model.to(device=device, dtype=dtype).eval()

        model_sample_rate = int(model_config["sample_rate"])
        model_sample_size = int(model_config["sample_size"])

        seconds_total = float(args.seconds)
        target_num_samples = int(round(seconds_total * model_sample_rate))
        sample_size = min(model_sample_size, target_num_samples)

        if target_num_samples > model_sample_size:
            logging.warning(
                "Requested duration exceeds model's maximum. "
                f"Requested samples={target_num_samples}, max samples={model_sample_size}. "
                "Clamping to model maximum duration."
            )
            sample_size = model_sample_size
            seconds_total = float(sample_size) / float(model_sample_rate)

        conditioning = build_conditioning(
            prompts=prompts,
            seconds_start=float(args.start_seconds),
            seconds_total=seconds_total,
        )

        logging.info("Conditioning:\n" + format_conditioning_summary(conditioning))
        logging.info(
            "Generation settings: "
            f"steps={args.steps}, cfg_scale={args.cfg_scale}, sampler={args.sampler}, "
            f"sigma_min={args.sigma_min}, sigma_max={args.sigma_max}, "
            f"batch_size={args.batch_size}, seconds={seconds_total:.3f}"
        )

        audio_batch = generate_audio_tensor(
            model=model,
            sample_size=sample_size,
            device=device,
            steps=int(args.steps),
            cfg_scale=float(args.cfg_scale),
            conditioning=conditioning,
            sigma_min=float(args.sigma_min),
            sigma_max=float(args.sigma_max),
            sampler_type=str(args.sampler),
            batch_size=int(args.batch_size),
        )

        audio = join_batch(audio_batch)
        audio = convert_channels(audio, channels=args.channels)

        audio, output_rate = resample_audio(
            audio=audio,
            original_rate=model_sample_rate,
            target_rate=args.sample_rate,
        )

        audio_numpy = finalize_audio_for_wav(
            audio=audio,
            normalize_mode=str(args.normalize),
            rms_target=float(args.rms_target),
            clip=not bool(args.no_clip),
        )

        sf.write(str(output_path), audio_numpy, output_rate, subtype="PCM_32")
        logging.warning(
            f"Saved: {output_path} (sr={output_rate}, shape={audio_numpy.shape})"
        )

    except KeyboardInterrupt:
        logging.error("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        logging.error(str(exc))
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
