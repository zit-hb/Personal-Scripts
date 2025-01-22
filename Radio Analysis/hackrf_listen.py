#!/usr/bin/env python3

# -------------------------------------------------------
# Script: hackrf_listen.py
#
# Description:
# A HackRF script to tune to a specified frequency
# and demodulate it in one of several modes:
#
#   - AM  (Amplitude Modulation, typical ~520 kHz - 1710 kHz)
#   - FM  (Wide FM, typical broadcast band ~87.5 MHz - 108 MHz)
#   - NFM (Narrowband FM)
#   - AIR (Aircraft VHF AM, typically 118 - 136 MHz)
#   - HF  (Shortwave / HF AM Broadcast, ~2 MHz - 26 MHz commonly)
#   - CB  (Citizen's Band, ~27 MHz, AM)
#
# Depending on the selected mode, the script sets
# appropriate defaults and DSP routines to demodulate
# the signal, writing the resulting audio to an OGG Vorbis file.
#
# Usage:
#   ./hackrf_listen.py [options]
#
# Options (common to all modes):
#   -m, --mode {am,fm,nfm,air,hf,cb}     Select demodulation mode.
#   -f, --freq FREQ_HZ                   Frequency in Hz (e.g. 1e6 for 1 MHz).
#   -s, --sample-rate SR                 HackRF sample rate in Hz (mode-specific default if not provided).
#   -b, --bandwidth BW                   Baseband filter bandwidth in Hz (mode-specific default if not provided).
#   -a, --audio-rate AR                  Output audio sample rate in Hz (default: 48000).
#   -d, --duration SEC                   Duration in seconds (0 => indefinite).
#   -o, --output FILE                    Output .ogg file name (default: recording.ogg).
#   -I, --device-index IDX               HackRF device index (default: 0).
#   -L, --lna-gain DB                    LNA gain in dB [0..40 in steps of 8] (default: 16).
#   -G, --vga-gain DB                    VGA gain in dB [0..62 in steps of 2] (default: 20).
#   -P, --amp                            Enable HackRF internal amplifier (default: off).
#   -F, --auto-scan                      Attempt to automatically find the "best" frequency for the chosen mode.
#   -S, --scan-start FREQ_HZ             Start frequency (Hz) for scanning (default depends on mode).
#   -T, --scan-stop FREQ_HZ              Stop frequency (Hz) for scanning (default depends on mode).
#   -p, --scan-step FREQ_HZ              Step in Hz for scanning (default depends on mode).
#   -W, --fm-deviation                   Deviation in Hz for wide-FM demod (default: 75000).
#   -N, --nfm-deviation                  Deviation in Hz for narrow-FM demod (default: 5000).
#   -v, --verbose                        Increase verbosity (-v => INFO, -vv => DEBUG).
#
# IMPORTANT:
#   - This script is for demonstration and educational use only.
#   - Always ensure you comply with local regulations when listening
#     to and/or recording any signals.
#
# Template: ubuntu22.04
#
# Requirements:
#   - libhackrf (install via: apt-get install -y libhackrf0 libusb-1.0-0-dev)
#   - numpy (install via: pip install numpy==2.2.2)
#   - scipy (install via: pip install scipy==1.15.1)
#   - soundfile (install via: pip install soundfile==0.13.0)
#   - libsndfile1 (install via: apt-get install -y libsndfile1)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import time
import math
import numpy as np
import scipy.signal
import soundfile as sf
import ctypes
import atexit
import threading
from typing import Dict, Any, Optional

_libhackrf = ctypes.CDLL('libhackrf.so.0')

HACKRF_SUCCESS = 0
_p_hackrf_device = ctypes.c_void_p


class _hackrf_transfer(ctypes.Structure):
    _fields_ = [
        ("device", _p_hackrf_device),
        ("buffer", ctypes.POINTER(ctypes.c_byte)),
        ("buffer_length", ctypes.c_int),
        ("valid_length", ctypes.c_int),
        ("rx_ctx", ctypes.c_void_p),
        ("tx_ctx", ctypes.c_void_p),
    ]


_transfer_callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_hackrf_transfer))

_libhackrf.hackrf_init.restype = ctypes.c_int
_libhackrf.hackrf_init.argtypes = []

_libhackrf.hackrf_exit.restype = ctypes.c_int
_libhackrf.hackrf_exit.argtypes = []

_libhackrf.hackrf_device_list.restype = ctypes.c_void_p
_libhackrf.hackrf_device_list.argtypes = []

_libhackrf.hackrf_device_list_open.restype = ctypes.c_int
_libhackrf.hackrf_device_list_open.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(_p_hackrf_device),
]

_libhackrf.hackrf_close.restype = ctypes.c_int
_libhackrf.hackrf_close.argtypes = [_p_hackrf_device]

_libhackrf.hackrf_set_sample_rate.restype = ctypes.c_int
_libhackrf.hackrf_set_sample_rate.argtypes = [_p_hackrf_device, ctypes.c_double]

_libhackrf.hackrf_set_baseband_filter_bandwidth.restype = ctypes.c_int
_libhackrf.hackrf_set_baseband_filter_bandwidth.argtypes = [_p_hackrf_device, ctypes.c_uint32]

_libhackrf.hackrf_set_amp_enable.restype = ctypes.c_int
_libhackrf.hackrf_set_amp_enable.argtypes = [_p_hackrf_device, ctypes.c_uint8]

_libhackrf.hackrf_set_lna_gain.restype = ctypes.c_int
_libhackrf.hackrf_set_lna_gain.argtypes = [_p_hackrf_device, ctypes.c_uint32]

_libhackrf.hackrf_set_vga_gain.restype = ctypes.c_int
_libhackrf.hackrf_set_vga_gain.argtypes = [_p_hackrf_device, ctypes.c_uint32]

_libhackrf.hackrf_set_freq.restype = ctypes.c_int
_libhackrf.hackrf_set_freq.argtypes = [_p_hackrf_device, ctypes.c_uint64]

_libhackrf.hackrf_start_rx.restype = ctypes.c_int
_libhackrf.hackrf_start_rx.argtypes = [_p_hackrf_device, _transfer_callback, ctypes.c_void_p]

_libhackrf.hackrf_stop_rx.restype = ctypes.c_int
_libhackrf.hackrf_stop_rx.argtypes = [_p_hackrf_device]

_libhackrf.hackrf_is_streaming.restype = ctypes.c_int
_libhackrf.hackrf_is_streaming.argtypes = [_p_hackrf_device]


_init_result = _libhackrf.hackrf_init()
if _init_result != HACKRF_SUCCESS:
    raise RuntimeError("Error initializing HackRF (libhackrf).")


def _finalize_hackrf() -> None:
    """
    Finalizes the HackRF library on program exit.
    """
    _libhackrf.hackrf_exit()


atexit.register(_finalize_hackrf)


def validate_frequency(freq: float) -> None:
    """
    Warns if the frequency is outside the typical HackRF range (~1 MHz..6 GHz).
    """
    if freq < 1e6:
        logging.warning(
            f"HackRF typically cannot tune below 1 MHz. "
            f"You requested {freq/1e6:.3f} MHz. Proceeding, but you may need an upconverter or a modified HackRF."
        )
    elif freq > 6e9:
        logging.warning(
            f"HackRF typically cannot tune above 6 GHz. "
            f"You requested {freq/1e6:.3f} MHz. Proceeding, but this may not work."
        )


def validate_sample_rate(sample_rate: float) -> None:
    """
    Warns if the provided sample rate is outside the typical HackRF range (~2 MHz..20 MHz).
    """
    if sample_rate < 2e6 or sample_rate > 20e6:
        logging.warning(
            f"Sample rate {sample_rate} Hz is outside the typical HackRF range (2 MHz..20 MHz). "
            f"Proceeding, but it may not work optimally."
        )


def get_valid_bandwidth_for_am(desired_bw: float) -> int:
    """
    Returns a valid HackRF baseband filter bandwidth for AM-like modes.
    """
    valid_bw = [150000, 200000, 300000, 500000, 1500000, 2400000, 2800000]
    if desired_bw <= valid_bw[0]:
        return valid_bw[0]
    if desired_bw >= valid_bw[-1]:
        return valid_bw[-1]
    return int(min(valid_bw, key=lambda x: abs(x - desired_bw)))


def get_valid_bandwidth_for_fm(desired_bw: float) -> int:
    """
    Returns a valid HackRF baseband filter bandwidth for wide FM-like modes.
    """
    valid_bw = [
        1750000, 2000000, 2500000, 5000000, 5500000, 6000000,
        7500000, 10000000, 15000000, 20000000, 24000000, 28000000
    ]
    if desired_bw <= valid_bw[0]:
        return valid_bw[0]
    if desired_bw >= valid_bw[-1]:
        return valid_bw[-1]
    return int(min(valid_bw, key=lambda x: abs(x - desired_bw)))


def get_valid_bandwidth_for_nfm(desired_bw: float) -> int:
    """
    Returns a valid HackRF baseband filter bandwidth for narrow FM-like modes.
    """
    valid_bw = [150000, 200000, 300000, 500000, 1500000, 2400000, 2800000]
    if desired_bw <= valid_bw[0]:
        return valid_bw[0]
    if desired_bw >= valid_bw[-1]:
        return valid_bw[-1]
    return int(min(valid_bw, key=lambda x: abs(x - desired_bw)))


def get_valid_bandwidth_for_air(desired_bw: float) -> int:
    """
    Returns a valid HackRF baseband filter bandwidth for aircraft AM-like modes.
    """
    valid_bw = [150000, 200000, 300000, 500000]
    if desired_bw <= valid_bw[0]:
        return valid_bw[0]
    if desired_bw >= valid_bw[-1]:
        return valid_bw[-1]
    return int(min(valid_bw, key=lambda x: abs(x - desired_bw)))


def clamp_lna_gain(gain: int) -> int:
    """
    Clamps the LNA gain to valid HackRF steps (0, 8, 16, 24, 32, 40).
    """
    valid_lna = [0, 8, 16, 24, 32, 40]
    clamped = min(valid_lna, key=lambda x: abs(x - gain))
    if clamped != gain:
        logging.warning(f"Requested LNA gain {gain} dB is invalid for HackRF; using {clamped} dB instead.")
    return clamped


def clamp_vga_gain(gain: int) -> int:
    """
    Clamps the VGA gain to valid HackRF steps (0..62 in steps of 2).
    """
    valid_vga = list(range(0, 63, 2))
    clamped = min(valid_vga, key=lambda x: abs(x - gain))
    if clamped != gain:
        logging.warning(f"Requested VGA gain {gain} dB is invalid for HackRF; using {clamped} dB instead.")
    return clamped


class HackRFDevice:
    """
    Represents a HackRF device, handling opening, configuration, and sample streaming.
    """

    def __init__(self, device_index: int = 0) -> None:
        devlist_handle = _libhackrf.hackrf_device_list()
        if not devlist_handle:
            raise RuntimeError("No HackRF devices found (device list is NULL).")

        self._dev = _p_hackrf_device(None)
        result = _libhackrf.hackrf_device_list_open(
            devlist_handle, device_index, ctypes.byref(self._dev)
        )
        if result != HACKRF_SUCCESS or not self._dev:
            raise RuntimeError(
                f"Could not open HackRF device idx={device_index}, code={result}."
            )

        # Tracking streaming state
        self._streaming = False

        # Lock/condition for thread-safe buffer access
        self._rx_buffer = bytearray()
        self._rx_buffer_lock = threading.Lock()
        self._rx_buffer_cond = threading.Condition(self._rx_buffer_lock)

        # HackRF callback must remain alive
        self._rx_callback_function = _transfer_callback(self._rx_callback)

        # Used to limit number of bytes for read_samples() if specified
        self._target_bytes = 0

    def close(self) -> None:
        """
        Closes the HackRF device if open, stopping any active streams first.
        """
        if self._dev:
            if self._streaming:
                _libhackrf.hackrf_stop_rx(self._dev)
                self._streaming = False
            _libhackrf.hackrf_close(self._dev)
            self._dev = _p_hackrf_device(None)

    def __del__(self) -> None:
        """
        Destructor ensures the device is closed.
        """
        self.close()

    def set_sample_rate(self, sr_hz: float) -> None:
        """
        Sets the HackRF device sample rate in Hz.
        """
        sr = ctypes.c_double(sr_hz)
        r = _libhackrf.hackrf_set_sample_rate(self._dev, sr)
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting sample rate to {sr_hz} Hz, code={r}")

    def set_baseband_filter_bandwidth(self, bw_hz: float, mode: str) -> None:
        """
        Selects a valid HackRF filter bandwidth for the given mode, then sets it on the device.
        """
        if mode in ['am', 'hf', 'cb']:
            valid_bw = get_valid_bandwidth_for_am(bw_hz)
        elif mode == 'fm':
            valid_bw = get_valid_bandwidth_for_fm(bw_hz)
        elif mode == 'nfm':
            valid_bw = get_valid_bandwidth_for_nfm(bw_hz)
        elif mode == 'air':
            valid_bw = get_valid_bandwidth_for_air(bw_hz)
        else:
            raise RuntimeError(f"Unknown mode for bandwidth selection: {mode}")

        if abs(valid_bw - bw_hz) > 1:
            logging.warning(
                f"Requested bandwidth {bw_hz} Hz adjusted to {valid_bw} Hz for mode '{mode}'."
            )

        r = _libhackrf.hackrf_set_baseband_filter_bandwidth(
            self._dev, ctypes.c_uint32(valid_bw)
        )
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting baseband filter BW to {valid_bw} Hz, code={r}")

    def set_freq(self, freq_hz: float) -> None:
        """
        Sets the HackRF center frequency in Hz.
        """
        r = _libhackrf.hackrf_set_freq(self._dev, ctypes.c_uint64(int(freq_hz)))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting frequency to {freq_hz} Hz, code={r}")

    def set_amp_enable(self, enable: bool) -> None:
        """
        Enables (True) or disables (False) the HackRF internal amplifier.
        """
        val = 1 if enable else 0
        r = _libhackrf.hackrf_set_amp_enable(self._dev, ctypes.c_uint8(val))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting amp enable={enable}, code={r}")

    def set_lna_gain(self, gain: int) -> None:
        """
        Sets the LNA gain in dB, clamping to valid HackRF steps.
        """
        gain_clamped = clamp_lna_gain(gain)
        r = _libhackrf.hackrf_set_lna_gain(self._dev, ctypes.c_uint32(gain_clamped))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting LNA gain={gain_clamped}, code={r}")

    def set_vga_gain(self, gain: int) -> None:
        """
        Sets the VGA gain in dB, clamping to valid HackRF steps.
        """
        gain_clamped = clamp_vga_gain(gain)
        r = _libhackrf.hackrf_set_vga_gain(self._dev, ctypes.c_uint32(gain_clamped))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting VGA gain={gain_clamped}, code={r}")

    def _rx_callback(self, transfer_ptr: ctypes.POINTER(_hackrf_transfer)) -> int:
        """
        HackRF RX callback; accumulates raw data into self._rx_buffer,
        stops streaming if the desired amount is reached.
        """
        transfer = transfer_ptr.contents
        buf_length = transfer.valid_length
        if buf_length > 0:
            data_array = ctypes.cast(
                transfer.buffer, ctypes.POINTER(ctypes.c_byte * buf_length)
            ).contents

            with self._rx_buffer_cond:
                self._rx_buffer.extend(data_array)

                # If we have a target, stop if it's reached
                ret = 0
                if self._target_bytes > 0 and len(self._rx_buffer) >= self._target_bytes:
                    ret = -1
                    self._streaming = False
                self._rx_buffer_cond.notify_all()
            return ret

        return 0

    def read_samples(self, num_samples: int = 0, duration: float = 0.0, chunk_size: int = 8192):
        """
        Generator that reads samples from the HackRF in thread-safe chunks.
        Yields chunks of raw IQ bytes in multiples of chunk_size*2.
        If num_samples > 0, stops after reading those samples total.
        If duration > 0, also stops after that time has elapsed.
        """
        if self._streaming:
            raise RuntimeError("Already streaming. Stop first or use a new device.")
        self._rx_buffer = bytearray()

        # If user wants a specific sample count, set a target in bytes
        if num_samples > 0:
            self._target_bytes = 2 * num_samples
        else:
            self._target_bytes = 0

        r = _libhackrf.hackrf_start_rx(self._dev, self._rx_callback_function, None)
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error starting RX, code={r}")

        self._streaming = True
        start_time = time.time()

        try:
            while True:
                with self._rx_buffer_cond:
                    # Wait for enough data or until streaming stops/duration reached
                    while len(self._rx_buffer) < chunk_size * 2 and self._streaming:
                        if duration > 0 and (time.time() - start_time) >= duration:
                            break
                        self._rx_buffer_cond.wait(timeout=0.1)

                    # Check conditions to exit
                    if not self._streaming:
                        break
                    if duration > 0 and (time.time() - start_time) >= duration:
                        break

                    if len(self._rx_buffer) >= chunk_size * 2:
                        out = self._rx_buffer[:chunk_size * 2]
                        del self._rx_buffer[:chunk_size * 2]
                    else:
                        # Not enough data or streaming ended
                        break

                yield out

            # Drain remaining buffer after stopping or time out
            while True:
                with self._rx_buffer_cond:
                    if len(self._rx_buffer) == 0:
                        break
                    needed = min(len(self._rx_buffer), chunk_size * 2)
                    out = self._rx_buffer[:needed]
                    del self._rx_buffer[:needed]
                yield out

        finally:
            _libhackrf.hackrf_stop_rx(self._dev)
            self._streaming = False


def _dc_block_singlepole(
    samples: np.ndarray,
    state: Dict[str, float],
    sample_rate: float,
    cutoff: float = 30.0
) -> np.ndarray:
    """
    Applies a single-pole DC blocking filter to 'samples' using a
    frequency-dependent 'alpha' parameter derived from the provided
    'cutoff' (in Hz) and 'sample_rate' (in samples/sec).

    The filter state is carried in 'state' for continuous streaming.

    By default, cutoff=30 Hz is used, which is typical for AM/FM audio
    to remove sub-audio DC offsets. Adjust as needed for other bandwidths.
    """
    if "prev_x" not in state:
        state["prev_x"] = 0.0
    if "prev_y" not in state:
        state["prev_y"] = 0.0

    # Compute alpha based on the desired cutoff and the current sample rate.
    # alpha = exp(-2*pi * cutoff / sample_rate)
    alpha = math.exp(-2.0 * math.pi * cutoff / sample_rate)

    prev_x = state["prev_x"]
    prev_y = state["prev_y"]

    out = np.zeros_like(samples, dtype=np.float32)
    for i, x_val in enumerate(samples):
        y_val = (x_val - prev_x) + alpha * prev_y
        out[i] = y_val
        prev_x = x_val
        prev_y = y_val

    state["prev_x"] = prev_x
    state["prev_y"] = prev_y
    return out


def _resample_if_needed(audio: np.ndarray, current_rate: float, target_rate: float) -> np.ndarray:
    """
    Resamples 'audio' from current_rate to target_rate using scipy.signal.resample_poly.
    Returns the new audio array.
    """
    if abs(current_rate - target_rate) < 1e-3:
        return audio
    from math import gcd
    sr_int = int(round(current_rate))
    ar_int = int(round(target_rate))
    if sr_int <= 0 or ar_int <= 0:
        return audio
    g = gcd(sr_int, ar_int)
    up = ar_int // g
    down = sr_int // g
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def _lpf_resample_cplx(iq: np.ndarray, sr_in: float, sr_out: float, cutoff: float, state: Dict[str, Any]) -> (np.ndarray, float):
    """
    Applies a low-pass FIR filter to complex samples (iq), then resamples to sr_out.
    The filter states are maintained in 'state'.
    Returns the filtered and resampled complex samples, plus the new sample rate.
    """
    if "lpf_resample_taps" not in state:
        norm_cutoff = cutoff / (sr_in / 2.0)
        num_taps = 129
        fir = scipy.signal.firwin(num_taps, norm_cutoff)
        state["lpf_resample_taps"] = fir
        state["lpf_resample_zi_i"] = np.zeros(len(fir) - 1, dtype=np.float32)
        state["lpf_resample_zi_q"] = np.zeros(len(fir) - 1, dtype=np.float32)

    fir = state["lpf_resample_taps"]

    i_samp = np.real(iq).astype(np.float32)
    q_samp = np.imag(iq).astype(np.float32)

    out_i, state["lpf_resample_zi_i"] = scipy.signal.lfilter(
        fir, [1.0], i_samp, zi=state["lpf_resample_zi_i"]
    )
    out_q, state["lpf_resample_zi_q"] = scipy.signal.lfilter(
        fir, [1.0], q_samp, zi=state["lpf_resample_zi_q"]
    )

    ratio_i = _resample_if_needed(out_i, sr_in, sr_out)
    ratio_q = _resample_if_needed(out_q, sr_in, sr_out)

    return ratio_i + 1j * ratio_q, sr_out


def demodulate_am(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sr: float,
    ar: float,
    cutoff: float = 5000.0
) -> np.ndarray:
    """
    Demodulates AM via envelope detection, then low-pass filtering,
    DC block (adjusted by the sample rate), and resampling.
    """
    envelope = np.abs(iq_chunk)

    if "lpf_taps" not in state:
        num_taps = 129
        fir = scipy.signal.firwin(num_taps, cutoff / (sr / 2.0))
        state["lpf_taps"] = fir
        state["lpf_zi"] = np.zeros(len(fir) - 1, dtype=np.float32)

    filtered, state["lpf_zi"] = scipy.signal.lfilter(
        state["lpf_taps"], [1.0], envelope, zi=state["lpf_zi"]
    )

    # Apply a proper single-pole DC block that depends on the current sample rate sr.
    if "dc_block" not in state:
        state["dc_block"] = {}
    dc_blocked = _dc_block_singlepole(filtered, state["dc_block"], sr, 30.0)

    audio = _resample_if_needed(dc_blocked, sr, ar)
    return audio


def demodulate_wfm(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sr: float,
    ar: float,
    if_cutoff: float = 100000.0,
    audio_cutoff: float = 15000.0,
    deemphasis: float = 7.5e-5,
    fm_dev: float = 75000.0
) -> np.ndarray:
    """
    Demodulates wide FM by downsampling IQ, phase demod, audio filtering,
    frequency-dependent DC block, de-emphasis, and final resampling.
    """
    inter_sr = 200000.0
    if "pre_downsample" not in state:
        state["pre_downsample"] = {}
    iq_ds, sr_ds = _lpf_resample_cplx(iq_chunk, sr, inter_sr, if_cutoff, state["pre_downsample"])

    if "prev_iq" not in state:
        state["prev_iq"] = np.complex64(0.0)

    combined = np.concatenate(([state["prev_iq"]], iq_ds))
    state["prev_iq"] = iq_ds[-1] if len(iq_ds) else state["prev_iq"]
    phase = np.angle(combined[1:] * np.conjugate(combined[:-1]))

    phase *= sr_ds / (2.0 * np.pi * fm_dev)

    if "dc_block" not in state:
        state["dc_block"] = {}
    freq_dem = _dc_block_singlepole(phase, state["dc_block"], sr_ds, 30.0)

    if "lpf_audio_taps" not in state:
        num_taps2 = 129
        cutoff2 = audio_cutoff / (sr_ds / 2.0)
        fir2 = scipy.signal.firwin(num_taps2, cutoff2)
        state["lpf_audio_taps"] = fir2
        state["lpf_audio_zi"] = np.zeros(len(fir2) - 1, dtype=np.float32)

    filtered2, state["lpf_audio_zi"] = scipy.signal.lfilter(
        state["lpf_audio_taps"], [1.0], freq_dem, zi=state["lpf_audio_zi"]
    )

    if "deemph_y" not in state:
        state["deemph_y"] = 0.0

    out_deemph = np.zeros_like(filtered2, dtype=np.float32)
    y_prev = state["deemph_y"]

    a = math.exp(-1.0 / (sr_ds * deemphasis))
    for i, x_val in enumerate(filtered2):
        y_val = a * y_prev + (1.0 - a) * x_val
        out_deemph[i] = y_val
        y_prev = y_val

    state["deemph_y"] = y_prev

    audio = _resample_if_needed(out_deemph, sr_ds, ar)
    return audio


def demodulate_nfm(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sr: float,
    ar: float,
    if_cutoff: float = 20000.0,
    nfm_dev: float = 5000.0
) -> np.ndarray:
    """
    Demodulates narrow FM by downsampling IQ, phase demod, voice low-pass,
    frequency-dependent DC block, and resampling to audio rate.
    """
    inter_sr = 48000.0
    if "pre_downsample" not in state:
        state["pre_downsample"] = {}
    iq_ds, sr_ds = _lpf_resample_cplx(iq_chunk, sr, inter_sr, if_cutoff, state["pre_downsample"])

    if "prev_iq" not in state:
        state["prev_iq"] = np.complex64(0.0)

    combined = np.concatenate(([state["prev_iq"]], iq_ds))
    state["prev_iq"] = iq_ds[-1] if len(iq_ds) else state["prev_iq"]
    phase = np.angle(combined[1:] * np.conjugate(combined[:-1]))

    phase *= sr_ds / (2.0 * np.pi * nfm_dev)

    if "dc_block" not in state:
        state["dc_block"] = {}
    freq_dem = _dc_block_singlepole(phase, state["dc_block"], sr_ds, 30.0)

    voice_cutoff = 5000.0
    if "lpf_voice_taps" not in state:
        num_taps = 129
        fir = scipy.signal.firwin(num_taps, voice_cutoff / (sr_ds / 2.0))
        state["lpf_voice_taps"] = fir
        state["lpf_voice_zi"] = np.zeros(len(fir) - 1, dtype=np.float32)

    filtered, state["lpf_voice_zi"] = scipy.signal.lfilter(
        state["lpf_voice_taps"], [1.0], freq_dem, zi=state["lpf_voice_zi"]
    )

    audio = _resample_if_needed(filtered, sr_ds, ar)
    return audio


def measure_audio_snr(audio: np.ndarray, fs: float, mode: str) -> float:
    """
    Estimates SNR in the demodulated audio, adapted by mode to reduce naive detection errors.
    Uses Welch PSD, integrates over a 'signal band' relevant to each mode, and a 'noise band'.
    Returns an approximate SNR in dB.
    """
    if len(audio) < 512 or fs <= 0:
        return -999.0

    band_settings = {
        'am':  (300.0,  5000.0,  7000.0,  10000.0),
        'fm':  (300.0,  15000.0, 17000.0, 20000.0),
        'nfm': (300.0,  3000.0,  4000.0,  8000.0),
        'air': (300.0,  5000.0,  7000.0,  10000.0),
        'hf':  (300.0,  5000.0,  7000.0,  10000.0),
        'cb':  (300.0,  5000.0,  7000.0,  10000.0),
    }

    sig_low, sig_high, noise_low, noise_high = band_settings.get(mode, (300.0, 5000.0, 7000.0, 10000.0))

    f, pxx = scipy.signal.welch(audio, fs=fs, nperseg=512)

    def clamp_ranges(low, high):
        high = min(high, fs / 2.0)
        low = max(low, 0.0)
        if high <= low:
            return None
        return (low, high)

    s_region = clamp_ranges(sig_low, sig_high)
    n_region = clamp_ranges(noise_low, noise_high)

    if not s_region or not n_region:
        signal_power = np.max(pxx)
        noise_floor = np.median(pxx) + 1e-12
        return 10.0 * np.log10(signal_power / noise_floor)

    s_low, s_high = s_region
    n_low, n_high = n_region

    s_idxs = np.where((f >= s_low) & (f <= s_high))[0]
    n_idxs = np.where((f >= n_low) & (f <= n_high))[0]

    if len(s_idxs) == 0 or len(n_idxs) == 0:
        signal_power = np.max(pxx)
        noise_floor = np.median(pxx) + 1e-12
        return 10.0 * np.log10(signal_power / noise_floor)

    bin_width = f[1] - f[0] if len(f) > 1 else 1.0
    signal_power = np.sum(pxx[s_idxs]) * bin_width
    noise_power = np.sum(pxx[n_idxs]) * bin_width + 1e-12

    if noise_power <= 0:
        return -999.0

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return snr_db


class BaseHackRFListener:
    """
    Base class for HackRF listening with a particular demodulation mode.
    Responsible for opening/closing device, reading samples, demodulating, and recording audio.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 200e3
    DEFAULT_SCAN_START = 1.0e6
    DEFAULT_SCAN_STOP = 2.0e6
    DEFAULT_SCAN_STEP = 0.1e6

    def __init__(
        self,
        mode: str,
        freq: float,
        sample_rate: Optional[float],
        audio_rate: float,
        duration: float,
        out_file: str,
        device_index: int,
        bandwidth: Optional[float],
        lna_gain: int,
        vga_gain: int,
        amp: bool
    ) -> None:
        """
        Initializes the listener with the given parameters (mode, freq, sample_rate, etc.).
        """
        self.mode = mode
        self.freq = freq
        self.sample_rate = sample_rate if sample_rate else self.DEFAULT_SAMPLE_RATE
        self.audio_rate = audio_rate
        self.duration = duration
        self.out_file = out_file
        self.device_index = device_index
        self.bandwidth = bandwidth if bandwidth else self.DEFAULT_BANDWIDTH
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain
        self.amp = amp

        self._dev: Optional[HackRFDevice] = None
        self._demod_state: Dict[str, Any] = {}

    def open(self) -> None:
        """
        Opens and configures the HackRF device according to the stored parameters.
        """
        if self.freq is not None:
            validate_frequency(self.freq)
        validate_sample_rate(self.sample_rate)

        self._dev = HackRFDevice(self.device_index)
        self._dev.set_sample_rate(self.sample_rate)
        self._dev.set_baseband_filter_bandwidth(self.bandwidth, self.mode)
        if self.freq is not None:
            self._dev.set_freq(self.freq)
        self._dev.set_lna_gain(self.lna_gain)
        self._dev.set_vga_gain(self.vga_gain)
        self._dev.set_amp_enable(self.amp)

    def close(self) -> None:
        """
        Closes the HackRF device if open.
        """
        if self._dev is not None:
            self._dev.close()
            logging.info("HackRF device closed.")
            self._dev = None

    def run(self) -> None:
        """
        Starts reading samples from the HackRF, processes them,
        and writes demodulated audio to the output file.
        """
        if not self._dev:
            raise RuntimeError("Device not opened. Call open() first.")

        logging.info(f"Recording {self.mode.upper()} to {self.out_file} ... (Ctrl+C to stop)")

        with sf.SoundFile(
            self.out_file,
            mode='w',
            samplerate=int(self.audio_rate),
            channels=1,
            format='OGG',
            subtype='VORBIS'
        ) as sfh:
            start_t = time.time()
            chunk_size = 8192

            try:
                # Read samples in chunks until duration or user interrupts
                for raw_chunk in self._dev.read_samples(
                    num_samples=0,
                    duration=self.duration,
                    chunk_size=chunk_size
                ):
                    audio = self._process_chunk(raw_chunk)
                    sfh.write(audio)

                    if self.duration > 0:
                        if (time.time() - start_t) >= self.duration:
                            break

            except KeyboardInterrupt:
                logging.warning("User interrupted. Stopping.")
            except Exception as ex:
                logging.error(f"Error in streaming: {ex}")

    def _process_chunk(self, raw_chunk: bytes) -> np.ndarray:
        """
        Converts raw byte chunk to complex64, demodulates it, and returns float32 audio samples.
        """
        arr_u8 = np.frombuffer(raw_chunk, dtype=np.uint8)
        arr_f = (arr_u8.astype(np.float32) - 128.0) / 128.0
        i_samp = arr_f[0::2]
        q_samp = arr_f[1::2]
        cplx = (i_samp + 1j * q_samp).astype(np.complex64)
        audio = self.demodulate(cplx, self._demod_state)
        return audio.astype(np.float32)

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        """
        Abstract method that must be overridden to demodulate a chunk of complex samples.
        """
        raise NotImplementedError()

    def validate_frequency(self) -> None:
        """
        Validates frequency range for the specific mode. Overridden by subclasses if needed.
        """
        pass

    def measure_audio_snr(self, audio: np.ndarray) -> float:
        """
        Measures approximate SNR in the demodulated audio for scanning purposes.
        """
        return measure_audio_snr(audio, fs=self.audio_rate, mode=self.mode)

    def scan_and_find_best_freq(self, start: float, stop: float, step: float, capture_time: float = 0.1) -> float:
        """
        Scans frequencies between start and stop, in increments of step,
        measuring approximate SNR at each frequency. Returns the best frequency.
        """
        if self._dev is None:
            raise RuntimeError("Device must be open before scanning frequencies.")

        if step <= 0:
            raise ValueError("scan-step must be > 0.")

        logging.info(f"Scanning {self.mode.upper()} from {start/1e6:.3f} MHz to {stop/1e6:.3f} MHz, step={step/1e6:.3f} MHz ...")

        best_freq = self.freq if self.freq is not None else start
        best_snr = -999.0

        span = stop - start
        if span < 0:
            logging.error(f"scan-start={start} must be < scan-stop={stop}.")
            raise ValueError("Invalid scanning range: start >= stop.")

        num_steps = int(math.floor(span / step + 0.5))
        freq_list = []
        for i in range(num_steps + 1):
            freq_val = start + i * step
            if freq_val > stop + 1e-9:
                break
            freq_list.append(freq_val)

        for test_freq in freq_list:
            self._dev.set_freq(test_freq)

            # Flush partial buffers
            flush_time = 0.2
            flush_chunk = 8192
            flush_start = time.time()
            for _ in self._dev.read_samples(duration=flush_time, chunk_size=flush_chunk):
                if (time.time() - flush_start) >= flush_time:
                    break

            local_best_snr = -999.0
            num_attempts = 3
            for attempt in range(num_attempts):
                # Short capture for measuring
                chunk = bytearray()
                capture_start = time.time()
                for raw in self._dev.read_samples(duration=capture_time, chunk_size=8192):
                    chunk.extend(raw)
                    if (time.time() - capture_start) >= capture_time:
                        break

                arr_u8 = np.frombuffer(chunk, dtype=np.uint8)
                arr_f = (arr_u8.astype(np.float32) - 128.0) / 128.0
                i_samp = arr_f[0::2]
                q_samp = arr_f[1::2]
                cplx = (i_samp + 1j*q_samp).astype(np.complex64)

                # Use a fresh state for measurement
                local_demod_state: Dict[str, Any] = {}
                audio = self.demodulate(cplx, local_demod_state)
                snr = self.measure_audio_snr(audio)
                logging.debug(f"Freq={test_freq/1e6:.3f} MHz (attempt={attempt+1}/{num_attempts}) => SNR={snr:.2f} dB")

                if snr > local_best_snr:
                    local_best_snr = snr

            if local_best_snr > best_snr:
                best_snr = local_best_snr
                best_freq = test_freq

        logging.info(f"Found best freq={best_freq/1e6:.3f} MHz, SNR={best_snr:.2f} dB")

        # Reset DSP state so scanning doesn't interfere with final demod
        self._demod_state = {}

        self.freq = best_freq
        return best_freq

    def auto_scan_frequency(self, scan_start: float, scan_stop: float, scan_step: float) -> float:
        """
        Performs scanning to find the best frequency in [scan_start..scan_stop..scan_step].
        """
        return self.scan_and_find_best_freq(scan_start, scan_stop, scan_step)


class AMListener(BaseHackRFListener):
    """
    AM demodulator listener class, with defaults and specialized demod for AM broadcast band.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 200e3
    DEFAULT_SCAN_START = 520e3
    DEFAULT_SCAN_STOP = 1710e3
    DEFAULT_SCAN_STEP = 100e3

    def __init__(self, **kwargs) -> None:
        super().__init__(mode='am', **kwargs)

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        return demodulate_am(iq_chunk, state, self.sample_rate, self.audio_rate, cutoff=5000.0)

    def validate_frequency(self) -> None:
        if self.freq is None:
            return
        if self.freq < 520e3 or self.freq > 1710e3:
            logging.warning("Frequency is outside the typical AM broadcast band (520 kHz..1710 kHz).")


class FMListener(BaseHackRFListener):
    """
    FM demodulator listener class, with defaults and specialized demod for broadcast FM.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 2e6
    DEFAULT_SCAN_START = 87.5e6
    DEFAULT_SCAN_STOP = 108e6
    DEFAULT_SCAN_STEP = 0.2e6

    def __init__(self, fm_deviation: Optional[float] = None, **kwargs) -> None:
        super().__init__(mode='fm', **kwargs)
        self.fm_deviation = fm_deviation if fm_deviation is not None else 75000.0
        if self.fm_deviation < 30000.0 or self.fm_deviation > 200000.0:
            logging.warning(
                f"FM deviation ({self.fm_deviation} Hz) is unusual. "
                "Typical broadcast FM is around 75 kHz in many regions."
            )

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        return demodulate_wfm(
            iq_chunk,
            state,
            self.sample_rate,
            self.audio_rate,
            if_cutoff=100000.0,
            audio_cutoff=15000.0,
            deemphasis=7.5e-5,
            fm_dev=self.fm_deviation
        )

    def validate_frequency(self) -> None:
        if self.freq is None:
            return
        if self.freq < 87.5e6 or self.freq > 108e6:
            logging.warning("Frequency is outside typical FM broadcast band (87.5 MHz..108 MHz).")


class NFMListener(BaseHackRFListener):
    """
    NFM demodulator listener class, with defaults for typical VHF/UHF narrow FM usage.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 200e3
    DEFAULT_SCAN_START = 144e6
    DEFAULT_SCAN_STOP = 148e6
    DEFAULT_SCAN_STEP = 25e3

    def __init__(self, nfm_deviation: Optional[float] = None, **kwargs) -> None:
        super().__init__(mode='nfm', **kwargs)
        self.nfm_deviation = nfm_deviation if nfm_deviation is not None else 5000.0
        if self.nfm_deviation < 2000.0 or self.nfm_deviation > 25000.0:
            logging.warning(
                f"NFM deviation ({self.nfm_deviation} Hz) is unusual. "
                "Typical narrow-FM dev is around 2.5kHz to 5kHz."
            )

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        return demodulate_nfm(
            iq_chunk,
            state,
            self.sample_rate,
            self.audio_rate,
            if_cutoff=20000.0,
            nfm_dev=self.nfm_deviation
        )

    def validate_frequency(self) -> None:
        if self.freq is None:
            return
        if self.freq < 1e7:
            logging.warning("Using NFM below ~10 MHz is unusual. Make sure this is correct.")


class AirbandAMListener(BaseHackRFListener):
    """
    Airband AM demodulator listener class, with defaults for VHF airband.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 200e3
    DEFAULT_SCAN_START = 118e6
    DEFAULT_SCAN_STOP = 136e6
    DEFAULT_SCAN_STEP = 25e3

    def __init__(self, **kwargs) -> None:
        super().__init__(mode='air', **kwargs)

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        return demodulate_am(iq_chunk, state, self.sample_rate, self.audio_rate, cutoff=6000.0)

    def validate_frequency(self) -> None:
        if self.freq is None:
            return
        if self.freq < 118e6 or self.freq > 136e6:
            logging.warning("Frequency is outside the typical VHF Air band (118 MHz..136 MHz).")


class HFAMListener(BaseHackRFListener):
    """
    HF AM demodulator listener class, with defaults for shortwave (HF) broadcast.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 100e3
    DEFAULT_SCAN_START = 6.0e6
    DEFAULT_SCAN_STOP = 10.0e6
    DEFAULT_SCAN_STEP = 1.0e6

    def __init__(self, **kwargs) -> None:
        super().__init__(mode='hf', **kwargs)

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        return demodulate_am(iq_chunk, state, self.sample_rate, self.audio_rate, cutoff=5000.0)

    def validate_frequency(self) -> None:
        if self.freq is None:
            return
        if self.freq < 2e6 or self.freq > 26e6:
            logging.warning("Frequency is outside typical shortwave (HF) broadcast range (2 MHz..26 MHz).")


class CBListener(BaseHackRFListener):
    """
    CB (Citizen's Band) AM demodulator listener class, for ~27 MHz.
    """

    DEFAULT_SAMPLE_RATE = 2e6
    DEFAULT_BANDWIDTH = 100e3
    DEFAULT_SCAN_START = 26.965e6
    DEFAULT_SCAN_STOP = 27.405e6
    DEFAULT_SCAN_STEP = 0.1e6

    def __init__(self, **kwargs) -> None:
        super().__init__(mode='cb', **kwargs)

    def demodulate(self, iq_chunk: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        return demodulate_am(iq_chunk, state, self.sample_rate, self.audio_rate, cutoff=5000.0)

    def validate_frequency(self) -> None:
        if self.freq is None:
            return
        if self.freq < 26.965e6 or self.freq > 27.405e6:
            logging.warning("Frequency is outside the standard CB channels (26.965 MHz..27.405 MHz).")


def create_listener(
    mode: str,
    freq: float,
    sample_rate: float,
    audio_rate: float,
    duration: float,
    out_file: str,
    device_index: int,
    bandwidth: float,
    lna_gain: int,
    vga_gain: int,
    amp: bool,
    fm_deviation: Optional[float] = None,
    nfm_deviation: Optional[float] = None
) -> BaseHackRFListener:
    """
    Factory function to create the appropriate listener class for the requested mode.
    """
    if mode == 'am':
        return AMListener(
            freq=freq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            duration=duration,
            out_file=out_file,
            device_index=device_index,
            bandwidth=bandwidth,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            amp=amp
        )
    elif mode == 'fm':
        return FMListener(
            freq=freq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            duration=duration,
            out_file=out_file,
            device_index=device_index,
            bandwidth=bandwidth,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            amp=amp,
            fm_deviation=fm_deviation
        )
    elif mode == 'nfm':
        return NFMListener(
            freq=freq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            duration=duration,
            out_file=out_file,
            device_index=device_index,
            bandwidth=bandwidth,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            amp=amp,
            nfm_deviation=nfm_deviation
        )
    elif mode == 'air':
        return AirbandAMListener(
            freq=freq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            duration=duration,
            out_file=out_file,
            device_index=device_index,
            bandwidth=bandwidth,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            amp=amp
        )
    elif mode == 'hf':
        return HFAMListener(
            freq=freq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            duration=duration,
            out_file=out_file,
            device_index=device_index,
            bandwidth=bandwidth,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            amp=amp
        )
    elif mode == 'cb':
        return CBListener(
            freq=freq,
            sample_rate=sample_rate,
            audio_rate=audio_rate,
            duration=duration,
            out_file=out_file,
            device_index=device_index,
            bandwidth=bandwidth,
            lna_gain=lna_gain,
            vga_gain=vga_gain,
            amp=amp
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the HackRF listening script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified HackRF script to tune and demodulate AM / FM / NFM / "
            "AIR / HF / CB, and record the resulting audio to an OGG file."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        required=True,
        choices=["am", "fm", "nfm", "air", "hf", "cb"],
        help="Demodulation mode: am / fm / nfm / air / hf / cb."
    )
    parser.add_argument(
        "-f", "--freq",
        type=float,
        default=None,
        help="Frequency in Hz (e.g. 1e6 for 1 MHz)."
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=float,
        default=None,
        help="HackRF sample rate in Hz."
    )
    parser.add_argument(
        "-b", "--bandwidth",
        type=float,
        default=None,
        help="Baseband filter bandwidth in Hz."
    )
    parser.add_argument(
        "-a", "--audio-rate",
        type=float,
        default=48000,
        help="Output audio sample rate in Hz."
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds (0 => indefinite)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="recording.ogg",
        help="Output .ogg file name."
    )
    parser.add_argument(
        "-I", "--device-index",
        type=int,
        default=0,
        help="HackRF device index."
    )
    parser.add_argument(
        "-L", "--lna-gain",
        type=int,
        default=16,
        help="LNA gain in dB [0..40 in steps of 8]."
    )
    parser.add_argument(
        "-G", "--vga-gain",
        type=int,
        default=20,
        help="VGA gain in dB [0..62 in steps of 2]."
    )
    parser.add_argument(
        "-P", "--amp",
        action='store_true',
        help="Enable HackRF internal amplifier."
    )
    parser.add_argument(
        "-F", "--auto-scan",
        action="store_true",
        help="Attempt to automatically find the best frequency for the chosen mode."
    )
    parser.add_argument(
        "-S", "--scan-start",
        type=float,
        default=None,
        help="Start frequency (Hz) for scanning."
    )
    parser.add_argument(
        "-T", "--scan-stop",
        type=float,
        default=None,
        help="Stop frequency (Hz) for scanning."
    )
    parser.add_argument(
        "-p", "--scan-step",
        type=float,
        default=None,
        help="Step in Hz for scanning."
    )
    parser.add_argument(
        "-W", "--fm-deviation",
        type=float,
        default=None,
        help="Deviation in Hz for wide-FM demod."
    )
    parser.add_argument(
        "-N", "--nfm-deviation",
        type=float,
        default=None,
        help="Deviation in Hz for narrow-FM demod."
    )
    parser.add_argument(
        "-v", "--verbose",
        action='count',
        default=0,
        help="Increase verbosity: -v => INFO, -vv => DEBUG."
    )

    args = parser.parse_args()

    # Make --freq required only if --auto-scan is NOT used
    if not args.auto_scan and args.freq is None:
        parser.error("--freq is required unless --auto-scan is enabled.")

    return args


def setup_logging(verbosity: int) -> None:
    """
    Sets up logging according to verbosity level.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')


def main() -> None:
    """
    Main entry point: parses args, creates listener, optionally scans for best freq, then records.
    """
    args = parse_args()
    setup_logging(args.verbose)

    freq = args.freq

    listener = create_listener(
        mode=args.mode.lower(),
        freq=freq,
        sample_rate=args.sample_rate,
        audio_rate=args.audio_rate,
        duration=args.duration,
        out_file=args.output,
        device_index=args.device_index,
        bandwidth=args.bandwidth,
        lna_gain=args.lna_gain,
        vga_gain=args.vga_gain,
        amp=args.amp,
        fm_deviation=args.fm_deviation,
        nfm_deviation=args.nfm_deviation
    )

    listener.validate_frequency()

    scan_start = args.scan_start
    scan_stop = args.scan_stop
    scan_step = args.scan_step

    if args.auto_scan:
        if scan_start is None:
            scan_start = listener.DEFAULT_SCAN_START
        if scan_stop is None:
            scan_stop = listener.DEFAULT_SCAN_STOP
        if scan_step is None:
            scan_step = listener.DEFAULT_SCAN_STEP

        listener.open()
        try:
            best_freq = listener.auto_scan_frequency(scan_start, scan_stop, scan_step)
            listener.close()
            logging.info(f"Re-opening at best freq={best_freq/1e6:.3f} MHz.")
            listener.freq = best_freq
        except Exception as ex:
            listener.close()
            logging.error(f"Error during frequency scanning: {ex}")
            return

    try:
        listener.open()
        listener.run()
    except KeyboardInterrupt:
        logging.warning("User interrupted.")
    except Exception as ex:
        logging.error(f"Error: {ex}")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
