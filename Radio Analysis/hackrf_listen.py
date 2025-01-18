#!/usr/bin/env python3

# -------------------------------------------------------
# Script: hackrf_listen.py
#
# Description:
# This script uses a HackRF device to tune to a specified
# frequency (or optionally scan for an active frequency),
# automatically determine the best demodulation mode if
# not specified (NFM, AM, USB, LSB, or WFM), and record
# the resulting audio to an OGG Vorbis file.
#
# Advanced scanning features allow the script to search
# over larger frequency ranges with smaller steps. It can
# attempt multiple demodulation modes at each candidate
# frequency and evaluate the resulting audio for sufficient
# signal quality before deciding on the best frequency
# and mode.
#
# Usage:
#   ./hackrf_listen.py --freq FREQ_HZ [options]
#
# Options:
#   -f, --freq FREQ_HZ           Tuning frequency in Hz (e.g. 100e6).
#   -m, --mode {nfm,am,usb,lsb,wfm}
#                                Demodulation mode. If not specified, the script
#                                attempts to auto-detect based on frequency.
#   -s, --sample-rate SR         HackRF sample rate in Hz (default: 2e6).
#   -b, --bandwidth BW           Baseband filter bandwidth in Hz (default: SR).
#   -a, --audio-rate AR          Output audio sample rate in Hz (default: 48000).
#   -d, --duration SEC           Duration in seconds (0 => indefinite).
#   -o, --output FILE            Output .ogg file name (default: recording.ogg).
#   -I, --device-index IDX       HackRF device index (default: 0).
#   -L, --lna-gain DB            LNA gain in dB [0..40 in steps of 8] (default: 16).
#   -G, --vga-gain DB            VGA gain in dB [0..62 in steps of 2] (default: 20).
#   -P, --amp                    Enable HackRF internal amplifier (default: off).
#   -T, --auto-tune              Advanced tune around --freq ± --auto-tune-range in
#                                increments of --auto-tune-step, testing multiple modes.
#   -S, --auto-scan              Perform an advanced FFT-based scan over [--freq-start..--freq-end].
#                                If --freq is given but no start/end, uses freq ± 1 MHz.
#   -X, --freq-start FREQ_HZ     Start frequency for auto-scan (Hz).
#   -Y, --freq-end FREQ_HZ       End frequency for auto-scan (Hz).
#   -H, --scan-threshold DB      Threshold in dB above noise floor for auto-scan (default=5.0).
#   -Z, --scan-fft-size N        FFT size for auto-scan (default=1024).
#   -W, --scan-dwell SEC         Dwell time in seconds for each step in auto-scan (default=0.2).
#   -R, --auto-tune-range HZ     Range ± around --freq for auto-tune (default=100000).
#   -E, --auto-tune-step HZ      Step size in Hz for auto-tune (default=10000).
#   -U, --adv-scan-steps HZ      Frequency step size in advanced scanning (default=20000).
#   -Q, --adv-scan-quality FLOAT Minimum audio "quality" threshold (default=0.01).
#   -N, --adv-scan-min-snr FLOAT Minimum SNR (dB) to consider a signal valid (default=5.0).
#   -v, --verbose                Increase verbosity (-v => INFO, -vv => DEBUG).
#
# Examples:
#   1) Wide FM broadcast at ~100 MHz:
#      ./hackrf_listen.py --freq 100e6 --mode wfm --duration 10 --output fm_test.ogg
#
#   2) 2m Amateur Band NFM (145.0 MHz), indefinite recording:
#      ./hackrf_listen.py --freq 145e6 --mode nfm --output my_recording.ogg
#
#   3) HF SSB (7.2 MHz, LSB), narrower sample rate for voice:
#      ./hackrf_listen.py --freq 7.2e6 --mode lsb --sample-rate 200000 \
#                         --output 40m_lsb.ogg
#
#   4) Automatic advanced scan to find best signal near 100 MHz:
#      ./hackrf_listen.py --freq 100e6 --auto-scan
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
#   - numpy (install via: pip install numpy==2.2.1)
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
import sys
import numpy as np
import scipy.signal
import soundfile as sf
import ctypes
import atexit
import math
from typing import Optional, Tuple, Generator, Any, Dict, List


# =========================================================================
# HackRF ctypes interface
# =========================================================================
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

# Initialize HackRF
_init_result = _libhackrf.hackrf_init()
if _init_result != HACKRF_SUCCESS:
    raise RuntimeError("Error initializing HackRF (libhackrf).")


def _finalize_hackrf() -> None:
    """
    Finalize HackRF upon script exit by calling hackrf_exit().
    """
    _libhackrf.hackrf_exit()


atexit.register(_finalize_hackrf)


# =========================================================================
# HackRF Device wrapper
# =========================================================================
class HackRFDevice:
    """
    Minimal HackRF device wrapper for receiving complex samples via ctypes.
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

        self._streaming = False
        self._rx_callback_function = _transfer_callback(self._rx_callback)
        self._rx_buffer = bytearray()
        self._target_samples = 0

    def close(self) -> None:
        """
        Close the HackRF device and stop any active streaming.
        """
        if self._dev:
            if self._streaming:
                _libhackrf.hackrf_stop_rx(self._dev)
                self._streaming = False
            _libhackrf.hackrf_close(self._dev)
            self._dev = _p_hackrf_device(None)

    def __del__(self) -> None:
        """
        Destructor: ensure device is closed.
        """
        self.close()

    def set_sample_rate(self, sr_hz: float) -> None:
        """
        Set the HackRF device sample rate.
        """
        sr = ctypes.c_double(sr_hz)
        r = _libhackrf.hackrf_set_sample_rate(self._dev, sr)
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting sample rate to {sr_hz} Hz, code={r}")

    def set_baseband_filter_bandwidth(self, bw_hz: float) -> None:
        """
        Set the HackRF baseband filter bandwidth.
        """
        r = _libhackrf.hackrf_set_baseband_filter_bandwidth(
            self._dev, ctypes.c_uint32(int(bw_hz))
        )
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting baseband filter BW to {bw_hz} Hz, code={r}")

    def set_freq(self, freq_hz: float) -> None:
        """
        Set the HackRF tuning frequency in Hz.
        """
        r = _libhackrf.hackrf_set_freq(self._dev, ctypes.c_uint64(int(freq_hz)))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting frequency to {freq_hz} Hz, code={r}")

    def set_amp_enable(self, enable: bool) -> None:
        """
        Enable or disable the HackRF internal amplifier.
        """
        val = 1 if enable else 0
        r = _libhackrf.hackrf_set_amp_enable(self._dev, ctypes.c_uint8(val))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting amp enable={enable}, code={r}")

    def set_lna_gain(self, gain: int) -> None:
        """
        Set the LNA gain in dB.
        """
        r = _libhackrf.hackrf_set_lna_gain(self._dev, ctypes.c_uint32(gain))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting LNA gain={gain}, code={r}")

    def set_vga_gain(self, gain: int) -> None:
        """
        Set the VGA gain in dB.
        """
        r = _libhackrf.hackrf_set_vga_gain(self._dev, ctypes.c_uint32(gain))
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting VGA gain={gain}, code={r}")

    def _rx_callback(self, transfer_ptr: ctypes.POINTER(_hackrf_transfer)) -> int:
        """
        Internal RX callback function for HackRF streaming.
        Accumulates raw byte data in self._rx_buffer.
        """
        transfer = transfer_ptr.contents
        buf_length = transfer.valid_length
        if buf_length > 0:
            data_array = ctypes.cast(
                transfer.buffer, ctypes.POINTER(ctypes.c_byte * buf_length)
            ).contents
            self._rx_buffer.extend(data_array)

        # If we need a certain # of samples, once we have them, stop streaming
        if self._target_samples > 0 and len(self._rx_buffer) >= self._target_samples:
            self._streaming = False
            return 0
        return 0

    def read_samples(
        self,
        num_samples: int = 0,
        duration: float = 0.0,
        chunk_size: int = 8192
    ) -> Generator[bytearray, None, None]:
        """
        Generator that yields chunks of raw I/Q data (uint8) from the HackRF.
        """
        if self._streaming:
            raise RuntimeError("Already streaming. Stop first or use another device.")

        self._rx_buffer = bytearray()
        # Each complex sample = 2 bytes: I (1 byte) + Q (1 byte)
        self._target_samples = 0 if num_samples <= 0 else (num_samples * 2)

        r = _libhackrf.hackrf_start_rx(self._dev, self._rx_callback_function, None)
        if r != HACKRF_SUCCESS:
            raise RuntimeError(f"Error starting RX, code={r}")

        self._streaming = True
        start_time = time.time()

        try:
            while True:
                # If enough data for a chunk, yield
                if len(self._rx_buffer) >= (chunk_size * 2):
                    out = self._rx_buffer[: (chunk_size * 2)]
                    del self._rx_buffer[: (chunk_size * 2)]
                    yield out

                # Check stopping conditions
                if not self._streaming:
                    break
                if duration > 0 and (time.time() - start_time) >= duration:
                    break

                time.sleep(0.01)

            # Yield leftover
            while len(self._rx_buffer) > 0:
                need = min(len(self._rx_buffer), chunk_size * 2)
                out = self._rx_buffer[:need]
                del self._rx_buffer[:need]
                yield out

        finally:
            _libhackrf.hackrf_stop_rx(self._dev)
            self._streaming = False


# =========================================================================
# DSP Routines (NFM, AM, SSB, WFM)
# =========================================================================
def demodulate_nfm(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sample_rate: float,
    audio_rate: float,
    cutoff: float = 15000
) -> np.ndarray:
    """
    Narrow FM demodulation for voice transmissions.
    """
    prev_iq = state.get("prev_iq", np.complex64(0.0))
    combined = np.concatenate(([prev_iq], iq_chunk))
    state["prev_iq"] = iq_chunk[-1] if len(iq_chunk) else prev_iq

    phase = np.angle(combined[1:] * np.conjugate(combined[:-1]))

    if "dc_block_prev" not in state:
        state["dc_block_prev"] = 0.0
    alpha = 0.9995
    dc_prev = state["dc_block_prev"]
    out = np.zeros_like(phase, dtype=np.float32)
    for i, val in enumerate(phase):
        tmp = val + alpha * dc_prev
        out[i] = tmp
        dc_prev = tmp
    state["dc_block_prev"] = dc_prev

    # Lowpass
    if "lpf_taps" not in state:
        num_taps = 101
        fir = scipy.signal.firwin(num_taps, cutoff / (sample_rate / 2.0))
        state["lpf_taps"] = fir
        state["lpf_zi"] = np.zeros(len(fir) - 1, dtype=np.float32)

    filtered, state["lpf_zi"] = scipy.signal.lfilter(
        state["lpf_taps"], [1.0], out, zi=state["lpf_zi"]
    )

    decim_factor = int(sample_rate // audio_rate)
    if decim_factor < 1:
        decim_factor = 1
    audio = filtered[::decim_factor]

    # Fine resampling
    true_ratio = audio_rate / (sample_rate / decim_factor)
    if abs(true_ratio - 1.0) > 0.0001:
        up = int(audio_rate * 1000)
        down = int((sample_rate / decim_factor) * 1000)
        audio = scipy.signal.resample_poly(audio, up, down)

    return audio.astype(np.float32)


def demodulate_am(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sample_rate: float,
    audio_rate: float,
    cutoff: float = 5000
) -> np.ndarray:
    """
    AM demodulation (envelope detection).
    """
    envelope = np.abs(iq_chunk)

    if "dc_block_prev" not in state:
        state["dc_block_prev"] = 0.0
    alpha = 0.9995
    dc_prev = state["dc_block_prev"]
    out = np.zeros_like(envelope, dtype=np.float32)
    for i, val in enumerate(envelope):
        tmp = val + alpha * dc_prev
        out[i] = tmp
        dc_prev = tmp
    state["dc_block_prev"] = dc_prev

    if "lpf_taps" not in state:
        num_taps = 101
        fir = scipy.signal.firwin(num_taps, cutoff / (sample_rate / 2.0))
        state["lpf_taps"] = fir
        state["lpf_zi"] = np.zeros(len(fir) - 1, dtype=np.float32)

    filtered, state["lpf_zi"] = scipy.signal.lfilter(
        state["lpf_taps"], [1.0], out, zi=state["lpf_zi"]
    )

    decim_factor = int(sample_rate // audio_rate)
    if decim_factor < 1:
        decim_factor = 1
    audio = filtered[::decim_factor]

    true_ratio = audio_rate / (sample_rate / decim_factor)
    if abs(true_ratio - 1.0) > 0.0001:
        up = int(audio_rate * 1000)
        down = int((sample_rate / decim_factor) * 1000)
        audio = scipy.signal.resample_poly(audio, up, down)

    return audio.astype(np.float32)


def demodulate_ssb(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sample_rate: float,
    audio_rate: float,
    sideband: str = 'usb',
    audio_bw: float = 3000
) -> np.ndarray:
    """
    Basic SSB demodulation for USB/LSB.
    """
    shift_freq = +1500 if sideband == 'usb' else -1500
    if "phase_acc" not in state:
        state["phase_acc"] = 0.0
    phase_acc = state["phase_acc"]

    ts = 1.0 / sample_rate
    n = len(iq_chunk)
    t_arr = np.arange(n) * ts
    full_phase = 2.0 * math.pi * shift_freq * t_arr + phase_acc

    if n > 0:
        state["phase_acc"] = full_phase[-1] + 2.0 * math.pi * shift_freq * ts

    mix = iq_chunk * np.exp(-1j * full_phase).astype(np.complex64)
    audio_signal = mix.real

    if "lpf_taps" not in state:
        num_taps = 101
        fir = scipy.signal.firwin(num_taps, audio_bw / (sample_rate / 2.0))
        state["lpf_taps"] = fir
        state["lpf_zi"] = np.zeros(len(fir) - 1, dtype=np.float32)

    filtered, state["lpf_zi"] = scipy.signal.lfilter(
        state["lpf_taps"], [1.0], audio_signal, zi=state["lpf_zi"]
    )

    decim_factor = int(sample_rate // audio_rate)
    if decim_factor < 1:
        decim_factor = 1
    audio = filtered[::decim_factor]

    true_ratio = audio_rate / (sample_rate / decim_factor)
    if abs(true_ratio - 1.0) > 0.0001:
        up = int(audio_rate * 1000)
        down = int((sample_rate / decim_factor) * 1000)
        audio = scipy.signal.resample_poly(audio, up, down)

    return audio.astype(np.float32)


def demodulate_wfm(
    iq_chunk: np.ndarray,
    state: Dict[str, Any],
    sample_rate: float,
    audio_rate: float,
    if_cutoff: float = 100000.0,
    audio_cutoff: float = 15000.0,
    deemphasis: float = 7.5e-5
) -> np.ndarray:
    """
    Wide FM demodulation (mono) with simple 75us de-emphasis.

    Steps:
      1) frequency discriminator
      2) wide LPF (~100kHz) + decimate to ~200k
      3) 75us de-emphasis
      4) final LPF (~15kHz) + decimate to audio_rate
    """
    prev_iq = state.get("prev_iq", np.complex64(0.0))
    combined = np.concatenate(([prev_iq], iq_chunk))
    state["prev_iq"] = iq_chunk[-1] if len(iq_chunk) else prev_iq

    # derivative freq discriminator
    phase = np.angle(combined[1:] * np.conjugate(combined[:-1]))

    # DC block
    if "dc_block_prev" not in state:
        state["dc_block_prev"] = 0.0
    alpha = 0.9995
    dc_prev = state["dc_block_prev"]
    freqdem = np.zeros_like(phase, dtype=np.float32)
    for i, val in enumerate(phase):
        tmp = val + alpha * dc_prev
        freqdem[i] = tmp
        dc_prev = tmp
    state["dc_block_prev"] = dc_prev

    # 1) wide LPF (~100kHz) + decimate to ~200k
    if "lpf_wide_taps" not in state:
        num_taps = 101
        cutoff_norm = if_cutoff / (sample_rate / 2.0)
        fir = scipy.signal.firwin(num_taps, cutoff_norm)
        state["lpf_wide_taps"] = fir
        state["lpf_wide_zi"] = np.zeros(len(fir) - 1, dtype=np.float32)

    filtered1, state["lpf_wide_zi"] = scipy.signal.lfilter(
        state["lpf_wide_taps"], [1.0], freqdem, zi=state["lpf_wide_zi"]
    )

    desired_int_rate = 200000.0
    decim_factor1 = int(sample_rate // desired_int_rate)
    if decim_factor1 < 1:
        decim_factor1 = 1

    interp1 = filtered1[::decim_factor1]
    actual_int_rate = sample_rate / decim_factor1

    # 2) simple 75us de-emphasis
    if "deemph_prev" not in state:
        state["deemph_prev"] = 0.0

    dt = 1.0 / actual_int_rate
    alpha_deemph = dt / (deemphasis + dt)
    out_deemph = np.zeros_like(interp1, dtype=np.float32)
    dp = state["deemph_prev"]
    for i, x in enumerate(interp1):
        dp += alpha_deemph * (x - dp)
        out_deemph[i] = dp
    state["deemph_prev"] = dp

    # 3) final LPF ~15kHz + decimate to audio_rate
    if "lpf_audio_taps" not in state:
        num_taps2 = 101
        cutoff2 = audio_cutoff / (actual_int_rate / 2.0)
        fir2 = scipy.signal.firwin(num_taps2, cutoff2)
        state["lpf_audio_taps"] = fir2
        state["lpf_audio_zi"] = np.zeros(len(fir2) - 1, dtype=np.float32)

    filtered2, state["lpf_audio_zi"] = scipy.signal.lfilter(
        state["lpf_audio_taps"], [1.0], out_deemph, zi=state["lpf_audio_zi"]
    )

    decim_factor2 = int(actual_int_rate // audio_rate)
    if decim_factor2 < 1:
        decim_factor2 = 1

    audio = filtered2[::decim_factor2]
    final_rate = actual_int_rate / decim_factor2

    # fine resample to audio_rate if needed
    ratio = audio_rate / final_rate
    if abs(ratio - 1.0) > 0.0001:
        up = int(audio_rate * 1000)
        down = int(final_rate * 1000)
        audio = scipy.signal.resample_poly(audio, up, down)

    return audio.astype(np.float32)


# =========================================================================
# Helper functions for scanning/auto-tune
# =========================================================================
def generate_frequency_steps(
    start_freq: float,
    end_freq: float,
    step_hz: float
) -> List[float]:
    """
    Generate a list of frequency values from start_freq to end_freq in increments of step_hz.
    Ensures at least one value is returned if step_hz is invalid or range is small.
    """
    freqs: List[float] = []
    if start_freq < 1e4:
        start_freq = 1e4
    if step_hz <= 0:
        step_hz = 1e5

    current = start_freq
    while current <= end_freq + 1.0:
        freqs.append(current)
        current += step_hz

    if not freqs:
        freqs = [start_freq]
    return freqs


def capture_short_chunk(
    dev: HackRFDevice,
    sample_rate: float,
    chunk_duration: float
) -> Optional[np.ndarray]:
    """
    Capture a short chunk of samples from the HackRF device and return as complex64 ndarray.
    Returns None if failed or insufficient data.
    """
    chunk_bytes_needed = int(sample_rate * 2 * chunk_duration)
    dev._rx_buffer = bytearray()
    dev._target_samples = chunk_bytes_needed

    # Start RX
    _libhackrf.hackrf_start_rx(dev._dev, dev._rx_callback_function, None)
    dev._streaming = True

    start_t = time.time()
    while dev._streaming:
        if len(dev._rx_buffer) >= chunk_bytes_needed:
            break
        if (time.time() - start_t) > (chunk_duration * 5.0):  # some margin
            break
        time.sleep(0.01)

    # Stop RX
    _libhackrf.hackrf_stop_rx(dev._dev)
    dev._streaming = False

    data = dev._rx_buffer[:chunk_bytes_needed]
    if len(data) < 2:
        return None

    iq_u8 = np.frombuffer(data, dtype=np.uint8)
    iq_f = (iq_u8.astype(np.float32) - 128.0)
    i_samples = iq_f[0::2]
    q_samples = iq_f[1::2]
    cplx = i_samples + 1j * q_samples
    return cplx.astype(np.complex64)


def compute_snr_db(power_spectrum: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean power, noise floor, and return SNR in dB.
    Also returns the noise floor estimate.
    """
    mean_power = np.mean(power_spectrum)
    noise_floor_est = np.median(power_spectrum)
    if noise_floor_est <= 0:
        noise_floor_est = 1e-10
    snr_db = 10.0 * np.log10(mean_power / noise_floor_est)
    return float(snr_db), float(noise_floor_est)


def test_multiple_narrow_modes(
    cplx: np.ndarray,
    sample_rate: float,
    candidate_modes: List[str],
    snr_db: float,
    min_snr_db: float,
    min_quality: float
) -> Tuple[float, Optional[str]]:
    """
    Given complex samples, compute a 'score' for each mode in candidate_modes
    (narrow signals). Return the best local score and best mode, or (0.0, None).
    """
    if snr_db < min_snr_db:
        # No point in demodulating if SNR is below min
        return 0.0, None

    best_local_score = 0.0
    best_local_mode = None

    # Quick test snippet: demodulate with each mode,
    # measure "quality" = mean(abs(audio)).
    for md in candidate_modes:
        demod_state: Dict[str, Any] = {}
        if md == "nfm":
            audio = demodulate_nfm(cplx, demod_state, sample_rate, 48000, cutoff=12000)
        elif md == "am":
            audio = demodulate_am(cplx, demod_state, sample_rate, 48000, cutoff=8000)
        elif md == "usb":
            audio = demodulate_ssb(cplx, demod_state, sample_rate, 48000, sideband='usb', audio_bw=3000)
        elif md == "lsb":
            audio = demodulate_ssb(cplx, demod_state, sample_rate, 48000, sideband='lsb', audio_bw=3000)
        else:
            continue

        quality = float(np.mean(np.abs(audio)))
        score = float(snr_db) + 100.0 * quality

        if quality >= min_quality and score > best_local_score:
            best_local_score = score
            best_local_mode = md

    return best_local_score, best_local_mode


def test_all_modes(
    cplx: np.ndarray,
    sample_rate: float,
    candidate_modes: List[str],
    snr_db: float,
    min_snr_db: float,
    min_quality: float
) -> Tuple[float, Optional[str]]:
    """
    Same as test_multiple_narrow_modes but includes 'wfm' if present
    in candidate_modes.
    """
    if snr_db < min_snr_db:
        return 0.0, None

    best_local_score = 0.0
    best_local_mode = None

    for md in candidate_modes:
        demod_state: Dict[str, Any] = {}
        if md == "nfm":
            audio = demodulate_nfm(cplx, demod_state, sample_rate, 48000, cutoff=12000)
        elif md == "am":
            audio = demodulate_am(cplx, demod_state, sample_rate, 48000, cutoff=8000)
        elif md == "usb":
            audio = demodulate_ssb(cplx, demod_state, sample_rate, 48000, sideband='usb', audio_bw=3000)
        elif md == "lsb":
            audio = demodulate_ssb(cplx, demod_state, sample_rate, 48000, sideband='lsb', audio_bw=3000)
        elif md == "wfm":
            audio = demodulate_wfm(cplx, demod_state, sample_rate, 48000,
                                   if_cutoff=100000.0,
                                   audio_cutoff=15000.0,
                                   deemphasis=7.5e-5)
        else:
            continue

        quality = float(np.mean(np.abs(audio)))
        score = float(snr_db) + 100.0 * quality

        if quality >= min_quality and score > best_local_score:
            best_local_score = score
            best_local_mode = md

    return best_local_score, best_local_mode


def auto_tune(
    device_index: int,
    base_freq: float,
    sample_rate: float,
    lna_gain: int,
    vga_gain: int,
    amp: bool,
    tune_range: float,
    tune_step: float,
    min_snr_db: float = 5.0,
    min_quality: float = 0.01
) -> Tuple[Optional[float], Optional[str]]:
    """
    Advanced auto-tune around base_freq ± tune_range in increments of tune_step.

    For each step:
      1) Tune to freq
      2) Perform short capture
      3) Compute SNR
      4) Attempt narrowband demod modes (NFM, AM, USB, LSB)
      5) Keep track of best freq & mode if above thresholds.
    """
    dev = HackRFDevice(device_index)
    dev.set_sample_rate(sample_rate)
    dev.set_baseband_filter_bandwidth(sample_rate)
    dev.set_lna_gain(lna_gain)
    dev.set_vga_gain(vga_gain)
    dev.set_amp_enable(amp)

    low_bound = base_freq - tune_range
    high_bound = base_freq + tune_range
    if low_bound < 1e4:
        low_bound = 1e4
    if high_bound < low_bound:
        high_bound = low_bound + 1e5

    freq_steps = generate_frequency_steps(low_bound, high_bound, tune_step)
    candidate_modes = ["nfm", "am", "usb", "lsb"]

    best_freq = None
    best_mode = None
    best_score = 0.0

    try:
        for freq_test in freq_steps:
            try:
                dev.set_freq(freq_test)
            except RuntimeError:
                logging.debug(f"Failed to set freq={freq_test} Hz; skipping.")
                continue

            cplx = capture_short_chunk(dev, sample_rate, chunk_duration=0.1)
            if cplx is None or len(cplx) < 2:
                continue

            mag = np.abs(cplx)
            power_spectrum = mag**2
            snr_db, _ = compute_snr_db(power_spectrum)

            local_score, local_mode = test_multiple_narrow_modes(
                cplx, sample_rate, candidate_modes,
                snr_db, min_snr_db, min_quality
            )
            logging.debug(
                f"AutoTune freq={freq_test/1e6:.4f} MHz, local_mode={local_mode}, "
                f"SNR={snr_db:.2f}, score={local_score:.4f}"
            )
            if local_mode and local_score > best_score:
                best_score = local_score
                best_mode = local_mode
                best_freq = freq_test

    finally:
        dev.close()

    if best_freq is None or best_mode is None:
        return (None, None)

    return (best_freq, best_mode)


def find_fft_peaks(
    cplx: np.ndarray,
    fft_size: int,
    threshold_db: float
) -> List[Tuple[int, float]]:
    """
    Perform FFT on a portion of cplx data with a Hanning window, find bins
    above threshold_db above noise floor, return list of (bin_index, peak_power_db).
    """
    if len(cplx) < fft_size:
        return []

    window = np.hanning(fft_size)
    chunk = cplx[:fft_size] * window
    fft_out = np.fft.fftshift(np.fft.fft(chunk))
    power_spectrum = 20 * np.log10(np.abs(fft_out) + 1e-9)

    noise_floor_est = np.median(power_spectrum)
    thr_val = noise_floor_est + threshold_db
    above_idx = np.where(power_spectrum >= thr_val)[0]
    if len(above_idx) == 0:
        return []

    peaks = [(idx, float(power_spectrum[idx])) for idx in above_idx]
    return peaks


def auto_scan(
    device_index: int,
    start_freq: float,
    end_freq: float,
    sample_rate: float,
    bandwidth: float,
    dwell_time: float,
    threshold_db: float,
    fft_size: int,
    lna_gain: int,
    vga_gain: int,
    amp: bool,
    scan_step: float = 20000.0,
    min_snr_db: float = 5.0,
    min_quality: float = 0.01
) -> Tuple[Optional[float], Optional[str]]:
    """
    Advanced FFT-based auto-scan from start_freq to end_freq in increments of scan_step.
    For each step:
      1) Center HackRF on freq
      2) Capture samples
      3) FFT -> find signals above threshold
      4) For each candidate bin, estimate SNR, try multiple modes
      5) Evaluate audio quality and choose best.
    """
    dev = HackRFDevice(device_index)
    dev.set_sample_rate(sample_rate)
    dev.set_baseband_filter_bandwidth(bandwidth)
    dev.set_lna_gain(lna_gain)
    dev.set_vga_gain(vga_gain)
    dev.set_amp_enable(amp)

    if start_freq < 1e4:
        start_freq = 1e4
    if end_freq <= start_freq:
        end_freq = start_freq + 1e6

    freq_list = generate_frequency_steps(start_freq, end_freq, scan_step)

    best_freq = None
    best_mode = None
    best_score = 0.0
    candidate_modes = ["nfm", "am", "usb", "lsb", "wfm"]

    try:
        for ftest in freq_list:
            try:
                dev.set_freq(ftest)
            except RuntimeError:
                logging.debug(f"Failed to set freq={ftest} Hz; skipping.")
                continue

            # Capture a chunk for dwell_time
            samples_needed = max(fft_size*2, int(sample_rate*dwell_time))
            dev._rx_buffer = bytearray()
            dev._target_samples = samples_needed * 2

            # Start streaming
            _libhackrf.hackrf_start_rx(dev._dev, dev._rx_callback_function, None)
            dev._streaming = True

            start_t = time.time()
            while dev._streaming:
                if (time.time() - start_t) >= dwell_time:
                    break
                time.sleep(0.01)

            _libhackrf.hackrf_stop_rx(dev._dev)
            dev._streaming = False

            data = dev._rx_buffer[:(samples_needed*2)]
            if len(data) < fft_size*2:
                logging.debug(f"Insufficient data at freq {ftest/1e6:.3f} MHz.")
                continue

            iq_u8 = np.frombuffer(data, dtype=np.uint8)
            iq_f = iq_u8.astype(np.float32) - 128.0
            i_samples = iq_f[0::2]
            q_samples = iq_f[1::2]
            cplx = (i_samples + 1j*q_samples).astype(np.complex64)

            # Find peaks in FFT
            peaks = find_fft_peaks(cplx, fft_size, threshold_db)
            if not peaks:
                continue

            # For each peak bin, do quick SNR check and mode test
            freq_axis = np.linspace(-0.5*sample_rate, 0.5*sample_rate,
                                    fft_size, endpoint=False)

            noise_floor_est = np.median([p[1] for p in peaks])  # not quite accurate, but enough for example

            for bin_idx, peak_power in peaks:
                snr_db = float(peak_power - noise_floor_est)
                offset_hz = freq_axis[bin_idx]
                signal_freq = ftest + offset_hz

                # Test all modes
                local_score, local_mode = test_all_modes(
                    cplx, sample_rate, candidate_modes,
                    snr_db, min_snr_db, min_quality
                )

                logging.debug(
                    f"Scan freq={ftest/1e6:.3f}, bin={bin_idx}, "
                    f"signal_freq={signal_freq/1e6:.3f}, peak_power={peak_power:.1f}, "
                    f"SNR={snr_db:.1f}, best_local_mode={local_mode}, score={local_score:.3f}"
                )

                if local_mode and local_score > best_score:
                    best_score = local_score
                    best_mode = local_mode
                    best_freq = signal_freq

    finally:
        dev.close()

    return (best_freq, best_mode)


# =========================================================================
# Main listening/recording
# =========================================================================
class HackRFListener:
    """
    Class to open HackRF, tune frequency, demodulate to audio, and record to OGG.
    """
    def __init__(
        self,
        freq: float,
        sample_rate: float,
        audio_rate: float,
        mode: str = 'wfm',
        duration: float = 0.0,
        out_file: str = 'recording.ogg',
        device_index: int = 0,
        bandwidth: Optional[float] = None,
        lna_gain: int = 16,
        vga_gain: int = 20,
        amp: bool = False
    ) -> None:
        self.freq = freq
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.mode = mode.lower() if mode else 'auto'
        self.duration = duration
        self.out_file = out_file
        self.device_index = device_index
        self.bandwidth = bandwidth if bandwidth else sample_rate
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain
        self.amp = amp
        self._dev: Optional[HackRFDevice] = None
        self._demod_state: Dict[str, Any] = {}

    def open(self) -> None:
        """
        Open the HackRF device and configure for the desired frequency, sample rate, etc.
        """
        self._dev = HackRFDevice(self.device_index)
        self._dev.set_sample_rate(self.sample_rate)
        self._dev.set_baseband_filter_bandwidth(self.bandwidth)
        self._dev.set_freq(self.freq)
        self._dev.set_lna_gain(self.lna_gain)
        self._dev.set_vga_gain(self.vga_gain)
        self._dev.set_amp_enable(self.amp)
        logging.info(
            f"HackRF opened: freq={self.freq/1e6:.4f} MHz, "
            f"sample_rate={self.sample_rate} Hz, bandwidth={self.bandwidth} Hz, "
            f"mode={self.mode}"
        )

    def close(self) -> None:
        """
        Close the underlying HackRF device.
        """
        if self._dev:
            self._dev.close()
            self._dev = None
            logging.info("HackRF device closed.")

    def run(self) -> None:
        """
        Start reading samples from HackRF, demodulating, and writing to the output OGG file.
        """
        if not self._dev:
            raise RuntimeError("Device not opened. Call open() first.")

        logging.info(f"Recording to {self.out_file} ... (Ctrl+C to stop if indefinite)")
        with sf.SoundFile(
            self.out_file,
            mode='w',
            samplerate=int(self.audio_rate),
            channels=1,
            format='OGG',
            subtype='VORBIS'
        ) as sfh:
            start_time = time.time()
            chunk_size = 8192

            try:
                for raw_chunk in self._dev.read_samples(
                    num_samples=0,
                    duration=self.duration,
                    chunk_size=chunk_size
                ):
                    iq_u8 = np.frombuffer(raw_chunk, dtype=np.uint8)
                    iq_float = (iq_u8.astype(np.float32) - 128.0)
                    i_samples = iq_float[0::2]
                    q_samples = iq_float[1::2]
                    iq_cplx = (i_samples + 1j * q_samples).astype(np.complex64)

                    audio = self._demod(iq_cplx)
                    sfh.write(audio)

                    if self.duration > 0:
                        if (time.time() - start_time) >= self.duration:
                            break

            except KeyboardInterrupt:
                logging.warning("KeyboardInterrupt - stopping.")
            except Exception as ex:
                logging.error(f"Error in streaming: {ex}")

    def _demod(self, iq_chunk: np.ndarray) -> np.ndarray:
        """
        Internal helper to demodulate an I/Q chunk according to the selected mode.
        """
        if self.mode == 'nfm':
            return demodulate_nfm(iq_chunk, self._demod_state,
                                  self.sample_rate, self.audio_rate,
                                  cutoff=12000)
        elif self.mode == 'am':
            return demodulate_am(iq_chunk, self._demod_state,
                                 self.sample_rate, self.audio_rate,
                                 cutoff=8000)
        elif self.mode == 'usb':
            return demodulate_ssb(iq_chunk, self._demod_state,
                                  self.sample_rate, self.audio_rate,
                                  sideband='usb', audio_bw=3000)
        elif self.mode == 'lsb':
            return demodulate_ssb(iq_chunk, self._demod_state,
                                  self.sample_rate, self.audio_rate,
                                  sideband='lsb', audio_bw=3000)
        elif self.mode == 'wfm':
            # Wide FM broadcast
            return demodulate_wfm(iq_chunk, self._demod_state,
                                  self.sample_rate, self.audio_rate,
                                  if_cutoff=100000.0,
                                  audio_cutoff=15000.0,
                                  deemphasis=7.5e-5)
        else:
            logging.debug(f"Auto-selecting demod mode for unknown '{self.mode}' => default WFM.")
            return demodulate_wfm(iq_chunk, self._demod_state,
                                  self.sample_rate, self.audio_rate)


# =========================================================================
# Command-line interface
# =========================================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments and return an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Tune HackRF to a frequency (or scan for signals), demodulate "
            "(NFM/AM/USB/LSB/WFM), and record audio to an OGG file. Also offers "
            "advanced scanning/tuning for automatic frequency and mode detection."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-f", "--freq",
        type=float,
        default=None,
        help="Tuning frequency in Hz (e.g. 100e6)."
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default=None,
        choices=["nfm", "am", "usb", "lsb", "wfm"],
        help="Demodulation mode. If not specified, auto-detect based on frequency."
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=float,
        default=2e6,
        help="HackRF sample rate in Hz."
    )
    parser.add_argument(
        "-b", "--bandwidth",
        type=float,
        default=None,
        help="Baseband filter bandwidth in Hz (default: same as sample rate)."
    )
    parser.add_argument(
        "-a", "--audio-rate",
        type=float,
        default=48000,
        help="Output audio sample rate in Hz (default: 48000)."
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds. 0 => indefinite until Ctrl+C."
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
        help="LNA gain in dB (0..40 in steps of 8)."
    )
    parser.add_argument(
        "-G", "--vga-gain",
        type=int,
        default=20,
        help="VGA gain in dB (0..62 in steps of 2)."
    )
    parser.add_argument(
        "-P", "--amp",
        action='store_true',
        help="Enable HackRF internal amplifier."
    )
    # Auto-tune or scanning
    parser.add_argument(
        "-T", "--auto-tune",
        action='store_true',
        help="Advanced tune around --freq ± --auto-tune-range in increments of --auto-tune-step."
    )
    parser.add_argument(
        "-S", "--auto-scan",
        action='store_true',
        help="Perform an advanced FFT-based scan over [--freq-start..--freq-end]. "
             "If --freq is given but no start/end, uses freq ± 1 MHz."
    )
    parser.add_argument(
        "-X", "--freq-start",
        type=float,
        default=None,
        help="Start frequency (Hz) for auto-scan."
    )
    parser.add_argument(
        "-Y", "--freq-end",
        type=float,
        default=None,
        help="End frequency (Hz) for auto-scan."
    )
    parser.add_argument(
        "-H", "--scan-threshold",
        type=float,
        default=5.0,
        help="Threshold in dB above noise floor for auto-scan."
    )
    parser.add_argument(
        "-Z", "--scan-fft-size",
        type=int,
        default=1024,
        help="FFT size for auto-scan."
    )
    parser.add_argument(
        "-W", "--scan-dwell",
        type=float,
        default=0.2,
        help="Dwell time in seconds for each step in auto-scan."
    )
    parser.add_argument(
        "-R", "--auto-tune-range",
        type=float,
        default=100000.0,
        help="Range ± around --freq for auto-tune."
    )
    parser.add_argument(
        "-E", "--auto-tune-step",
        type=float,
        default=10000.0,
        help="Step size in Hz for auto-tune."
    )
    # Additional advanced scan parameters
    parser.add_argument(
        "-U", "--adv-scan-steps",
        type=float,
        default=20000.0,
        help="Frequency step size (Hz) in advanced scanning."
    )
    parser.add_argument(
        "-Q", "--adv-scan-quality",
        type=float,
        default=0.01,
        help="Minimum audio 'quality' threshold in advanced scanning."
    )
    parser.add_argument(
        "-N", "--adv-scan-min-snr",
        type=float,
        default=5.0,
        help="Minimum SNR (dB) for a signal to be considered."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v => INFO, -vv => DEBUG."
    )

    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    """
    Configure the logging verbosity based on the given level.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )


def auto_select_mode(freq_hz: float) -> str:
    """
    Attempt to select the best mode automatically based on frequency range.
    Simple heuristics:
      - < 12 MHz => LSB
      - 12 MHz - 30 MHz => USB
      - 88 MHz - 108 MHz => WFM
      - else => NFM
    """
    if freq_hz < 12e6:
        return 'lsb'
    elif freq_hz < 30e6:
        return 'usb'
    elif 88e6 <= freq_hz <= 108e6:
        return 'wfm'
    else:
        return 'nfm'


def main() -> None:
    """
    Entry point for the HackRF listening/recording script with advanced scanning/tuning.
    """
    args = parse_args()
    setup_logging(args.verbose)

    freq: Optional[float] = args.freq
    if freq is not None and freq < 300:  # heuristic for MHz
        logging.info(f"Frequency {freq} is suspiciously low; interpreting as MHz.")
        freq *= 1e6

    # Check if advanced auto-scan is requested
    if args.auto_scan:
        start_f = args.freq_start
        end_f = args.freq_end
        if start_f is None and end_f is None:
            # default ±1 MHz around freq
            if freq is None:
                logging.error("auto-scan requires a --freq or --freq-start/--freq-end.")
                sys.exit(1)
            start_f = freq - 1e6
            end_f = freq + 1e6
        elif start_f is None or end_f is None:
            logging.error("--freq-start and --freq-end must both be specified or left unset together.")
            sys.exit(1)

        logging.info(f"Performing advanced FFT-based auto-scan from {start_f/1e6:.3f} MHz to {end_f/1e6:.3f} MHz.")
        found_freq, found_mode = auto_scan(
            device_index=args.device_index,
            start_freq=start_f,
            end_freq=end_f,
            sample_rate=args.sample_rate,
            bandwidth=args.bandwidth if args.bandwidth else args.sample_rate,
            dwell_time=args.scan_dwell,
            threshold_db=args.scan_threshold,
            fft_size=args.scan_fft_size,
            lna_gain=args.lna_gain,
            vga_gain=args.vga_gain,
            amp=args.amp,
            scan_step=args.adv_scan_steps,
            min_snr_db=args.adv_scan_min_snr,
            min_quality=args.adv_scan_quality
        )
        if found_freq is None or found_mode is None:
            logging.warning("No strong signals found by auto-scan. Exiting.")
            sys.exit(1)
        freq = found_freq
        final_mode = found_mode
        logging.info(f"Auto-scan found best signal at {freq/1e6:.5f} MHz, mode={final_mode}.")

    # Check if advanced auto-tune is requested
    elif args.auto_tune:
        if freq is None:
            logging.error("auto-tune requires an initial --freq.")
            sys.exit(1)

        logging.info(
            f"Performing advanced auto-tune around {freq/1e6:.4f} MHz ± {args.auto_tune_range/1e6:.4f} MHz "
            f"in steps of {args.auto_tune_step/1e3:.1f} kHz."
        )
        found_freq, found_mode = auto_tune(
            device_index=args.device_index,
            base_freq=freq,
            sample_rate=args.sample_rate,
            lna_gain=args.lna_gain,
            vga_gain=args.vga_gain,
            amp=args.amp,
            tune_range=args.auto_tune_range,
            tune_step=args.auto_tune_step,
            min_snr_db=args.adv_scan_min_snr,
            min_quality=args.adv_scan_quality
        )
        if found_freq is None or found_mode is None:
            logging.warning("No strong signals found by auto-tune. Exiting.")
            sys.exit(1)
        freq = found_freq
        final_mode = found_mode
        logging.info(f"Auto-tune found best signal at {freq/1e6:.5f} MHz, mode={final_mode}.")

    else:
        # Normal operation (no advanced scanning/tuning)
        if freq is None:
            logging.error("No frequency specified. Use --auto-scan or --freq.")
            sys.exit(1)

        # If user provided a mode, use it; else auto-detect
        if args.mode:
            final_mode = args.mode
        else:
            final_mode = auto_select_mode(freq)
            logging.info(f"No mode specified; auto-selected mode={final_mode} based on freq={freq/1e6:.3f} MHz.")

    if freq is None:
        logging.error("No valid frequency found. Exiting.")
        sys.exit(1)

    listener = HackRFListener(
        freq=freq,
        sample_rate=args.sample_rate,
        audio_rate=args.audio_rate,
        mode=final_mode,
        duration=args.duration,
        out_file=args.output,
        device_index=args.device_index,
        bandwidth=args.bandwidth,
        lna_gain=args.lna_gain,
        vga_gain=args.vga_gain,
        amp=args.amp
    )

    try:
        listener.open()
        listener.run()
    except KeyboardInterrupt:
        logging.warning("User interrupted. Stopping.")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
