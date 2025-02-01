#!/usr/bin/env python3

# -------------------------------------------------------
# Script: hackrf_scan.py
#
# Description:
# A HackRF-based script that scans a specified frequency range
# (or only known bands) in increments of step_size, acquires multiple
# short bursts of samples, performs FFT(s) on each burst, and detects
# carriers that consistently appear above a threshold relative
# to the noise floor across repeated captures. This helps rule out
# transient spikes or spurious peaks.
#
# Usage:
#   ./hackrf_scan.py [options]
#
# Options:
#   -s, --start-hz FREQ       Start frequency in Hz (default: 2e6).
#   -e, --end-hz FREQ         End frequency in Hz (default: 6e9).
#   -t, --threshold DB        Power threshold in dB above noise floor (default: 5).
#   -r, --sample-rate SR      HackRF sample rate in Hz (default: 2e6).
#   -b, --bandwidth BW        Baseband filter bandwidth in Hz (default: sample rate).
#   -S, --step-size STEP      Frequency step size in Hz (default: 2e6).
#   -F, --fft-size N          FFT size (default: 1024).
#   -d, --dwell-time SEC      Dwell/settle time in seconds after retuning (default: 0.05).
#   -m, --merge-gap-khz KHZ   Merge gap in kHz for contiguous active bins (default: 100).
#   -k, --only-known-bands    If set, only scan known frequency bands (default: off).
#   -L, --lna-gain DB         LNA gain in dB [0..40 in steps of 8] (default: 16).
#   -G, --vga-gain DB         VGA gain in dB [0..62 in steps of 2] (default: 20).
#   -a, --amp                 Enable HackRF internal amplifier (default: off).
#   -R, --robust-floor        Use robust noise floor estimation (clipping) (default: off).
#   --repeat-captures N       Number of short captures to take at each frequency step
#                             (default: 3).
#   --capture-duration SEC    Duration (in seconds) of each short capture (default: 0.02).
#   --stable-ratio FLOAT      Fraction of captures in which a bin must appear to be considered
#                             a stable carrier (default: 0.8). E.g. if repeat-captures=5 and
#                             stable-ratio=0.8, then a bin must be above threshold in at least
#                             4 out of 5 captures.
#   -dI, --device-index IDX   HackRF device index (default: 0).
#   -v, --verbose             Increase verbosity (-v => INFO, -vv => DEBUG).
#
# Examples:
#   1) Quick broad scan from 2 MHz to 6 GHz, minimal dwell time:
#      ./hackrf_scan.py -s 2e6 -e 6e9 -d 0.01 -F 512 --repeat-captures 2
#
#   2) More thorough scan with multiple short captures per step:
#      ./hackrf_scan.py -s 144e6 -e 146e6 -F 1024 --repeat-captures 5 \
#                       --capture-duration 0.025 --stable-ratio 0.75
#
#   3) Only scan known frequency bands within 50 MHz to 200 MHz:
#      ./hackrf_scan.py -s 50e6 -e 200e6 --only-known-bands
#
# IMPORTANT:
#   - This script is for demonstration and educational use.
#     Exact frequency allocations vary by region, country, and service.
#   - Always ensure you comply with local regulations when scanning.
#
# Template: ubuntu22.04
#
# Requirements:
#   - libhackrf (install via: apt-get install -y libhackrf0 libusb-1.0-0-dev)
#   - numpy (install via: pip install numpy==2.2.1)
#   - prettytable (install via: pip install prettytable==3.12.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------


import argparse
import logging
import time
import sys
import numpy as np
import ctypes
import atexit
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from threading import Lock

# Attempt to import PrettyTable (optional)
try:
    from prettytable import PrettyTable

    HAS_PRETTYTABLE = True
except ImportError:
    HAS_PRETTYTABLE = False


# =========================================================================
# HackRF ctypes interface
# =========================================================================

_libhackrf = ctypes.CDLL("libhackrf.so.0")
HACKRF_SUCCESS = 0
_p_hackrf_device = ctypes.c_void_p


class _hackrf_transfer(ctypes.Structure):
    """
    Low-level structure that holds HackRF transfer parameters.
    """

    _fields_ = [
        ("device", _p_hackrf_device),
        ("buffer", ctypes.POINTER(ctypes.c_ubyte)),
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
_libhackrf.hackrf_set_baseband_filter_bandwidth.argtypes = [
    _p_hackrf_device,
    ctypes.c_uint32,
]

_libhackrf.hackrf_set_amp_enable.restype = ctypes.c_int
_libhackrf.hackrf_set_amp_enable.argtypes = [_p_hackrf_device, ctypes.c_uint8]

_libhackrf.hackrf_set_lna_gain.restype = ctypes.c_int
_libhackrf.hackrf_set_lna_gain.argtypes = [_p_hackrf_device, ctypes.c_uint32]

_libhackrf.hackrf_set_vga_gain.restype = ctypes.c_int
_libhackrf.hackrf_set_vga_gain.argtypes = [_p_hackrf_device, ctypes.c_uint32]

_libhackrf.hackrf_set_freq.restype = ctypes.c_int
_libhackrf.hackrf_set_freq.argtypes = [_p_hackrf_device, ctypes.c_uint64]

_libhackrf.hackrf_start_rx.restype = ctypes.c_int
_libhackrf.hackrf_start_rx.argtypes = [
    _p_hackrf_device,
    _transfer_callback,
    ctypes.c_void_p,
]

_libhackrf.hackrf_stop_rx.restype = ctypes.c_int
_libhackrf.hackrf_stop_rx.argtypes = [_p_hackrf_device]

_libhackrf.hackrf_is_streaming.restype = ctypes.c_int
_libhackrf.hackrf_is_streaming.argtypes = [_p_hackrf_device]

_init_result = _libhackrf.hackrf_init()
if _init_result != HACKRF_SUCCESS:
    raise RuntimeError("Error initializing HackRF (libhackrf).")


def _finalize_hackrf() -> None:
    """
    Finalizes the HackRF library on exit.
    """
    _libhackrf.hackrf_exit()


atexit.register(_finalize_hackrf)


def clamp_lna_gain(gain: int) -> int:
    """
    Clamps LNA gain to valid steps: 0..40 in steps of 8.
    """
    valid_lna = [0, 8, 16, 24, 32, 40]
    return min(valid_lna, key=lambda x: abs(x - gain))


def clamp_vga_gain(gain: int) -> int:
    """
    Clamps VGA gain to valid steps: 0..62 in steps of 2.
    """
    valid_vga = list(range(0, 63, 2))
    return min(valid_vga, key=lambda x: abs(x - gain))


def get_valid_hackrf_bandwidth(desired_bw: float) -> int:
    """
    Maps a desired baseband filter bandwidth to the closest valid HackRF filter.
    """
    valid_bw = [
        1750000,
        2500000,
        5000000,
        5500000,
        6000000,
        7500000,
        10000000,
        15000000,
        20000000,
        24000000,
        28000000,
    ]
    if desired_bw <= valid_bw[0]:
        return valid_bw[0]
    if desired_bw >= valid_bw[-1]:
        return valid_bw[-1]
    return int(min(valid_bw, key=lambda x: abs(x - desired_bw)))


@dataclass
class FrequencyBand:
    """
    Holds information about a known or generic frequency band.
    """

    name: str
    freq_start: float
    freq_end: float
    typical_bandwidth_hz: float
    metadata: Dict[str, Any]


@dataclass
class ActiveRange:
    """
    Represents an active sub-range with a peak power and optional classification.
    """

    start_hz: float
    end_hz: float
    peak_power_db: float
    likely_protocol: Optional[str] = None


KNOWN_BANDS: List[FrequencyBand] = [
    FrequencyBand(
        name="Shortwave 31m Broadcast",
        freq_start=9.4e6,
        freq_end=9.9e6,
        typical_bandwidth_hz=5e3,
        metadata={"modulation": "AM/SSB"},
    ),
    FrequencyBand(
        name="FM Broadcast",
        freq_start=87.5e6,
        freq_end=108e6,
        typical_bandwidth_hz=200e3,
        metadata={"modulation": "WFM"},
    ),
    FrequencyBand(
        name="Air Band (AM)",
        freq_start=118e6,
        freq_end=137e6,
        typical_bandwidth_hz=25e3,
        metadata={"modulation": "AM"},
    ),
    FrequencyBand(
        name="2m Amateur Radio",
        freq_start=144e6,
        freq_end=148e6,
        typical_bandwidth_hz=16e3,
        metadata={"modulation": "NFM"},
    ),
    FrequencyBand(
        name="NOAA Weather Radio",
        freq_start=162.4e6,
        freq_end=162.55e6,
        typical_bandwidth_hz=25e3,
        metadata={"modulation": "FM"},
    ),
    FrequencyBand(
        name="70cm Amateur Radio",
        freq_start=430e6,
        freq_end=440e6,
        typical_bandwidth_hz=25e3,
        metadata={"modulation": "NFM/SSB"},
    ),
    FrequencyBand(
        name="L-band Aero",
        freq_start=960e6,
        freq_end=1215e6,
        typical_bandwidth_hz=1e6,
        metadata={"comment": "Aero nav signals"},
    ),
    FrequencyBand(
        name="Wi-Fi 2.4 GHz",
        freq_start=2400e6,
        freq_end=2483.5e6,
        typical_bandwidth_hz=20e6,
        metadata={"modulation": "OFDM"},
    ),
    FrequencyBand(
        name="Bluetooth 2.4 GHz",
        freq_start=2402e6,
        freq_end=2480e6,
        typical_bandwidth_hz=2e6,
        metadata={"modulation": "GFSK"},
    ),
    FrequencyBand(
        name="5 GHz Wi-Fi",
        freq_start=5150e6,
        freq_end=5850e6,
        typical_bandwidth_hz=20e6,
        metadata={"modulation": "OFDM"},
    ),
    FrequencyBand(
        name="C-Band Radar",
        freq_start=5250e6,
        freq_end=5925e6,
        typical_bandwidth_hz=80e6,
        metadata={"modulation": "Pulse/Chirp"},
    ),
]


GENERIC_BANDS: List[FrequencyBand] = [
    FrequencyBand(
        name="Generic HF (3-30 MHz)",
        freq_start=3e6,
        freq_end=30e6,
        typical_bandwidth_hz=0,
        metadata={},
    ),
    FrequencyBand(
        name="Generic VHF (30-300 MHz)",
        freq_start=30e6,
        freq_end=300e6,
        typical_bandwidth_hz=0,
        metadata={},
    ),
    FrequencyBand(
        name="Generic UHF (300 MHz-1 GHz)",
        freq_start=300e6,
        freq_end=1e9,
        typical_bandwidth_hz=0,
        metadata={},
    ),
    FrequencyBand(
        name="Generic L-Band (1-2 GHz)",
        freq_start=1e9,
        freq_end=2e9,
        typical_bandwidth_hz=0,
        metadata={},
    ),
    FrequencyBand(
        name="Generic S-Band (2-4 GHz)",
        freq_start=2e9,
        freq_end=4e9,
        typical_bandwidth_hz=0,
        metadata={},
    ),
    FrequencyBand(
        name="Generic C-Band (4-8 GHz)",
        freq_start=4e9,
        freq_end=8e9,
        typical_bandwidth_hz=0,
        metadata={},
    ),
]


def merge_band_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merges overlapping band intervals into larger contiguous intervals.
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = sorted_ranges[0]
    for s, e in sorted_ranges[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


def overlap_fraction(st1: float, en1: float, st2: float, en2: float) -> float:
    """
    Returns the fraction of overlap between two intervals over the smaller interval.
    """
    if en1 < st2 or en2 < st1:
        return 0.0
    overlap_start = max(st1, st2)
    overlap_end = min(en1, en2)
    overlap_len = overlap_end - overlap_start
    if overlap_len <= 0:
        return 0.0
    span1 = en1 - st1
    span2 = en2 - st2
    small_span = min(span1, span2)
    return overlap_len / small_span if small_span > 0 else 0.0


def merge_active_bins(
    bins: List[Tuple[float, float, float]], gap_khz: float
) -> List[Tuple[float, float, float]]:
    """
    Merges adjacent or overlapping bins whose boundaries are within a certain gap.
    """
    if not bins:
        return []
    gap_hz = gap_khz * 1000
    sorted_bins = sorted(bins, key=lambda x: x[0])
    merged: List[Tuple[float, float, float]] = []
    cur_start, cur_end, cur_power = sorted_bins[0]
    for s, e, p in sorted_bins[1:]:
        if s <= (cur_end + gap_hz):
            cur_end = max(cur_end, e)
            cur_power = max(cur_power, p)  # keep highest peak
        else:
            merged.append((cur_start, cur_end, cur_power))
            cur_start, cur_end, cur_power = s, e, p
    merged.append((cur_start, cur_end, cur_power))
    return merged


def compute_noise_floor_linear(power_arr: np.ndarray, robust: bool) -> float:
    """
    Estimates the noise floor in linear domain, optionally clipping outliers.
    """
    if not len(power_arr):
        return 1e-12
    if robust:
        # 5-95 percentile clip in linear domain
        low = np.percentile(power_arr, 5)
        high = np.percentile(power_arr, 95)
        clipped = np.clip(power_arr, low, high)
        return float(np.median(clipped))
    else:
        # simpler median in linear domain
        return float(np.median(power_arr))


class HackRFDevice:
    """
    Wraps low-level HackRF device access with a ring buffer for continuous streaming.
    """

    def __init__(self, device_index: int = 0):
        """
        Opens a HackRF device for streaming at the given index.
        """
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

        # Increased ring buffer size to help avoid dropping data
        self._max_buffer_size = 2**22  # 4 MB ring buffer

        self._buffer = bytearray()
        self._ring_lock = Lock()

    def close(self) -> None:
        """
        Closes the HackRF device and stops streaming if active.
        """
        if self._dev:
            if self._streaming:
                _libhackrf.hackrf_stop_rx(self._dev)
                self._streaming = False
            _libhackrf.hackrf_close(self._dev)
            self._dev = _p_hackrf_device(None)

    def __del__(self) -> None:
        """
        Destructor to ensure closure of the HackRF device.
        """
        self.close()

    def _rx_callback(self, transfer_ptr: ctypes.POINTER(_hackrf_transfer)) -> int:
        """
        Callback that receives IQ samples from the HackRF device. Maintains
        a ring buffer of newly received data. Uses a blocking lock to avoid
        dropping data.
        """
        transfer = transfer_ptr.contents
        length = transfer.valid_length
        if length > 0:
            arr_type = ctypes.c_ubyte * length
            data_array = ctypes.cast(transfer.buffer, ctypes.POINTER(arr_type)).contents

            # Acquire the lock *blocking* to avoid dropping samples
            with self._ring_lock:
                needed_space = len(self._buffer) + length
                if needed_space > self._max_buffer_size:
                    remove_count = needed_space - self._max_buffer_size
                    del self._buffer[:remove_count]
                self._buffer.extend(data_array)

        return 0

    def start_streaming(self) -> None:
        """
        Starts continuous RX streaming on the HackRF device.
        """
        if self._streaming:
            return
        result = _libhackrf.hackrf_start_rx(self._dev, self._rx_callback_function, None)
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error starting continuous RX, code={result}")
        self._streaming = True

    def set_sample_rate(self, sr_hz: float) -> None:
        """
        Sets the sample rate of the HackRF device.
        """
        sr = ctypes.c_double(sr_hz)
        result = _libhackrf.hackrf_set_sample_rate(self._dev, sr)
        if result != HACKRF_SUCCESS:
            raise RuntimeError(
                f"Error setting sample rate to {sr_hz} Hz, code={result}"
            )

    def set_baseband_filter_bandwidth(self, bw_hz: float) -> None:
        """
        Sets the baseband filter bandwidth on the HackRF device.
        """
        valid_bw = get_valid_hackrf_bandwidth(bw_hz)
        result = _libhackrf.hackrf_set_baseband_filter_bandwidth(
            self._dev, ctypes.c_uint32(valid_bw)
        )
        if result != HACKRF_SUCCESS:
            raise RuntimeError(
                f"Error setting baseband filter BW to {valid_bw} Hz, code={result}"
            )

    def set_freq(self, freq_hz: float) -> None:
        """
        Sets the LO frequency on the HackRF device, clamping to [1e6, 6e9] if needed.
        """
        if freq_hz < 1e6:
            logging.warning(
                f"Requested frequency {freq_hz / 1e6:.3f} MHz < 1 MHz. Clamping to 1 MHz."
            )
            freq_hz = 1e6
        elif freq_hz > 6e9:
            logging.warning(
                f"Requested frequency {freq_hz / 1e6:.3f} MHz > 6 GHz. Clamping to 6 GHz."
            )
            freq_hz = 6e9
        result = _libhackrf.hackrf_set_freq(self._dev, ctypes.c_uint64(int(freq_hz)))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(
                f"Error setting frequency to {freq_hz} Hz, code={result}"
            )

    def set_amp_enable(self, enable: bool) -> None:
        """
        Enables or disables the HackRF's built-in amplifier.
        """
        val = 1 if enable else 0
        result = _libhackrf.hackrf_set_amp_enable(self._dev, ctypes.c_uint8(val))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting amp enable={enable}, code={result}")

    def set_lna_gain(self, gain: int) -> None:
        """
        Sets the LNA gain with valid clamping in steps of 8 dB.
        """
        g = clamp_lna_gain(gain)
        result = _libhackrf.hackrf_set_lna_gain(self._dev, ctypes.c_uint32(g))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting LNA gain={g}, code={result}")

    def set_vga_gain(self, gain: int) -> None:
        """
        Sets the VGA gain with valid clamping in steps of 2 dB.
        """
        g = clamp_vga_gain(gain)
        result = _libhackrf.hackrf_set_vga_gain(self._dev, ctypes.c_uint32(g))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting VGA gain={g}, code={result}")

    def clear_buffer(self) -> None:
        """
        Clears the ring buffer of any old IQ data.
        """
        with self._ring_lock:
            self._buffer.clear()

    def read_samples_blocking(self, num_bytes: int, timeout_s: float = 1.0) -> bytes:
        """
        Reads exactly num_bytes from the ring buffer, blocking until we get enough
        or until timeout. Then zero-pads with 128 if not enough data arrived.
        """
        start_t = time.time()
        out = bytearray()

        while len(out) < num_bytes:
            with self._ring_lock:
                chunk_len = len(self._buffer)
                if chunk_len > 0:
                    take = min(chunk_len, num_bytes - len(out))
                    out.extend(self._buffer[:take])
                    del self._buffer[:take]

            if len(out) >= num_bytes:
                break

            if (time.time() - start_t) >= timeout_s:
                break  # Timeout
            time.sleep(0.001)

        # If still not enough, zero-pad with 128 (HackRF IQ midpoint)
        if len(out) < num_bytes:
            needed = num_bytes - len(out)
            out += bytes([128] * needed)
            logging.debug(
                f"Partial read - got {len(out) - needed} bytes, padded {needed} bytes with 128."
            )

        return bytes(out)


class HackRFScanner:
    """
    Implements a scanner that steps through frequencies with a HackRF device
    and detects active signals above a threshold, ensuring consistency
    across multiple short captures.
    """

    def __init__(
        self,
        start_hz: float,
        end_hz: float,
        step_size: float,
        sample_rate: float,
        bandwidth: float,
        threshold_db: float,
        fft_size: int,
        dwell_time: float,
        merge_gap_khz: float,
        only_known_bands: bool,
        device_index: int,
        lna_gain: int,
        vga_gain: int,
        amp: bool,
        robust_floor: bool,
        repeat_captures: int = 3,
        capture_duration: float = 0.02,
        stable_ratio: float = 0.8,
        suppress_dc_bin: bool = True,
    ) -> None:
        """
        Creates a HackRFScanner object with the given parameters.
        - repeat_captures: how many short captures to do per LO step
        - capture_duration: each short capture length in seconds
        - stable_ratio: fraction of captures in which a bin must appear above threshold
          to be considered an actual carrier
        """
        if start_hz < 1e6:
            logging.warning(
                f"Start frequency {start_hz / 1e6:.3f} MHz < 1 MHz. Clamping to 1 MHz."
            )
            start_hz = 1e6
        if end_hz > 6e9:
            logging.warning(
                f"End frequency {end_hz / 1e6:.3f} MHz > 6 GHz. Clamping to 6 GHz."
            )
            end_hz = 6e9

        self.start_hz = start_hz
        self.end_hz = end_hz
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.threshold_db = threshold_db
        self.fft_size = fft_size
        self.dwell_time = dwell_time
        self.merge_gap_khz = merge_gap_khz
        self.only_known_bands = only_known_bands
        self.device_index = device_index
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain
        self.amp = amp
        self.robust_floor = robust_floor
        self.suppress_dc_bin = suppress_dc_bin

        self.repeat_captures = repeat_captures
        self.capture_duration = capture_duration
        self.stable_ratio = stable_ratio

        # Final list of discovered active ranges
        self.active_ranges: List[ActiveRange] = []
        self._device: Optional[HackRFDevice] = None

        # Warn if step_size == sample_rate => potential coverage gaps
        if abs(self.step_size - self.sample_rate) < 1e-9:
            logging.info(
                "Warning: step_size == sample_rate. Edge coverage may be incomplete. "
                "Consider a smaller step_size for overlap."
            )

    def open(self) -> None:
        """
        Opens the HackRF device and starts continuous streaming.
        """
        self._device = HackRFDevice(device_index=self.device_index)
        self._device.set_sample_rate(self.sample_rate)
        bw = self.bandwidth if self.bandwidth is not None else self.sample_rate
        self._device.set_baseband_filter_bandwidth(bw)
        self._device.set_lna_gain(self.lna_gain)
        self._device.set_vga_gain(self.vga_gain)
        self._device.set_amp_enable(self.amp)
        self._device.start_streaming()
        logging.info(
            f"Opened HackRF idx={self.device_index}, sr={self.sample_rate}, bw={bw}, "
            f"LNA={self.lna_gain}, VGA={self.vga_gain}, amp={self.amp}"
        )

    def close(self) -> None:
        """
        Closes the HackRF device.
        """
        if self._device:
            self._device.close()
            self._device = None
        logging.info("HackRF device closed.")

    def scan(self) -> None:
        """
        Performs the frequency sweep in increments of step_size,
        and detects stable carriers above the threshold.
        """
        if not self._device:
            raise RuntimeError("HackRF device not open. Call .open() first.")

        freq_plan = self._build_frequency_plan()
        logging.info(f"Scanning {len(freq_plan)} center frequencies...")

        bytes_per_frame = self.fft_size * 2  # (I+Q) bytes per sample

        for cfreq in freq_plan:
            try:
                self._device.set_freq(cfreq)
            except RuntimeError as e:
                logging.debug(f"Skipping freq={cfreq / 1e6:.3f} MHz: {e}")
                continue

            # Flush old data so we only collect fresh samples at new freq
            self._device.clear_buffer()
            time.sleep(self.dwell_time)

            # We do repeat_captures short captures, each capture is `capture_duration` seconds
            # Then we accumulate sets of 'active bins' for each capture
            all_bins: List[List[Tuple[float, float, float]]] = []

            for _capture_idx in range(self.repeat_captures):
                # read enough bytes for capture_duration
                capture_samples = int(self.capture_duration * self.sample_rate)
                capture_bytes = capture_samples * 2

                raw_bytes = self._device.read_samples_blocking(
                    capture_bytes, timeout_s=self.capture_duration * 5
                )
                cplx_iq = self._convert_u8_iq(raw_bytes)

                # apply window, compute FFT
                window = np.hanning(self.fft_size)
                if len(cplx_iq) < self.fft_size:
                    # pad with zeros if short
                    pad_len = self.fft_size - len(cplx_iq)
                    cplx_iq = np.concatenate(
                        [cplx_iq, np.zeros(pad_len, dtype=np.complex64)]
                    )
                else:
                    cplx_iq = cplx_iq[: self.fft_size]

                cplx_win = cplx_iq * window
                fft_out = np.fft.fftshift(np.fft.fft(cplx_win, n=self.fft_size))
                power_lin = np.abs(fft_out) ** 2

                # estimate noise floor in linear domain
                noise_lin = compute_noise_floor_linear(
                    power_lin, robust=self.robust_floor
                )
                noise_db = 10.0 * np.log10(noise_lin + 1e-12)
                psd_db = 10.0 * np.log10(power_lin + 1e-12)

                # optionally suppress DC bin
                if self.suppress_dc_bin and self.fft_size > 2:
                    dc_idx = self.fft_size // 2
                    psd_db[dc_idx] = noise_db - 100.0

                threshold_val = noise_db + self.threshold_db

                freq_axis = np.linspace(
                    -0.5 * self.sample_rate,
                    0.5 * self.sample_rate,
                    self.fft_size,
                    endpoint=False,
                )

                # find bins above threshold
                mask = psd_db >= threshold_val
                local_bins: List[Tuple[float, float, float]] = []

                i = 0
                while i < self.fft_size:
                    if mask[i]:
                        start_i = i
                        while i < self.fft_size and mask[i]:
                            i += 1
                        end_i = i - 1
                        peak_power = float(np.max(psd_db[start_i : end_i + 1]))
                        start_off = freq_axis[start_i]
                        end_off = freq_axis[end_i]
                        abs_start = cfreq + start_off
                        abs_end = cfreq + end_off
                        if abs_start > abs_end:
                            abs_start, abs_end = abs_end, abs_start
                        local_bins.append((abs_start, abs_end, peak_power))
                    else:
                        i += 1

                # merge bins
                merged_capture_bins = merge_active_bins(local_bins, self.merge_gap_khz)
                all_bins.append(merged_capture_bins)

            # Now we have all_bins for each capture. We need to find stable bins
            # that appear in at least stable_ratio * repeat_captures captures.
            stable_bins = self._find_stable_bins(all_bins)

            for s, e, p in stable_bins:
                self.active_ranges.append(
                    ActiveRange(
                        start_hz=s,
                        end_hz=e,
                        peak_power_db=p,
                        likely_protocol=self._classify_range(s, e),
                    )
                )

            logging.debug(
                f"[{cfreq / 1e6:.3f} MHz] found {len(stable_bins)} stable bin(s)."
            )

    def _find_stable_bins(
        self, captures_bins: List[List[Tuple[float, float, float]]]
    ) -> List[Tuple[float, float, float]]:
        """
        Given a list of merged bins for each capture, find bins that appear
        in at least stable_ratio * repeat_captures captures. We do a naive approach:
          1) Flatten all bins, but keep a reference to which capture they came from
          2) For each unique bin (by approximate frequency overlap), count how many captures had it
          3) If count >= stable_count_threshold, it's stable
          4) Merge the stable bins again

        This approach helps rule out spurious peaks that only appear in 1 capture.
        """
        # Flatten with capture index
        # We'll store: (capture_idx, start, end, peak)
        flattened: List[Tuple[int, float, float, float]] = []
        for cidx, bins in enumerate(captures_bins):
            for s, e, p in bins:
                flattened.append((cidx, s, e, p))

        if not flattened:
            return []

        # In order to consider two bins the "same," they must overlap significantly in freq
        stable_count_needed = int(np.ceil(self.repeat_captures * self.stable_ratio))

        # We'll group by approx overlapping frequencies. A quick approach:
        # sort by start freq, then do a pass to group overlaps if they have a decent overlap
        flattened.sort(key=lambda x: x[1])  # sort by start
        groups: List[List[Tuple[int, float, float, float]]] = []
        current_group: List[Tuple[int, float, float, float]] = [flattened[0]]

        def bins_overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
            if e1 < s2 or e2 < s1:
                return False
            return True

        for cidx, s, e, p in flattened[1:]:
            last_s, last_e = current_group[-1][1], current_group[-1][2]
            if bins_overlap(last_s, last_e, s, e):
                # same group
                current_group.append((cidx, s, e, p))
            else:
                groups.append(current_group)
                current_group = [(cidx, s, e, p)]
        groups.append(current_group)

        stable_bins: List[Tuple[float, float, float]] = []
        for grp in groups:
            # check how many distinct captures are in this group
            captures_hit = set([g[0] for g in grp])  # capture_idx
            if len(captures_hit) >= stable_count_needed:
                # It's stable
                # We define the final freq range as min_start..max_end,
                # peak power = max of p
                min_s = min([g[1] for g in grp])
                max_e = max([g[2] for g in grp])
                peak_p = max([g[3] for g in grp])
                stable_bins.append((min_s, max_e, peak_p))

        # final merge again, in case groups can overlap each other
        stable_merged = merge_active_bins(stable_bins, self.merge_gap_khz)
        return stable_merged

    def _build_frequency_plan(self) -> List[float]:
        """
        Builds a list of LO center frequencies to scan based on user inputs
        and known-band-only mode if specified. Avoids duplicates at the end.
        """
        if self.only_known_bands:
            band_ranges = []
            for b in KNOWN_BANDS:
                # Skip if out of range
                if b.freq_end < self.start_hz or b.freq_start > self.end_hz:
                    continue
                bs = max(b.freq_start, self.start_hz)
                be = min(b.freq_end, self.end_hz)
                band_ranges.append((bs, be))
            band_ranges = merge_band_ranges(band_ranges)
            logging.info(
                f"Scanning only known bands within "
                f"{self.start_hz / 1e6:.3f}-{self.end_hz / 1e6:.3f} MHz. "
                f"Found {len(band_ranges)} band range(s)."
            )
            freq_plan: List[float] = []
            for bs, be in band_ranges:
                freq_plan.extend(self._create_freq_steps_for_range(bs, be))
            return freq_plan
        else:
            return self._create_freq_steps_for_range(self.start_hz, self.end_hz)

    def _create_freq_steps_for_range(
        self, rng_start: float, rng_end: float
    ) -> List[float]:
        """
        Creates a frequency plan for a single range [rng_start, rng_end],
        stepping in increments of self.step_size, and avoiding duplicates at the end.
        """
        plan = []
        f = rng_start
        epsilon = 1e-9
        while f < rng_end - epsilon:
            plan.append(f)
            f += self.step_size
        # Append the end frequency if not already effectively covered
        if not plan or (rng_end - plan[-1] > epsilon):
            plan.append(rng_end)
        return plan

    @staticmethod
    def _convert_u8_iq(raw_bytes: bytes) -> np.ndarray:
        """
        Converts interleaved unsigned 8-bit IQ samples to a complex64 NumPy array.
        """
        arr_u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
        arr_f = arr_u8.astype(np.float32) - 128.0
        i_samples = arr_f[0::2]
        q_samples = arr_f[1::2]
        return (i_samples + 1j * q_samples).astype(np.complex64)

    @staticmethod
    def _compute_merged_classification(
        sub_ranges: List[Tuple[float, float, float, Optional[str]]],
        final_start: float,
        final_end: float,
    ) -> str:
        """
        For a group of sub-ranges that are merging, compute which classification
        covers the largest fraction of (final_start, final_end). If there's a tie,
        return "Multiple", otherwise return the single classification name.
        """
        if not sub_ranges:
            return "Unknown/Other"
        total_span = final_end - final_start
        if total_span <= 0:
            return "Unknown/Other"

        # Sum coverage for each classification
        coverage_map: Dict[str, float] = {}
        for s, e, _, proto in sub_ranges:
            if proto is None:
                proto = "Unknown/Other"
            # Overlap with final range
            seg_start = max(final_start, s)
            seg_end = min(final_end, e)
            seg_len = seg_end - seg_start
            if seg_len > 0:
                coverage_map[proto] = coverage_map.get(proto, 0.0) + seg_len

        # Pick the classification with the largest coverage
        if not coverage_map:
            return "Unknown/Other"

        best_classes: List[str] = []
        best_coverage = max(coverage_map.values())
        for c, cov in coverage_map.items():
            if abs(cov - best_coverage) < 1e-12:
                best_classes.append(c)

        if len(best_classes) == 1:
            return best_classes[0]
        else:
            return "Multiple"

    @staticmethod
    def merge_final_ranges(
        ranges: List[ActiveRange], gap_khz: float
    ) -> List[ActiveRange]:
        """
        Merges overlapping ActiveRanges across all LO steps, respecting gap_khz.
        - The classification in the final merged range is chosen by coverage fraction:
          whichever classification's sub-ranges collectively cover the largest portion
          of the final span. If multiple tie, final classification is "Multiple".
        - The peak power is the maximum among the sub-ranges.
        """
        if not ranges:
            return []

        # Convert to (start, end, power, classification)
        temp = [
            (r.start_hz, r.end_hz, r.peak_power_db, r.likely_protocol) for r in ranges
        ]
        temp.sort(key=lambda x: x[0])

        merged: List[ActiveRange] = []
        gap_hz = gap_khz * 1000

        cur_block = [temp[0]]
        cur_start, cur_end, cur_power, _ = temp[0]

        for s, e, p, proto in temp[1:]:
            if s <= (cur_end + gap_hz):
                cur_end = max(cur_end, e)
                cur_power = max(cur_power, p)
                cur_block.append((s, e, p, proto))
            else:
                final_class = HackRFScanner._compute_merged_classification(
                    cur_block, cur_start, cur_end
                )
                merged.append(
                    ActiveRange(
                        start_hz=cur_start,
                        end_hz=cur_end,
                        peak_power_db=cur_power,
                        likely_protocol=final_class,
                    )
                )
                cur_block = [(s, e, p, proto)]
                cur_start, cur_end, cur_power, _ = (s, e, p, proto)

        final_class = HackRFScanner._compute_merged_classification(
            cur_block, cur_start, cur_end
        )
        merged.append(
            ActiveRange(
                start_hz=cur_start,
                end_hz=cur_end,
                peak_power_db=cur_power,
                likely_protocol=final_class,
            )
        )

        return merged

    def _classify_range(self, fstart: float, fend: float) -> str:
        """
        Classifies a frequency range by comparing overlap with known and generic bands,
        returning the band name that has the largest overlap fraction. If none match,
        returns "Unknown/Other".
        """
        best_known = ("Unknown/Other", 0.0)
        for band in KNOWN_BANDS:
            of = overlap_fraction(fstart, fend, band.freq_start, band.freq_end)
            if of > best_known[1]:
                best_known = (band.name, of)

        if best_known[1] > 0.0:
            return best_known[0]

        # fallback to generic bands
        best_generic = ("Unknown/Other", 0.0)
        for gband in GENERIC_BANDS:
            of = overlap_fraction(fstart, fend, gband.freq_start, gband.freq_end)
            if of > best_generic[1]:
                best_generic = (gband.name, of)

        if best_generic[1] > 0.0:
            return best_generic[0]

        return "Unknown/Other"


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments and returns them.
    """
    parser = argparse.ArgumentParser(
        description="A HackRF-based FFT scanner for detecting stable carriers above threshold.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--start-hz",
        type=float,
        default=2e6,
        help="Start frequency in Hz.",
    )
    parser.add_argument(
        "-e",
        "--end-hz",
        type=float,
        default=6e9,
        help="End frequency in Hz.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=5.0,
        help="Power threshold in dB above noise floor.",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=float,
        default=2e6,
        help="HackRF sample rate in Hz.",
    )
    parser.add_argument(
        "-b",
        "--bandwidth",
        type=float,
        default=None,
        help="Baseband filter bandwidth in Hz (default: same as sample rate).",
    )
    parser.add_argument(
        "-S",
        "--step-size",
        type=float,
        default=2e6,
        help="Frequency step size in Hz.",
    )
    parser.add_argument(
        "-F",
        "--fft-size",
        type=int,
        default=1024,
        help="FFT size (default: 1024).",
    )
    parser.add_argument(
        "-d",
        "--dwell-time",
        type=float,
        default=0.05,
        help="Dwell/settle time in seconds after retuning.",
    )
    parser.add_argument(
        "-m",
        "--merge-gap-khz",
        type=float,
        default=100.0,
        help="Merge gap in kHz for adjacent active bins.",
    )
    parser.add_argument(
        "-k",
        "--only-known-bands",
        action="store_true",
        help="Only scan known frequency bands from the database.",
    )
    parser.add_argument(
        "-L",
        "--lna-gain",
        type=int,
        default=16,
        help="LNA gain in dB [0..40 in steps of 8].",
    )
    parser.add_argument(
        "-G",
        "--vga-gain",
        type=int,
        default=20,
        help="VGA gain in dB [0..62 in steps of 2].",
    )
    parser.add_argument(
        "-a",
        "--amp",
        action="store_true",
        help="Enable HackRF internal amplifier.",
    )
    parser.add_argument(
        "-R",
        "--robust-floor",
        action="store_true",
        help="Use percentile-based clipping for noise floor (slower).",
    )
    parser.add_argument(
        "--repeat-captures",
        type=int,
        default=3,
        help="Number of short captures to take per frequency step.",
    )
    parser.add_argument(
        "--capture-duration",
        type=float,
        default=0.02,
        help="Duration of each short capture in seconds.",
    )
    parser.add_argument(
        "--stable-ratio",
        type=float,
        default=0.8,
        help="Fraction of captures in which a bin must appear above threshold.",
    )
    parser.add_argument(
        "-dI",
        "--device-index",
        type=int,
        default=0,
        help="HackRF device index.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v => INFO, -vv => DEBUG).",
    )

    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    """
    Sets up Python's logging module based on the verbosity level.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """
    Main entry point for the script. Parses arguments, runs the scan, prints results.
    """
    args = parse_args()
    setup_logging(args.verbose)

    if args.bandwidth is None:
        args.bandwidth = args.sample_rate

    scanner = HackRFScanner(
        start_hz=args.start_hz,
        end_hz=args.end_hz,
        step_size=args.step_size,
        sample_rate=args.sample_rate,
        bandwidth=args.bandwidth,
        threshold_db=args.threshold,
        fft_size=args.fft_size,
        dwell_time=args.dwell_time,
        merge_gap_khz=args.merge_gap_khz,
        only_known_bands=args.only_known_bands,
        device_index=args.device_index,
        lna_gain=args.lna_gain,
        vga_gain=args.vga_gain,
        amp=args.amp,
        robust_floor=args.robust_floor,
        repeat_captures=args.repeat_captures,
        capture_duration=args.capture_duration,
        stable_ratio=args.stable_ratio,
    )

    try:
        scanner.open()
        scanner.scan()
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt received. Stopping.")
    except Exception as e:
        logging.error(f"Error during scanning: {e}")
        sys.exit(1)
    finally:
        scanner.close()

    # Final global merge of all sub-ranges found in different LO steps
    merged_ranges = scanner.merge_final_ranges(
        scanner.active_ranges, scanner.merge_gap_khz
    )

    if not merged_ranges:
        print("No stable carriers detected above threshold.")
        return

    if HAS_PRETTYTABLE:
        table = PrettyTable()
        table.field_names = [
            "#",
            "Start (MHz)",
            "End (MHz)",
            "Peak Power (dB)",
            "Protocol",
        ]
        for i, ar in enumerate(merged_ranges, start=1):
            table.add_row(
                [
                    i,
                    f"{ar.start_hz / 1e6:.3f}",
                    f"{ar.end_hz / 1e6:.3f}",
                    f"{ar.peak_power_db:.2f}",
                    ar.likely_protocol,
                ]
            )
        print(table)
    else:
        print("Detected stable carriers (above threshold) in frequency ranges:")
        for idx, ar in enumerate(merged_ranges, start=1):
            print(
                f"[{idx}] {ar.start_hz / 1e6:.3f} MHz - {ar.end_hz / 1e6:.3f} MHz "
                f"| Peak Power: {ar.peak_power_db:.2f} dB "
                f"| Protocol: {ar.likely_protocol}"
            )


if __name__ == "__main__":
    main()
