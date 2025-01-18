#!/usr/bin/env python3

# -------------------------------------------------------
# Script: hackrf_scan.py
#
# Description:
# This script uses a HackRF device to scan a given frequency
# range and detect actively broadcasting sub-ranges. It performs
# an FFT within each step, identifies peaks above a threshold,
# and merges contiguous frequency bins into sub-ranges. For each
# group of active frequencies, it attempts to identify a likely
# protocol or usage.
#
# Usage:
#   ./hackrf_scan.py [options]
#
# Options:
#   -s, --start-hz FREQ        Start frequency in Hz (default: 2e6).
#   -e, --end-hz FREQ          End frequency in Hz   (default: 6e9).
#   -t, --threshold DB         Power threshold in dB above noise floor
#                              for considering a frequency bin 'active'
#                              (default: 5).
#   -r, --sample-rate SR       Sample rate in Hz (default: 2e6).
#   -b, --bandwidth BW         Receive bandwidth in Hz (default: 2e6).
#   -S, --step-size HZ         Frequency step size in Hz (default: 2e6).
#   -n, --fft-size N           FFT size (default: 1024).
#   -w, --dwell-time SEC       Dwell time in seconds (default: 0.05).
#   -m, --merge-gap-khz KHZ    Merge gap in kHz for active bin merging
#                              (default: 100.0).
#   -k, --only-known-bands     If set, only scan known frequency bands
#                              from the internal database (default: off).
#   -p, --preset NAME          Preset scan mode: fast, reasonable, slow
#                              (overrides some defaults).
#   -d, --device-index IDX     HackRF device index (default: 0).
#   -v, --verbose              Increase verbosity (INFO).
#   -vv, --debug               Increase verbosity further (DEBUG).
#
# IMPORTANT:
#   - This script is for demonstration and educational use.
#     Exact frequency allocations vary by region, country, and service.
#   - Always ensure you comply with local regulations when scanning.
#
# Template: ubuntu22.04
#
# Requirements:
#   - libhackrf (install via: apt-get install -y libhackrf0)
#   - numpy (install via: pip install numpy==2.2.1)
#   - prettytable (install via: pip install prettytable==3.12.0)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import logging
import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import ctypes
import atexit

# Attempt to import PrettyTable (optional)
try:
    from prettytable import PrettyTable
    HAS_PRETTYTABLE = True
except ImportError:
    HAS_PRETTYTABLE = False


# =========================================================================
# Direct libhackrf (via ctypes) code
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
_libhackrf.hackrf_start_rx.argtypes = [_p_hackrf_device, _transfer_callback, ctypes.c_void_p]

_libhackrf.hackrf_stop_rx.restype = ctypes.c_int
_libhackrf.hackrf_stop_rx.argtypes = [_p_hackrf_device]

_libhackrf.hackrf_is_streaming.restype = ctypes.c_int
_libhackrf.hackrf_is_streaming.argtypes = [_p_hackrf_device]

_init_result = _libhackrf.hackrf_init()
if _init_result != HACKRF_SUCCESS:
    raise RuntimeError("Error initializing HackRF (libhackrf).")


def _finalize_hackrf():
    _libhackrf.hackrf_exit()


atexit.register(_finalize_hackrf)


class HackRFDevice:
    """
    Minimal HackRF device wrapper for receiving.
    """

    def __init__(self, device_index: int = 0):
        devlist_handle = _libhackrf.hackrf_device_list()
        if not devlist_handle:
            raise RuntimeError("No HackRF devices found (device list is NULL).")

        self._dev = _p_hackrf_device(None)
        result = _libhackrf.hackrf_device_list_open(
            devlist_handle, device_index, ctypes.byref(self._dev)
        )
        if result != HACKRF_SUCCESS or not self._dev:
            raise RuntimeError(
                f"Could not open HackRF device index {device_index}. Error code: {result}"
            )

        self._streaming = False
        self._rx_callback_function = _transfer_callback(self._rx_callback)
        self._rx_buffer = bytearray()
        self._target_samples = 0

        # Default moderate gains
        self.set_amp_enable(False)
        self.set_lna_gain(16)
        self.set_vga_gain(20)

    def close(self):
        if self._dev:
            if self._streaming:
                _libhackrf.hackrf_stop_rx(self._dev)
                self._streaming = False
            _libhackrf.hackrf_close(self._dev)
            self._dev = _p_hackrf_device(None)

    def __del__(self):
        self.close()

    def set_sample_rate(self, sr_hz: float):
        sr = ctypes.c_double(sr_hz)
        result = _libhackrf.hackrf_set_sample_rate(self._dev, sr)
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting sample rate to {sr_hz} Hz. Code: {result}")

    def set_baseband_filter_bandwidth(self, bw_hz: float):
        result = _libhackrf.hackrf_set_baseband_filter_bandwidth(
            self._dev, ctypes.c_uint32(int(bw_hz))
        )
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting baseband filter to {bw_hz} Hz. Code: {result}")

    def set_freq(self, freq_hz: float):
        result = _libhackrf.hackrf_set_freq(self._dev, ctypes.c_uint64(int(freq_hz)))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting frequency to {freq_hz} Hz. Code: {result}")

    def set_amp_enable(self, enable: bool):
        val = 1 if enable else 0
        result = _libhackrf.hackrf_set_amp_enable(self._dev, ctypes.c_uint8(val))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting amp_enable={enable}. Code: {result}")

    def set_lna_gain(self, gain: int):
        """
        HackRF internal steps are in multiples of 8 for LNA.
        This method will simply apply the requested gain,
        letting the firmware do the final quantization.
        """
        result = _libhackrf.hackrf_set_lna_gain(self._dev, ctypes.c_uint32(gain))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting LNA gain={gain}. Code: {result}")

    def set_vga_gain(self, gain: int):
        """
        HackRF internal steps for VGA are in multiples of 2.
        This method will let the firmware handle the final step size.
        """
        result = _libhackrf.hackrf_set_vga_gain(self._dev, ctypes.c_uint32(gain))
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error setting VGA gain={gain}. Code: {result}")

    def _rx_callback(self, transfer_ptr: ctypes.POINTER(_hackrf_transfer)) -> int:
        transfer = transfer_ptr.contents
        buf_length = transfer.valid_length
        if buf_length > 0:
            data_array = ctypes.cast(
                transfer.buffer, ctypes.POINTER(ctypes.c_byte * buf_length)
            ).contents
            self._rx_buffer.extend(data_array)

        if self._target_samples > 0 and len(self._rx_buffer) >= self._target_samples:
            self._streaming = False
            return 0
        return 0

    def read_samples(
        self, num_samples: int, duration: float = 0.0
    ) -> Optional[bytes]:
        """
        Acquire samples from the HackRF device.

        :param num_samples: Number of complex samples to read.
        :param duration: If > 0, stop after this many seconds even if
                         num_samples has not been reached.
        :return: Byte array of interleaved I/Q samples (uint8).
        """
        if self._streaming:
            raise RuntimeError("Already streaming. Stop first.")

        self._rx_buffer = bytearray()
        self._target_samples = num_samples * 2  # 2 bytes per complex sample (I/Q each 1 byte)

        result = _libhackrf.hackrf_start_rx(self._dev, self._rx_callback_function, None)
        if result != HACKRF_SUCCESS:
            raise RuntimeError(f"Error starting RX. Code: {result}")

        self._streaming = True

        start_time = time.time()
        while self._streaming:
            if duration > 0 and (time.time() - start_time) >= duration:
                break
            time.sleep(0.01)

        _libhackrf.hackrf_stop_rx(self._dev)
        self._streaming = False

        return self._rx_buffer[: self._target_samples]


# =========================================================================
# KNOWN_BANDS plus fallback GENERIC_BANDS
# =========================================================================
@dataclass
class FrequencyBand:
    name: str
    freq_start: float
    freq_end: float
    typical_bandwidth_hz: float
    metadata: Dict[str, Any]


@dataclass
class ActiveRange:
    start_hz: float
    end_hz: float
    peak_power_db: float
    likely_protocol: Optional[str] = None


class HackRFScanner:
    """
    A HackRF-based scanner that steps through a specified
    frequency range in increments of step_size, acquires samples,
    performs an FFT, and identifies sub-band peaks above threshold.
    Contiguous bins are grouped into ActiveRange objects and then
    matched to a known frequency band or a generic fallback range.
    """

    # Specific known allocations up to ~6 GHz
    KNOWN_BANDS: List[FrequencyBand] = [
        FrequencyBand(
            name="LF RFID",
            freq_start=125e3,
            freq_end=134.2e3,
            typical_bandwidth_hz=2e3,
            metadata={"comment": "Animal tags, key fobs"}
        ),
        FrequencyBand(
            name="Medium Wave AM Broadcast (Americas)",
            freq_start=520e3,
            freq_end=1710e3,
            typical_bandwidth_hz=10e3,
            metadata={"modulation": "AM", "comment": "AM radio Region 2"}
        ),
        FrequencyBand(
            name="Shortwave 31m Broadcast",
            freq_start=9.4e6,
            freq_end=9.9e6,
            typical_bandwidth_hz=5e3,
            metadata={"modulation": "AM/SSB", "comment": "Shortwave band"}
        ),
        FrequencyBand(
            name="FM Broadcast",
            freq_start=87.5e6,
            freq_end=108e6,
            typical_bandwidth_hz=200e3,
            metadata={"modulation": "WFM", "comment": "Commercial radio"}
        ),
        FrequencyBand(
            name="2m Amateur Radio",
            freq_start=144e6,
            freq_end=148e6,
            typical_bandwidth_hz=16e3,
            metadata={"modulation": "NFM/SSB", "comment": "Amateur VHF band"}
        ),
        FrequencyBand(
            name="Air Band (AM)",
            freq_start=118e6,
            freq_end=137e6,
            typical_bandwidth_hz=25e3,
            metadata={"modulation": "AM", "comment": "Aircraft communications"}
        ),
        FrequencyBand(
            name="NOAA Weather Radio",
            freq_start=162.4e6,
            freq_end=162.55e6,
            typical_bandwidth_hz=25e3,
            metadata={"modulation": "FM", "comment": "US weather broadcast"}
        ),
        FrequencyBand(
            name="70cm Amateur Radio",
            freq_start=430e6,
            freq_end=440e6,
            typical_bandwidth_hz=25e3,
            metadata={"modulation": "NFM/SSB", "comment": "Amateur UHF band"}
        ),
        FrequencyBand(
            name="Wi-Fi 2.4 GHz",
            freq_start=2400e6,
            freq_end=2483.5e6,
            typical_bandwidth_hz=20e6,
            metadata={"modulation": "OFDM", "comment": "802.11b/g/n"}
        ),
        FrequencyBand(
            name="Bluetooth 2.4 GHz",
            freq_start=2402e6,
            freq_end=2480e6,
            typical_bandwidth_hz=2e6,
            metadata={"modulation": "GFSK", "comment": "Bluetooth / BLE"}
        ),
        FrequencyBand(
            name="L-band Aero",
            freq_start=960e6,
            freq_end=1215e6,
            typical_bandwidth_hz=1e6,
            metadata={"comment": "Aero nav signals"}
        ),
        FrequencyBand(
            name="C-Band Radar",
            freq_start=5250e6,
            freq_end=5925e6,
            typical_bandwidth_hz=80e6,
            metadata={"modulation": "Pulse/Chirp", "comment": "Weather/military radar"}
        ),
        FrequencyBand(
            name="5 GHz Wi-Fi",
            freq_start=5150e6,
            freq_end=5850e6,
            typical_bandwidth_hz=20e6,
            metadata={"modulation": "OFDM", "comment": "802.11a/n/ac"}
        ),
    ]

    # Generic fallback ranges (up to ~8 GHz)
    GENERIC_BANDS: List[FrequencyBand] = [
        FrequencyBand(
            name="Generic LF (<300 kHz)",
            freq_start=0.0,
            freq_end=300e3,
            typical_bandwidth_hz=0,
            metadata={"comment": "Covers <300 kHz if not matched by known band"}
        ),
        FrequencyBand(
            name="Generic MF (300 kHz-3 MHz)",
            freq_start=300e3,
            freq_end=3e6,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic medium frequency range"}
        ),
        FrequencyBand(
            name="Generic HF (3-30 MHz)",
            freq_start=3e6,
            freq_end=30e6,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic high frequency range"}
        ),
        FrequencyBand(
            name="Generic VHF (30-300 MHz)",
            freq_start=30e6,
            freq_end=300e6,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic very high frequency range"}
        ),
        FrequencyBand(
            name="Generic UHF (300 MHz-1 GHz)",
            freq_start=300e6,
            freq_end=1e9,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic ultra high frequency range"}
        ),
        FrequencyBand(
            name="Generic L-Band (1-2 GHz)",
            freq_start=1e9,
            freq_end=2e9,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic L-band (1-2 GHz)"}
        ),
        FrequencyBand(
            name="Generic S-Band (2-4 GHz)",
            freq_start=2e9,
            freq_end=4e9,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic S-band (2-4 GHz)"}
        ),
        FrequencyBand(
            name="Generic C-Band (4-8 GHz)",
            freq_start=4e9,
            freq_end=8e9,
            typical_bandwidth_hz=0,
            metadata={"comment": "Generic C-band (4-8 GHz)"}
        ),
    ]

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
        device_index: int = 0
    ) -> None:
        self.start_hz: float = start_hz
        self.end_hz: float = end_hz
        self.step_size: float = step_size
        self.sample_rate: float = sample_rate
        self.bandwidth: float = bandwidth
        self.threshold_db: float = threshold_db
        self.fft_size: int = fft_size
        self.dwell_time: float = dwell_time
        self.merge_gap_khz: float = merge_gap_khz
        self.only_known_bands: bool = only_known_bands
        self.device_index: int = device_index

        self.active_ranges: List[ActiveRange] = []
        self._device: Optional[HackRFDevice] = None

    def open(self) -> None:
        self._device = HackRFDevice(device_index=self.device_index)
        self._device.set_sample_rate(self.sample_rate)
        self._device.set_baseband_filter_bandwidth(self.bandwidth)
        # Example moderate gains
        self._device.set_lna_gain(16)
        self._device.set_vga_gain(20)
        self._device.set_amp_enable(False)
        logging.info(
            f"Opened HackRF idx={self.device_index}, sample_rate={self.sample_rate}, "
            f"bandwidth={self.bandwidth}"
        )

    def close(self) -> None:
        if self._device:
            self._device.close()
            self._device = None
        logging.info("HackRF device closed.")

    def scan(self) -> None:
        if self._device is None:
            raise RuntimeError("HackRF device not open. Call .open() first.")

        # Construct the list of frequencies we will actually tune to
        if self.only_known_bands:
            # Restrict scanning to known bands overlapping [start_hz, end_hz]
            band_ranges = []
            for b in self.KNOWN_BANDS:
                if b.freq_end < self.start_hz or b.freq_start > self.end_hz:
                    continue
                bs = max(b.freq_start, self.start_hz)
                be = min(b.freq_end, self.end_hz)
                band_ranges.append((bs, be))

            # Merge overlapping or adjacent known-band intervals
            band_ranges = self._merge_band_ranges(band_ranges)

            logging.info(
                f"Scanning only known bands within {self.start_hz/1e6:.3f} - {self.end_hz/1e6:.3f} MHz. "
                f"Found {len(band_ranges)} band range(s) to scan."
            )

            freq_plan = []
            for (bs, be) in band_ranges:
                f = bs
                while f <= be:
                    freq_plan.append(f)
                    f += self.step_size
                # If the last step overshoots, ensure we include the band end
                if freq_plan and freq_plan[-1] < be:
                    freq_plan.append(be)
        else:
            # Step from start_hz to end_hz
            freq_plan = []
            f = self.start_hz
            while f <= self.end_hz:
                freq_plan.append(f)
                f += self.step_size
            if freq_plan and freq_plan[-1] < self.end_hz:
                freq_plan.append(self.end_hz)

            logging.info(
                f"Starting FFT-based scan from {self.start_hz/1e6:.2f} MHz to {self.end_hz/1e6:.2f} MHz "
                f"step={self.step_size/1e6:.2f} MHz, fft_size={self.fft_size}"
            )

        global_active_bins: List[Tuple[float, float, float]] = []

        # Tune to each center frequency in freq_plan, capture + FFT
        for cfreq in freq_plan:
            logging.debug(f"Tuning HackRF to {cfreq/1e6:.3f} MHz")

            self._device.set_freq(cfreq)

            # Collect enough samples for dwell_time or at least 2x fft_size
            samples_needed = max(self.fft_size * 2, int(self.sample_rate * self.dwell_time))
            raw_bytes = self._device.read_samples(
                num_samples=samples_needed,
                duration=self.dwell_time * 1.5
            )
            if not raw_bytes or len(raw_bytes) < self.fft_size * 2:
                logging.warning(f"Insufficient samples at {cfreq/1e6:.3f} MHz.")
                continue

            # Convert to complex64
            iq_data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int16) - 128
            iq_data = iq_data.reshape((-1, 2))
            complex_samples = iq_data[:, 0] + 1j * iq_data[:, 1]
            complex_samples = complex_samples.astype(np.complex64)

            # Window + FFT
            window = np.hanning(len(complex_samples))
            windowed_samples = complex_samples * window
            fft_result = np.fft.fftshift(np.fft.fft(windowed_samples, n=self.fft_size))
            power_spectrum = 20.0 * np.log10(np.abs(fft_result) + 1e-9)

            # Thresholding
            noise_floor_est = np.median(power_spectrum)
            threshold_value = noise_floor_est + self.threshold_db
            logging.debug(
                f"Center={cfreq/1e6:.3f} MHz, median_power={noise_floor_est:.2f} dB, "
                f"threshold={threshold_value:.2f} dB"
            )

            freq_axis = np.linspace(-0.5 * self.sample_rate, 0.5 * self.sample_rate, self.fft_size, endpoint=False)

            # Find contiguous bins above threshold
            above_thresh = (power_spectrum >= threshold_value)
            idx = 0
            local_active = []

            while idx < len(above_thresh):
                if above_thresh[idx]:
                    start_bin = idx
                    while idx < len(above_thresh) and above_thresh[idx]:
                        idx += 1
                    end_bin = idx - 1
                    peak_power = float(np.max(power_spectrum[start_bin:end_bin+1]))
                    start_freq_offset = freq_axis[start_bin]
                    end_freq_offset = freq_axis[end_bin]
                    abs_start = cfreq + start_freq_offset
                    abs_end = cfreq + end_freq_offset
                    if abs_start > abs_end:
                        abs_start, abs_end = abs_end, abs_start
                    local_active.append((abs_start, abs_end, peak_power))
                else:
                    idx += 1

            global_active_bins.extend(local_active)
            logging.debug(
                f"Found {len(local_active)} above-threshold sub-bands at center={cfreq/1e6:.3f} MHz."
            )

        # Merge across all collected bins
        merged = self._merge_active_bins(global_active_bins, self.merge_gap_khz)
        self.active_ranges = [
            ActiveRange(start_hz=m[0], end_hz=m[1], peak_power_db=m[2])
            for m in merged
        ]

        # Determine likely usage for each range
        for ar in self.active_ranges:
            ar.likely_protocol = self._match_protocol(ar.start_hz, ar.end_hz)

    @staticmethod
    def _merge_band_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not ranges:
            return []
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        current_start, current_end = sorted_ranges[0]

        for (bs, be) in sorted_ranges[1:]:
            if bs <= current_end:
                current_end = max(current_end, be)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = bs, be

        merged.append((current_start, current_end))
        return merged

    @staticmethod
    def _merge_active_bins(bins: List[Tuple[float, float, float]],
                           gap_khz: float) -> List[Tuple[float, float, float]]:
        """
        Merge adjacent or overlapping bins whose boundaries are within gap_khz.
        """
        if not bins:
            return []
        sorted_bins = sorted(bins, key=lambda x: x[0])
        merged: List[Tuple[float, float, float]] = []
        cur_start, cur_end, cur_power = sorted_bins[0]
        gap_hz = gap_khz * 1000

        for (s, e, p) in sorted_bins[1:]:
            if s <= (cur_end + gap_hz):
                # Merge
                cur_end = max(cur_end, e)
                cur_power = max(cur_power, p)
            else:
                merged.append((cur_start, cur_end, cur_power))
                cur_start, cur_end, cur_power = s, e, p

        merged.append((cur_start, cur_end, cur_power))
        return merged

    def _match_protocol(self, fstart: float, fend: float) -> str:
        center_freq = (fstart + fend) / 2.0
        best_match: Optional[str] = None
        best_score = 0.0

        # 1) Try to match specific known band
        for band in self.KNOWN_BANDS:
            if band.freq_start <= center_freq <= band.freq_end:
                overlap_frac = self._overlap_fraction(fstart, fend,
                                                      band.freq_start, band.freq_end)
                if overlap_frac > best_score:
                    best_score = overlap_frac
                    best_match = band.name

        # 2) If no strong specific match, try generic classification
        if not best_match:
            for gband in self.GENERIC_BANDS:
                if gband.freq_start <= center_freq <= gband.freq_end:
                    best_match = gband.name
                    break

        # 3) If it does not fall in any known or generic range => Unknown
        return best_match if best_match else "Unknown/Other"

    @staticmethod
    def _overlap_fraction(st1: float, en1: float,
                          st2: float, en2: float) -> float:
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A HackRF-based script to scan for active radio signals "
            "via an FFT-based approach, grouping contiguous active "
            "bins and identifying them via known or generic ranges."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-s', '--start-hz',
        type=float,
        default=2e6,
        help='Start frequency in Hz.'
    )
    parser.add_argument(
        '-e', '--end-hz',
        type=float,
        default=6e9,
        help='End frequency in Hz.'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=5.0,
        help='Power threshold in dB above noise floor.'
    )
    parser.add_argument(
        '-r', '--sample-rate',
        type=float,
        default=2e6,
        help='Sample rate in Hz.'
    )
    parser.add_argument(
        '-b', '--bandwidth',
        type=float,
        default=2e6,
        help='Baseband filter bandwidth in Hz.'
    )
    parser.add_argument(
        '-S', '--step-size',
        type=float,
        default=2e6,
        help='Frequency step size in Hz.'
    )
    parser.add_argument(
        '-n', '--fft-size',
        type=int,
        default=1024,
        help='Number of FFT bins (default: 1024).'
    )
    parser.add_argument(
        '-w', '--dwell-time',
        type=float,
        default=0.05,
        help='Dwell time in seconds for each step.'
    )
    parser.add_argument(
        '-m', '--merge-gap-khz',
        type=float,
        default=100.0,
        help='Merge gap in kHz for adjacent active bins.'
    )
    parser.add_argument(
        '-k', '--only-known-bands',
        action='store_true',
        help='If set, only scan known frequency bands from the database.'
    )
    parser.add_argument(
        '-p', '--preset',
        type=str,
        choices=['fast', 'reasonable', 'slow'],
        default=None,
        help='Preset scan mode that overrides some defaults.'
    )
    parser.add_argument(
        '-d', '--device-index',
        type=int,
        default=0,
        help='HackRF device index.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity. -v => INFO, -vv => DEBUG.'
    )

    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    """
    Adjust arguments based on chosen preset,
    without forcing only_known_bands unless user explicitly sets it.
    """
    if args.preset == 'fast':
        args.fft_size = 512
        args.dwell_time = 0.02
    elif args.preset == 'reasonable':
        args.fft_size = 1024
        args.dwell_time = 0.05
    elif args.preset == 'slow':
        args.fft_size = 4096
        args.dwell_time = 0.1


def setup_logging(verbosity: int) -> None:
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> None:
    args = parse_arguments()
    setup_logging(args.verbose)

    apply_preset(args)

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
        device_index=args.device_index
    )

    try:
        scanner.open()
        scanner.scan()
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt received. Stopping.")
    except Exception as e:
        logging.error(f"Error during scanning: {e}")
    finally:
        scanner.close()

    if not scanner.active_ranges:
        print("No active signals detected above the threshold.")
        return

    # Output
    if HAS_PRETTYTABLE:
        table = PrettyTable()
        table.field_names = ["#", "Start (MHz)", "End (MHz)", "Peak Power (dB)", "Protocol"]
        for i, ar in enumerate(scanner.active_ranges, start=1):
            table.add_row([
                i,
                f"{ar.start_hz/1e6:.3f}",
                f"{ar.end_hz/1e6:.3f}",
                f"{ar.peak_power_db:.2f}",
                ar.likely_protocol
            ])
        print(table)
    else:
        print("Detected active frequency ranges:")
        for idx, ar in enumerate(scanner.active_ranges, start=1):
            print(
                f"[{idx}] {ar.start_hz/1e6:.3f} MHz - {ar.end_hz/1e6:.3f} MHz | "
                f"Peak Power: {ar.peak_power_db:.2f} dB | "
                f"Protocol: {ar.likely_protocol}"
            )


if __name__ == "__main__":
    main()
