import mne
import numpy as np
from scipy import signal
import time
from typing import Dict, Tuple, Optional
from eeg_processor import EEGProcessor


class RealtimeEEGProcessor:
    """Handles real-time EEG data processing by splitting data into chunks"""

    def __init__(self):
        self.eeg_processor = EEGProcessor()
        self.raw_data = None
        self.current_position = 0
        self.chunk_duration = 3  # seconds
        self.window_size = 10  # seconds for analysis
        self.sfreq = None
        self.total_samples = 0

    def initialize(self, file_path: str, chunk_duration: float, window_size: float,
                   low_freq: float, high_freq: float):
        """
        Initialize the real-time processor with EEG data

        Args:
            file_path: Path to .bdf file
            chunk_duration: Duration of each chunk in seconds
            window_size: Size of analysis window in seconds
            low_freq: High-pass filter frequency
            high_freq: Low-pass filter frequency
        """

        # Load and preprocess the full EEG file
        print(f"Loading EEG file: {file_path}")
        raw = self.eeg_processor.load_bdf_file(file_path)

        print("Preprocessing EEG data...")
        self.raw_data = self.eeg_processor.preprocess_eeg(raw, low_freq, high_freq)

        # Store parameters
        self.chunk_duration = chunk_duration
        self.window_size = window_size
        self.sfreq = self.raw_data.info['sfreq']
        self.total_samples = len(self.raw_data.times)
        self.current_position = 0

        print(f"Initialized real-time processor:")
        print(f"  - Sampling rate: {self.sfreq} Hz")
        print(f"  - Total duration: {self.raw_data.times[-1]:.1f} seconds")
        print(f"  - Chunk duration: {chunk_duration} seconds")
        print(f"  - Analysis window: {window_size} seconds")

    def get_next_chunk(self) -> Optional[Tuple[Dict[str, float], str, str, mne.io.Raw]]:
        """
        Get the next chunk of EEG data for real-time processing

        Returns:
            Tuple of (band_powers, dominant_band, attention_state, raw_chunk) or None if end reached
        """

        if self.raw_data is None:
            raise ValueError("Real-time processor not initialized")

        # Calculate sample indices
        chunk_samples = int(self.chunk_duration * self.sfreq)
        window_samples = int(self.window_size * self.sfreq)

        # Check if we've reached the end
        if self.current_position >= self.total_samples - window_samples:
            return None

        # Extract analysis window (larger than chunk for better frequency resolution)
        start_idx = max(0, self.current_position - window_samples + chunk_samples)
        end_idx = min(self.current_position + chunk_samples, self.total_samples)

        # Create a chunk of raw data for analysis
        raw_chunk = self.raw_data.copy().crop(
            tmin=start_idx / self.sfreq,
            tmax=end_idx / self.sfreq
        )

        # Extract frequency bands from the chunk
        band_powers, dominant_band = self.eeg_processor.extract_frequency_bands(raw_chunk)

        # Map to attention state
        attention_state = self.eeg_processor.map_to_attention_state(dominant_band)

        # Move to next position
        self.current_position += chunk_samples

        # Create a smaller chunk for visualization (just the current chunk)
        viz_start = max(0, end_idx - chunk_samples)
        viz_chunk = self.raw_data.copy().crop(
            tmin=viz_start / self.sfreq,
            tmax=end_idx / self.sfreq
        )

        return band_powers, dominant_band, attention_state, viz_chunk

    def get_progress(self) -> float:
        """Get current progress through the data (0-1)"""
        if self.total_samples == 0:
            return 0.0
        return min(1.0, self.current_position / self.total_samples)

    def reset(self):
        """Reset to beginning of data"""
        self.current_position = 0

    def seek_to_time(self, time_seconds: float):
        """Seek to specific time in the data"""
        if self.sfreq is not None:
            self.current_position = int(time_seconds * self.sfreq)
            self.current_position = max(0, min(self.current_position, self.total_samples))

    def get_current_time(self) -> float:
        """Get current time position in seconds"""
        if self.sfreq is None:
            return 0.0
        return self.current_position / self.sfreq

    def get_total_duration(self) -> float:
        """Get total duration of the data in seconds"""
        if self.raw_data is None:
            return 0.0
        return self.raw_data.times[-1]
