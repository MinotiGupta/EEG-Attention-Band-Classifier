"""
Generate sample EEG data for testing the application
This creates synthetic .bdf files for demonstration purposes
"""

import mne
import numpy as np
from pathlib import Path
import os

def create_sample_bdf_data(output_dir: str = "./sample_data"):
    """
    Create sample .bdf files for testing

    Args:
        output_dir: Directory to save sample files
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # EEG parameters
    n_channels = 32
    sfreq = 256  # Sampling frequency
    duration = 60  # Duration in seconds

    # Create channel names (standard 10-20 system)
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ]

    # Create different attention states
    scenarios = {
        'focused_session': {
            'dominant_band': 'Beta',
            'description': 'High beta activity - focused state'
        },
        'relaxed_session': {
            'dominant_band': 'Alpha',
            'description': 'High alpha activity - calm state'
        },
        'sleepy_session': {
            'dominant_band': 'Delta',
            'description': 'High delta activity - sleepy state'
        }
    }

    for scenario_name, scenario_info in scenarios.items():
        print(f"Creating {scenario_name}...")

        # Create subject directory
        subject_dir = output_path / f"subject_01"
        subject_dir.mkdir(exist_ok=True)

        # Generate synthetic EEG data
        data = generate_synthetic_eeg(
            n_channels, sfreq, duration,
            dominant_band=scenario_info['dominant_band']
        )

        # Create MNE info structure
        info = mne.create_info(
            ch_names=ch_names[:n_channels],
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Create Raw object
        raw = mne.io.RawArray(data, info)

        # Save as .bdf file
        output_file = subject_dir / f"{scenario_name}.bdf"
        raw.export(str(output_file), fmt='BDF', overwrite=True)

        print(f"  Saved: {output_file}")
        print(f"  Description: {scenario_info['description']}")

    print(f"\nSample data created in: {output_path}")
    print("You can now use this data to test the EEG analyzer!")

def generate_synthetic_eeg(n_channels: int, sfreq: float, duration: float,
                          dominant_band: str = 'Alpha') -> np.ndarray:
    """
    Generate synthetic EEG data with specified dominant frequency band

    Args:
        n_channels: Number of EEG channels
        sfreq: Sampling frequency
        duration: Duration in seconds
        dominant_band: Which frequency band should be dominant

    Returns:
        Synthetic EEG data array
    """

    n_samples = int(sfreq * duration)
    time = np.linspace(0, duration, n_samples)

    # Frequency bands and their typical frequencies
    band_freqs = {
        'Delta': [1, 2, 3],
        'Theta': [5, 6, 7],
        'Alpha': [9, 10, 11],
        'Beta': [15, 20, 25],
        'Gamma': [35, 40, 45]
    }

    data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Add noise
        noise = np.random.normal(0, 5, n_samples)

        # Add frequency components
        signal = noise.copy()

        for band, freqs in band_freqs.items():
            for freq in freqs:
                # Amplitude depends on whether this is the dominant band
                if band == dominant_band:
                    amplitude = np.random.uniform(15, 25)  # Higher amplitude
                else:
                    amplitude = np.random.uniform(3, 8)   # Lower amplitude

                # Add sinusoidal component
                phase = np.random.uniform(0, 2*np.pi)
                signal += amplitude * np.sin(2 * np.pi * freq * time + phase)

        # Add some channel-specific variation
        channel_factor = 0.8 + 0.4 * np.random.random()
        data[ch, :] = signal * channel_factor

    return data

if __name__ == "__main__":
    create_sample_bdf_data()
