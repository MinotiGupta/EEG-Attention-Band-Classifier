import mne
import numpy as np
from scipy import signal
from typing import Dict, Tuple
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level('ERROR')


class EEGProcessor:
    """Handles EEG data processing and analysis"""

    def __init__(self):
        # Define frequency bands
        self.frequency_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

        # Attention state mapping
        self.attention_mapping = {
            'Delta': 'Sleepy',
            'Theta': 'Relaxed',
            'Alpha': 'Calm',
            'Beta': 'Focused',
            'Gamma': 'Highly engaged'
        }

    def load_bdf_file(self, file_path: str) -> mne.io.Raw:
        """
        Load a .bdf file using MNE

        Args:
            file_path: Path to .bdf file

        Returns:
            MNE Raw object
        """
        try:
            print(f"Loading .bdf file: {file_path}")
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
            print(f"Successfully loaded: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
            return raw
        except Exception as e:
            raise Exception(f"Failed to load .bdf file: {str(e)}")

    def preprocess_eeg(self, raw: mne.io.Raw, low_freq: float = 1.0, high_freq: float = 50.0) -> mne.io.Raw:
        """
        Preprocess EEG data with filtering and referencing

        Args:
            raw: MNE Raw object
            low_freq: High-pass filter frequency
            high_freq: Low-pass filter frequency

        Returns:
            Preprocessed MNE Raw object
        """
        print("Preprocessing EEG data...")

        # Create a copy to avoid modifying original
        raw_processed = raw.copy()

        # Pick only EEG channels (exclude non-EEG channels)
        raw_processed.pick_types(eeg=True, exclude='bads')
        print(f"Selected {len(raw_processed.ch_names)} EEG channels")

        # Apply average reference
        raw_processed.set_eeg_reference('average', projection=True)
        raw_processed.apply_proj()
        print("Applied average reference")

        # Apply band-pass filter
        raw_processed.filter(low_freq, high_freq, fir_design='firwin', verbose=False)
        print(f"Applied band-pass filter: {low_freq}-{high_freq} Hz")

        return raw_processed

    def extract_frequency_bands(self, raw: mne.io.Raw) -> Tuple[Dict[str, float], str]:
        """
        Extract power in different frequency bands using Welch's method

        Args:
            raw: Preprocessed MNE Raw object

        Returns:
            Tuple of (band_powers dict, dominant_band string)
        """
        # Get data
        data = raw.get_data()
        sfreq = raw.info['sfreq']

        # Compute power spectral density using Welch's method
        # Use longer segments for better frequency resolution
        nperseg = min(int(sfreq * 4), data.shape[1])  # 4 second segments or full length
        freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=1)

        # Average across channels
        psd_mean = np.mean(psd, axis=0)

        # Extract power in each frequency band
        band_powers = {}

        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

            if np.any(freq_mask):
                # Calculate average power in the band
                band_power = np.mean(psd_mean[freq_mask])
                band_powers[band_name] = band_power
            else:
                band_powers[band_name] = 0.0

        # Find dominant band
        if band_powers:
            dominant_band = max(band_powers, key=band_powers.get)
        else:
            dominant_band = 'Alpha'  # Default

        return band_powers, dominant_band

    def map_to_attention_state(self, dominant_band: str) -> str:
        """
        Map dominant frequency band to attention state

        Args:
            dominant_band: Name of dominant frequency band

        Returns:
            Attention state string
        """
        return self.attention_mapping.get(dominant_band, 'Unknown')

    def compute_band_ratios(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """
        Compute useful band ratios for additional analysis

        Args:
            band_powers: Dictionary of band powers

        Returns:
            Dictionary of band ratios
        """
        ratios = {}

        # Common ratios used in EEG analysis
        if 'Alpha' in band_powers and 'Beta' in band_powers and band_powers['Beta'] > 0:
            ratios['Alpha/Beta'] = band_powers['Alpha'] / band_powers['Beta']

        if 'Theta' in band_powers and 'Beta' in band_powers and band_powers['Beta'] > 0:
            ratios['Theta/Beta'] = band_powers['Theta'] / band_powers['Beta']

        if 'Delta' in band_powers and 'Alpha' in band_powers and band_powers['Alpha'] > 0:
            ratios['Delta/Alpha'] = band_powers['Delta'] / band_powers['Alpha']

        return ratios
