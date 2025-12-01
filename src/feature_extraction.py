"""
Feature Extraction Module for Vibration Signal Analysis

This module provides comprehensive feature extraction for pump fault detection:
- Time-domain features (statistical)
- Frequency-domain features (spectral)
- Time-frequency features (wavelets, envelope analysis)
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt
import pywt


class VibrationFeatureExtractor:
    """Extract features from vibration signals for fault detection."""

    def __init__(self, fs=10240):
        """
        Initialize feature extractor.

        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz (default: 10240 Hz)
        """
        self.fs = fs

    def extract_all_features(self, signal_data, axis_name='X'):
        """
        Extract all features from a signal window.

        Parameters:
        -----------
        signal_data : np.array
            1D array of vibration signal
        axis_name : str
            Name of the axis (e.g., 'X', 'Y', 'Z')

        Returns:
        --------
        dict
            Dictionary of feature_name: value
        """
        features = {}

        # Time-domain features
        time_features = self.extract_time_features(signal_data)
        features.update({f"{axis_name}_{k}": v for k, v in time_features.items()})

        # Frequency-domain features
        freq_features = self.extract_frequency_features(signal_data)
        features.update({f"{axis_name}_{k}": v for k, v in freq_features.items()})

        # Wavelet features
        wavelet_features = self.extract_wavelet_features(signal_data)
        features.update({f"{axis_name}_{k}": v for k, v in wavelet_features.items()})

        # Envelope features
        envelope_features = self.extract_envelope_features(signal_data)
        features.update({f"{axis_name}_{k}": v for k, v in envelope_features.items()})

        return features

    def extract_time_features(self, signal_data):
        """
        Extract time-domain statistical features.

        Returns 8 features:
        - RMS, Peak, Crest Factor, Kurtosis, Skewness
        - Shape Factor, Impulse Factor, Clearance Factor
        """
        features = {}

        # Basic statistics
        rms = np.sqrt(np.mean(signal_data**2))
        peak = np.max(np.abs(signal_data))
        mean_abs = np.mean(np.abs(signal_data))

        # Statistical features
        features['rms'] = rms
        features['peak'] = peak
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['kurtosis'] = stats.kurtosis(signal_data)
        features['skewness'] = stats.skew(signal_data)

        # Shape factors (avoid division by zero)
        if rms > 1e-10:
            features['crest_factor'] = peak / rms
            features['shape_factor'] = rms / (mean_abs + 1e-10)
        else:
            features['crest_factor'] = 0
            features['shape_factor'] = 0

        if mean_abs > 1e-10:
            features['impulse_factor'] = peak / mean_abs
        else:
            features['impulse_factor'] = 0

        # Clearance factor
        sqrt_mean = np.mean(np.sqrt(np.abs(signal_data)))**2
        if sqrt_mean > 1e-10:
            features['clearance_factor'] = peak / sqrt_mean
        else:
            features['clearance_factor'] = 0

        return features

    def extract_frequency_features(self, signal_data):
        """
        Extract frequency-domain features using FFT.

        Returns:
        - Spectral statistics (peak freq, peak amplitude, centroid, spread, entropy)
        - Band power features (4 frequency bands)
        """
        features = {}

        # Compute FFT
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1/self.fs)[:N//2]
        power_spectrum = 2.0/N * np.abs(yf[:N//2])

        # Normalize power spectrum for entropy calculation
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)

        # Peak frequency and amplitude
        peak_idx = np.argmax(power_spectrum)
        features['peak_frequency'] = xf[peak_idx]
        features['peak_amplitude'] = power_spectrum[peak_idx]

        # Spectral centroid (center of mass)
        features['spectral_centroid'] = np.sum(xf * power_norm)

        # Spectral spread (standard deviation around centroid)
        features['spectral_spread'] = np.sqrt(
            np.sum(((xf - features['spectral_centroid'])**2) * power_norm)
        )

        # Spectral entropy
        power_norm_nonzero = power_norm[power_norm > 1e-10]
        features['spectral_entropy'] = -np.sum(
            power_norm_nonzero * np.log(power_norm_nonzero + 1e-10)
        )

        # Band power features
        # Define frequency bands
        bands = {
            'low': (1, 10),        # Misalignment, imbalance
            'mid': (10, 1000),     # Bearing outer race
            'high': (1000, 5000),  # Bearing inner race
            'vhigh': (5000, self.fs/2)  # Early bearing damage
        }

        for band_name, (f_low, f_high) in bands.items():
            mask = (xf >= f_low) & (xf <= f_high)
            features[f'band_power_{band_name}'] = np.sum(power_spectrum[mask]**2)

        # Band power ratios
        total_power = np.sum(power_spectrum**2) + 1e-10
        features['band_ratio_high_low'] = (
            features['band_power_high'] / (features['band_power_low'] + 1e-10)
        )
        features['band_ratio_mid_total'] = features['band_power_mid'] / total_power

        return features

    def extract_wavelet_features(self, signal_data, wavelet='db4', level=5):
        """
        Extract wavelet-based features using Discrete Wavelet Transform.

        Parameters:
        -----------
        wavelet : str
            Wavelet type (default: 'db4' - Daubechies 4)
        level : int
            Decomposition level (default: 5)

        Returns:
        --------
        dict
            Energy and entropy of detail coefficients at each level
        """
        features = {}

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        # Extract features from detail coefficients
        for i, detail_coeffs in enumerate(coeffs[1:], 1):  # Skip approximation
            # Energy in detail coefficients
            energy = np.sum(detail_coeffs**2)
            features[f'wavelet_energy_d{i}'] = energy

            # Entropy of detail coefficients
            detail_norm = np.abs(detail_coeffs) / (np.sum(np.abs(detail_coeffs)) + 1e-10)
            detail_norm = detail_norm[detail_norm > 1e-10]
            entropy = -np.sum(detail_norm * np.log(detail_norm + 1e-10))
            features[f'wavelet_entropy_d{i}'] = entropy

        return features

    def extract_envelope_features(self, signal_data, band=(2000, 8000)):
        """
        Extract envelope analysis features for bearing fault detection.

        Parameters:
        -----------
        band : tuple
            Band-pass filter range (f_low, f_high) in Hz
            Default: (2000, 8000) - typical bearing resonance range

        Returns:
        --------
        dict
            Features from envelope spectrum
        """
        features = {}

        # Band-pass filter
        nyq = 0.5 * self.fs
        low = band[0] / nyq
        high = band[1] / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal_data)

        # Hilbert transform to get envelope
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)

        # FFT of envelope
        N = len(envelope)
        yf = fft(envelope)
        xf = fftfreq(N, 1/self.fs)[:N//2]
        envelope_spectrum = 2.0/N * np.abs(yf[:N//2])

        # Extract features from envelope spectrum
        features['envelope_rms'] = np.sqrt(np.mean(envelope**2))
        features['envelope_peak'] = np.max(envelope)
        features['envelope_kurtosis'] = stats.kurtosis(envelope)

        # Peak in envelope spectrum
        peak_idx = np.argmax(envelope_spectrum)
        features['envelope_peak_freq'] = xf[peak_idx]
        features['envelope_peak_amplitude'] = envelope_spectrum[peak_idx]

        return features


def preprocess_signal(signal_data, fs=10240, highpass_cutoff=1.0):
    """
    Preprocess raw vibration signal.

    Steps:
    1. Detrend (remove DC offset and linear trend)
    2. High-pass filter (remove low-frequency motion artifacts)
    3. Outlier removal (clip extreme values)

    Parameters:
    -----------
    signal_data : np.array
        Raw vibration signal
    fs : float
        Sampling frequency in Hz
    highpass_cutoff : float
        High-pass filter cutoff frequency in Hz

    Returns:
    --------
    np.array
        Preprocessed signal
    """
    # Detrend
    signal_detrended = signal.detrend(signal_data)

    # High-pass filter
    nyq = 0.5 * fs
    low = highpass_cutoff / nyq
    b, a = butter(4, low, btype='high')
    signal_filtered = filtfilt(b, a, signal_detrended)

    # Outlier removal (clip at 5 sigma)
    sigma = np.std(signal_filtered)
    mean = np.mean(signal_filtered)
    signal_clipped = np.clip(signal_filtered, mean - 5*sigma, mean + 5*sigma)

    return signal_clipped


def segment_signal(signal_data, window_size, overlap=0.5, fs=10240):
    """
    Segment long signal into windows.

    Parameters:
    -----------
    signal_data : np.array
        Long vibration signal
    window_size : float
        Window size in seconds
    overlap : float
        Overlap fraction (0-1)
    fs : float
        Sampling frequency in Hz

    Returns:
    --------
    list of np.array
        List of signal windows
    """
    window_samples = int(window_size * fs)
    hop_samples = int(window_samples * (1 - overlap))

    windows = []
    start = 0

    while start + window_samples <= len(signal_data):
        window = signal_data[start:start + window_samples]
        windows.append(window)
        start += hop_samples

    return windows


def extract_features_from_multiaxis(signals_xyz, fs=10240, axis_names=['X', 'Y', 'Z']):
    """
    Extract features from multi-axis vibration data.

    Parameters:
    -----------
    signals_xyz : list of np.array
        List of signals for each axis [signal_x, signal_y, signal_z]
    fs : float
        Sampling frequency
    axis_names : list of str
        Names for each axis

    Returns:
    --------
    dict
        Combined feature dictionary with all axes
    """
    extractor = VibrationFeatureExtractor(fs=fs)

    all_features = {}

    # Extract features for each axis
    for signal_data, axis_name in zip(signals_xyz, axis_names):
        features = extractor.extract_all_features(signal_data, axis_name=axis_name)
        all_features.update(features)

    # Cross-axis features
    # Correlation between axes
    all_features['corr_XY'] = np.corrcoef(signals_xyz[0], signals_xyz[1])[0, 1]
    all_features['corr_XZ'] = np.corrcoef(signals_xyz[0], signals_xyz[2])[0, 1]
    all_features['corr_YZ'] = np.corrcoef(signals_xyz[1], signals_xyz[2])[0, 1]

    # Total vibration magnitude
    magnitude = np.sqrt(signals_xyz[0]**2 + signals_xyz[1]**2 + signals_xyz[2]**2)
    all_features['magnitude_rms'] = np.sqrt(np.mean(magnitude**2))
    all_features['magnitude_peak'] = np.max(magnitude)

    return all_features


if __name__ == "__main__":
    # Example usage
    print("Vibration Feature Extraction Module")
    print("=" * 50)

    # Generate synthetic signal (demo)
    fs = 10240  # Hz
    duration = 2.56  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Synthetic signal: healthy pump (low noise) + small bearing fault
    signal_healthy = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50 Hz rotation
    signal_fault = 0.05 * np.sin(2 * np.pi * 1234 * t)  # Bearing fault frequency
    noise = 0.02 * np.random.randn(len(t))
    signal_x = signal_healthy + signal_fault + noise

    # Preprocess
    signal_preprocessed = preprocess_signal(signal_x, fs=fs)

    # Extract features
    extractor = VibrationFeatureExtractor(fs=fs)
    features = extractor.extract_all_features(signal_preprocessed, axis_name='X')

    print(f"\nExtracted {len(features)} features:")
    for name, value in list(features.items())[:10]:
        print(f"  {name:30s}: {value:.6f}")
    print("  ...")

    print("\nFeature extraction successful!")
