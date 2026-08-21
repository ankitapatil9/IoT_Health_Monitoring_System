"""
HealthSense - ECG Preprocessing Module
Modular, production-ready preprocessing functions for ECG signal filtering,
beat segmentation, normalization, and AAMI EC57 annotation mapping.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# Sampling rate specification
DEFAULT_FS = 360.0  # MIT-BIH sampling rate (Hz)
BEAT_WINDOW_SIZE = 256  # 256 samples (~711 ms at 360 Hz)
PRE_PEAK_SAMPLES = 100   # Samples before R-peak
POST_PEAK_SAMPLES = 156  # Samples after R-peak (PRE + POST = 256)

# -------------------------------------------------------------------------
# AAMI EC57 Annotation Class Mapping
# -------------------------------------------------------------------------
# Standard 5-class AAMI mapping:
# N: Normal beat and bundle branch block beats
# S: Supraventricular ectopic beat
# V: Ventricular ectopic beat
# F: Fusion beat
# Q: Unknown / Paced / Unclassifiable beat
AAMI_5CLASS_MAPPING = {
    # N - Normal
    'N': 'N',  # Normal beat
    'L': 'N',  # Left bundle branch block beat
    'R': 'N',  # Right bundle branch block beat
    'e': 'N',  # Atrial escape beat
    'j': 'N',  # Nodal (junctional) escape beat
    'B': 'N',  # Bundle branch block unspecified
    
    # S - Supraventricular Ectopic
    'A': 'S',  # Atrial premature beat
    'a': 'S',  # Aberrated atrial premature beat
    'J': 'S',  # Nodal (junctional) premature beat
    'S': 'S',  # Supraventricular premature beat
    
    # V - Ventricular Ectopic
    'V': 'V',  # Premature ventricular contraction
    'E': 'V',  # Ventricular escape beat
    
    # F - Fusion
    'F': 'F',  # Fusion of ventricular and normal beat
    
    # Q - Unknown / Paced
    '/': 'Q',  # Paced beat
    'f': 'Q',  # Fusion of paced and normal beat
    'Q': 'Q',  # Unclassifiable beat
    '?': 'Q',  # Beat not classified by annotator
}

# Integer class indices
CLASS_TO_INT_5CLASS = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
INT_TO_CLASS_5CLASS = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

# Binary classification mapping: NORMAL vs ABNORMAL
# NORMAL = N (0)
# ABNORMAL = S, V, F, Q (1)
CLASS_TO_INT_BINARY = {'NORMAL': 0, 'ABNORMAL': 1}
INT_TO_CLASS_BINARY = {0: 'NORMAL', 1: 'ABNORMAL'}

# Excluded non-beat annotations (rhythm markers, signal quality markers, comments)
EXCLUDED_ANNOTATIONS = {
    '+': 'Rhythm change',
    '~': 'Signal quality change / artifact',
    '!': 'Ventricular flutter wave',
    '"': 'Comment annotation',
    'x': 'Non-conducted P-wave (blocked APC)',
    '|': 'Isolated QRS-like artifact',
    '[': 'Start of ventricular flutter/fibrillation',
    ']': 'End of ventricular flutter/fibrillation',
}


def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = DEFAULT_FS, order: int = 3) -> np.ndarray:
    """
    Applies zero-phase 3rd order Butterworth bandpass filter to suppress
    baseline wander (<0.5 Hz) and high-frequency noise / EMG artifacts (>45 Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered


def apply_notch_filter(data: np.ndarray, notch_freq: float = 50.0, fs: float = DEFAULT_FS, q: float = 30.0) -> np.ndarray:
    """
    Applies IIR notch filter to remove powerline interference (50 Hz or 60 Hz).
    """
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, q)
    filtered = filtfilt(b, a, data)
    return filtered


def z_score_normalize(signal_segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standard Z-score normalization per segment: (x - mean) / (std + eps).
    """
    mean = np.mean(signal_segment)
    std = np.std(signal_segment)
    if std < eps:
        return signal_segment - mean
    return (signal_segment - mean) / (std + eps)


def preprocess_ecg_signal(raw_signal: np.ndarray, fs: float = DEFAULT_FS, apply_notch: bool = False, notch_freq: float = 50.0) -> np.ndarray:
    """
    Full continuous signal preprocessing pipeline:
    1. Bandpass filter (0.5 Hz - 45.0 Hz) to eliminate baseline wander and high-freq noise.
    2. Optional powerline notch filter.
    """
    filtered = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=45.0, fs=fs, order=3)
    if apply_notch:
        filtered = apply_notch_filter(filtered, notch_freq=notch_freq, fs=fs)
    return filtered


def extract_beat_window(signal: np.ndarray, r_peak_idx: int, pre_peak: int = PRE_PEAK_SAMPLES, post_peak: int = POST_PEAK_SAMPLES) -> np.ndarray:
    """
    Extracts a fixed-length window around the R-peak: [r_peak - pre_peak, r_peak + post_peak].
    Total window length = pre_peak + post_peak (default 256 samples).
    Returns None if boundary constraints are violated.
    """
    start_idx = r_peak_idx - pre_peak
    end_idx = r_peak_idx + post_peak
    
    if start_idx < 0 or end_idx > len(signal):
        return None
    
    segment = signal[start_idx:end_idx]
    if len(segment) != (pre_peak + post_peak):
        return None
        
    return segment


def preprocess_single_beat(raw_beat_segment: np.ndarray) -> np.ndarray:
    """
    Processes an individual extracted beat for real-time model inference.
    Performs Z-score normalization.
    """
    return z_score_normalize(raw_beat_segment)


def map_annotation_symbol(symbol: str):
    """
    Maps MIT-BIH annotation symbol to:
    - aami_5class: 'N', 'S', 'V', 'F', 'Q' (or None if excluded)
    - binary_class: 'NORMAL', 'ABNORMAL' (or None if excluded)
    - is_beat: True if valid beat, False if excluded non-beat marker
    """
    if symbol in AAMI_5CLASS_MAPPING:
        aami_class = AAMI_5CLASS_MAPPING[symbol]
        binary_class = 'NORMAL' if aami_class == 'N' else 'ABNORMAL'
        return {
            "aami_5class": aami_class,
            "aami_5class_int": CLASS_TO_INT_5CLASS[aami_class],
            "binary_class": binary_class,
            "binary_class_int": CLASS_TO_INT_BINARY[binary_class],
            "is_beat": True,
            "original_symbol": symbol
        }
    else:
        return {
            "aami_5class": None,
            "aami_5class_int": -1,
            "binary_class": None,
            "binary_class_int": -1,
            "is_beat": False,
            "original_symbol": symbol,
            "exclusion_reason": EXCLUDED_ANNOTATIONS.get(symbol, "Non-standard/auxiliary annotation")
        }
