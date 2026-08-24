"""
HealthSense - ECG Stream Processor & Sliding-Window Inference Manager
Manages real-time FIFO sample buffers, beat window segmentation, digital filtering,
and thread-safe inference invocation against the optimized ML arrhythmia model.
"""

import os
import sys
import collections
import threading
import time
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.signal import find_peaks

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.predict_ecg_optimized import get_predictor, EXPECTED_WINDOW_SIZE, DEFAULT_THRESHOLD
from ml.preprocessing import butter_bandpass_filter


class ECGStreamProcessor:
    """
    Continuous streaming processor for raw single-lead ECG telemetry.
    Maintains ring buffers, applies real-time filtering, detects R-peaks,
    and runs ML arrhythmia inference on extracted 256-sample windows
    using the verified optimized model and 0.30 threshold.
    """

    def __init__(
        self,
        buffer_size: int = 1500,
        sampling_rate: int = 250,
        model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD
    ):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

        # Ring buffers for waveform telemetry
        self.raw_buffer = collections.deque(maxlen=buffer_size)
        self.filtered_buffer = collections.deque(maxlen=buffer_size)
        self.timestamps = collections.deque(maxlen=buffer_size)

        # Verified ML Predictor (0.30 threshold)
        self.predictor = get_predictor(model_path=model_path, threshold=threshold)

        # State tracking
        self.last_inferred_peak_idx = -1
        self.total_samples_ingested = 0
        self.last_beat_prediction: Dict[str, Any] = {
            "prediction": "NORMAL",
            "class_id": 0,
            "abnormal_probability": 0.05,
            "confidence": 0.95,
            "is_abnormal": False,
            "threshold": self.predictor.threshold,
            "status": "SUCCESS",
            "timestamp": time.time()
        }
        self.recent_predictions = collections.deque(maxlen=20)
        self.total_beats_evaluated = 0
        self.total_abnormal_beats = 0

    def ingest_samples(self, samples: List[float], timestamp: Optional[float] = None) -> None:
        """
        Ingests a batch or single ECG voltage sample into the sliding buffer.
        """
        if not samples:
            return

        ts_now = timestamp if timestamp is not None else time.time()
        n_samples = len(samples)
        step = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.004

        with self.lock:
            for i, val in enumerate(samples):
                t = ts_now - (n_samples - 1 - i) * step
                self.raw_buffer.append(float(val))
                self.timestamps.append(t)
                self.total_samples_ingested += 1

            # Run real-time filtering on the latest window if enough data
            if len(self.raw_buffer) >= 30:
                raw_arr = np.array(self.raw_buffer)
                try:
                    lowcut = 0.5
                    highcut = min(45.0, (self.sampling_rate / 2.0) - 1.0)
                    filtered = butter_bandpass_filter(raw_arr, lowcut=lowcut, highcut=highcut, fs=self.sampling_rate, order=3)
                    self.filtered_buffer = collections.deque(filtered, maxlen=self.buffer_size)
                except Exception:
                    self.filtered_buffer = collections.deque(raw_arr, maxlen=self.buffer_size)

    def detect_and_classify_beats(self) -> Optional[Dict[str, Any]]:
        """
        Checks current buffer for unanalyzed R-peaks and triggers ML inference
        on the most recent complete 256-sample beat window (100 samples pre-R, 156 samples post-R).
        """
        with self.lock:
            if len(self.filtered_buffer) < EXPECTED_WINDOW_SIZE:
                return None

            sig = np.array(self.filtered_buffer)
            total_len = len(sig)

        min_dist = int(0.4 * self.sampling_rate)
        sig_mean = np.mean(sig)
        sig_std = np.std(sig)
        height_thresh = sig_mean + 0.6 * sig_std

        peaks, _ = find_peaks(sig, distance=min_dist, height=height_thresh)

        if len(peaks) == 0:
            return None

        pre_r = 100
        post_r = 156

        valid_peaks = [p for p in peaks if (p >= pre_r and (p + post_r) <= total_len)]
        if not valid_peaks:
            return None

        target_peak = valid_peaks[-1]
        beat_window = sig[target_peak - pre_r : target_peak + post_r]

        if len(beat_window) != EXPECTED_WINDOW_SIZE:
            return None

        try:
            pred_result = self.predictor.predict(beat_window, apply_zscore=True)
            pred_result["timestamp"] = time.time()
            pred_result["peak_sample_index"] = int(target_peak)

            with self.lock:
                self.last_beat_prediction = pred_result
                self.recent_predictions.append(pred_result)
                self.total_beats_evaluated += 1
                if pred_result.get("is_abnormal", False):
                    self.total_abnormal_beats += 1

            return pred_result
        except Exception as e:
            return {
                "prediction": "ERROR",
                "class_id": -1,
                "abnormal_probability": 0.0,
                "confidence": 0.0,
                "is_abnormal": False,
                "threshold": self.predictor.threshold,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_latest_waveform_snapshot(self, max_points: int = 500) -> Dict[str, Any]:
        """
        Returns recent raw & filtered ECG waveform points for dashboard rendering.
        """
        with self.lock:
            n = min(len(self.filtered_buffer), max_points)
            if n == 0:
                return {
                    "raw": [],
                    "filtered": [],
                    "timestamps": [],
                    "count": 0
                }

            filtered_slice = list(self.filtered_buffer)[-n:]
            raw_slice = list(self.raw_buffer)[-n:] if len(self.raw_buffer) >= n else []
            ts_slice = list(self.timestamps)[-n:] if len(self.timestamps) >= n else []

            return {
                "raw": [round(float(v), 3) for v in raw_slice],
                "filtered": [round(float(v), 3) for v in filtered_slice],
                "timestamps": [round(float(t), 3) for t in ts_slice],
                "count": n
            }

    def get_screening_summary(self) -> Dict[str, Any]:
        """
        Calculates recent arrhythmia burden and returns prediction stats.
        """
        with self.lock:
            recent_list = list(self.recent_predictions)
            recent_count = len(recent_list)
            recent_abnormal = sum(1 for p in recent_list if p.get("is_abnormal", False))
            recent_burden = (recent_abnormal / recent_count * 100.0) if recent_count > 0 else 0.0

            return {
                "latest_beat": self.last_beat_prediction,
                "recent_evaluated_beats": recent_count,
                "recent_abnormal_count": recent_abnormal,
                "recent_arrhythmia_burden_pct": round(recent_burden, 1),
                "total_beats_evaluated": self.total_beats_evaluated,
                "total_abnormal_beats": self.total_abnormal_beats
            }
