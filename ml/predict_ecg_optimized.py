"""
HealthSense - Optimized ECG Arrhythmia Inference & Prediction Module
Provides real-time, robust inference on 256-sample ECG windows using the
trained HealthSense Optimized 1D-CNN binary classification model.

Model: ml/models/ecg_model_optimized.keras
Validated Decision Threshold: 0.30
Input Dimension: 256 samples, 1D-CNN (1, 256, 1)
"""

import os
import sys
from typing import Dict, Any, Union, List, Optional
import numpy as np

# Suppress verbose TensorFlow C++ logging
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "ecg_model_optimized.keras")

# Preprocessing & Model constants matching training pipeline
EXPECTED_WINDOW_SIZE = 256
DEFAULT_THRESHOLD = 0.30


def z_score_normalize(signal_segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standard Z-score normalization per beat segment: (x - mean) / (std + eps).
    Matches the exact training preprocessing pipeline in ml/preprocessing.py.
    """
    mean = np.mean(signal_segment)
    std = np.std(signal_segment)
    if std < eps:
        return signal_segment - mean
    return (signal_segment - mean) / (std + eps)


class ECGOptimizedPredictor:
    """
    ECG Arrhythmia Predictor wrapper for the trained HealthSense Optimized 1D-CNN model.
    Handles model loading, input validation, preprocessing, and single/batch inference.
    """

    def __init__(self, model_path: Optional[str] = None, threshold: float = DEFAULT_THRESHOLD):
        """
        Initializes the optimized predictor.
        
        Args:
            model_path: Path to ecg_model_optimized.keras. Defaults to ml/models/ecg_model_optimized.keras.
            threshold: Validated classification decision threshold for abnormality (default: 0.30).
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.threshold = float(threshold)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the saved Keras model into memory with comprehensive error checking.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Optimized ECG model file not found at: {self.model_path}. "
                "Please verify that ml/models/ecg_model_optimized.keras exists."
            )
        
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load optimized ECG model from {self.model_path}: {str(e)}"
            ) from e

    def preprocess_input(
        self,
        ecg_window: Union[List[float], np.ndarray],
        apply_zscore: bool = True
    ) -> np.ndarray:
        """
        Validates and preprocesses a single ECG window of 256 samples.

        Args:
            ecg_window: Iterable or numpy array containing 256 ECG voltage/amplitude values.
            apply_zscore: Whether to apply Z-score normalization. (True for raw ECG segments).

        Returns:
            np.ndarray of shape (1, 256, 1) with dtype float32.
        """
        if ecg_window is None:
            raise ValueError("Input ECG window cannot be None.")

        # Check for empty input
        if hasattr(ecg_window, '__len__') and len(ecg_window) == 0:
            raise ValueError("Input ECG window is empty. Expected exactly 256 samples.")

        # Convert to numpy array
        try:
            arr = np.asarray(ecg_window, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Input ECG window could not be converted to float32 array: {e}") from e

        # Empty array check (e.g. np.array([]))
        if arr.size == 0:
            raise ValueError("Input ECG window is empty. Expected exactly 256 samples.")

        # Flatten any trailing singleton dimensions
        flat_arr = arr.squeeze()

        # Validate dimension and size
        if flat_arr.ndim != 1:
            raise ValueError(
                f"Expected 1D ECG signal window, got array with shape {arr.shape}."
            )

        if len(flat_arr) != EXPECTED_WINDOW_SIZE:
            raise ValueError(
                f"Invalid ECG window length: {len(flat_arr)} samples. "
                f"Expected exactly {EXPECTED_WINDOW_SIZE} samples."
            )

        # Check for NaN or Inf
        if np.isnan(flat_arr).any():
            raise ValueError("Input ECG window contains NaN values.")
        if np.isinf(flat_arr).any():
            raise ValueError("Input ECG window contains Inf values.")

        # Apply Z-score normalization if required
        if apply_zscore:
            flat_arr = z_score_normalize(flat_arr)

        # Reshape to (1, 256, 1) expected by 1D-CNN
        processed = flat_arr.reshape((1, EXPECTED_WINDOW_SIZE, 1)).astype(np.float32)
        return processed

    def predict(
        self,
        ecg_window: Union[List[float], np.ndarray],
        apply_zscore: bool = True
    ) -> Dict[str, Any]:
        """
        Runs inference on a single 256-sample ECG window.

        Args:
            ecg_window: A list or 1D array of 256 ECG amplitude samples.
            apply_zscore: Whether to apply Z-score normalization.

        Returns:
            dict containing:
                - prediction: 'NORMAL' or 'ABNORMAL'
                - class_id: 0 (NORMAL) or 1 (ABNORMAL)
                - abnormal_probability: float between 0.0 and 1.0
                - confidence: float confidence score (0.0 to 1.0)
                - is_abnormal: bool
                - threshold: float threshold used (0.30)
                - status: 'SUCCESS'
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        # 1. Preprocess and validate input safely (rejects invalid input before calling model)
        input_tensor = self.preprocess_input(ecg_window, apply_zscore=apply_zscore)

        # 2. Run model forward pass
        try:
            raw_output = self.model(input_tensor, training=False)
            abnormal_prob = float(raw_output.numpy()[0][0])
        except Exception as e:
            raise RuntimeError(f"Error during model prediction: {str(e)}") from e

        # 3. Apply validated decision threshold (0.30)
        is_abnormal = bool(abnormal_prob >= self.threshold)
        prediction = "ABNORMAL" if is_abnormal else "NORMAL"
        class_id = 1 if is_abnormal else 0
        
        # Confidence represents certainty for the predicted class
        confidence = abnormal_prob if is_abnormal else (1.0 - abnormal_prob)

        return {
            "prediction": prediction,
            "class_id": class_id,
            "abnormal_probability": round(abnormal_prob, 6),
            "confidence": round(float(confidence), 6),
            "is_abnormal": is_abnormal,
            "threshold": self.threshold,
            "status": "SUCCESS"
        }

    def predict_batch(
        self,
        ecg_windows: Union[List[List[float]], np.ndarray],
        apply_zscore: bool = True,
        batch_size: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Runs batch inference on multiple 256-sample ECG windows.

        Args:
            ecg_windows: Array-like of shape (N, 256) or list of windows.
            apply_zscore: Whether to normalize each window.
            batch_size: Mini-batch size for model inference.

        Returns:
            List of prediction dictionary results.
        """
        if ecg_windows is None:
            raise ValueError("Input batch cannot be None.")

        raw_arr = np.asarray(ecg_windows, dtype=np.float32)
        if raw_arr.ndim != 2 or raw_arr.shape[1] != EXPECTED_WINDOW_SIZE:
            raise ValueError(
                f"Expected 2D array of shape (N, {EXPECTED_WINDOW_SIZE}), got {raw_arr.shape}"
            )

        if np.isnan(raw_arr).any():
            raise ValueError("Batch contains NaN values.")
        if np.isinf(raw_arr).any():
            raise ValueError("Batch contains Inf values.")

        if apply_zscore:
            processed = np.array([z_score_normalize(win) for win in raw_arr], dtype=np.float32)
        else:
            processed = raw_arr

        input_tensor = np.expand_dims(processed, axis=-1)  # (N, 256, 1)

        # Batch prediction with chunking for memory efficiency and performance
        raw_preds = self.model.predict(input_tensor, batch_size=batch_size, verbose=0).flatten()

        results = []
        for prob in raw_preds:
            p = float(prob)
            is_ab = bool(p >= self.threshold)
            pred = "ABNORMAL" if is_ab else "NORMAL"
            cid = 1 if is_ab else 0
            conf = p if is_ab else (1.0 - p)
            results.append({
                "prediction": pred,
                "class_id": cid,
                "abnormal_probability": round(p, 6),
                "confidence": round(float(conf), 6),
                "is_abnormal": is_ab,
                "threshold": self.threshold,
                "status": "SUCCESS"
            })

        return results


# Global singleton instance for efficient module-level calls
_OPTIMIZED_PREDICTOR_INSTANCE: Optional[ECGOptimizedPredictor] = None


def get_predictor(model_path: Optional[str] = None, threshold: float = DEFAULT_THRESHOLD) -> ECGOptimizedPredictor:
    """
    Returns a cached or newly initialized ECGOptimizedPredictor singleton.
    """
    global _OPTIMIZED_PREDICTOR_INSTANCE
    if _OPTIMIZED_PREDICTOR_INSTANCE is None or (model_path and _OPTIMIZED_PREDICTOR_INSTANCE.model_path != model_path):
        _OPTIMIZED_PREDICTOR_INSTANCE = ECGOptimizedPredictor(model_path=model_path, threshold=threshold)
    return _OPTIMIZED_PREDICTOR_INSTANCE


def predict_ecg_beat(
    ecg_window: Union[List[float], np.ndarray],
    model_path: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    apply_zscore: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to predict abnormality for a single 256-sample ECG beat window
    using the optimized model and validated 0.30 threshold.

    Args:
        ecg_window: 256 numerical samples representing an ECG beat window.
        model_path: Optional path to ecg_model_optimized.keras.
        threshold: Validated decision threshold for abnormality (default: 0.30).
        apply_zscore: Whether to apply Z-score normalization.

    Returns:
        Dictionary with prediction, class_id, abnormal_probability, confidence, threshold, and status.
    """
    predictor = get_predictor(model_path=model_path, threshold=threshold)
    return predictor.predict(ecg_window, apply_zscore=apply_zscore)


if __name__ == "__main__":
    print("=" * 80)
    print("HealthSense Optimized ECG Predictor - Standalone Verification")
    print("=" * 80)

    predictor = ECGOptimizedPredictor()
    print(f"Loaded Model: {predictor.model_path}")
    print(f"Validated Decision Threshold: {predictor.threshold}\n")

    test_data_path = os.path.join(BASE_DIR, "data", "test_data.npz")
    if os.path.exists(test_data_path):
        print(f"Loading test dataset from: {test_data_path}")
        test_npz = np.load(test_data_path)
        X_test = test_npz['X']
        y_test = test_npz['y_binary']

        # Normal Sample (Index 0)
        norm_idx = 0
        res_norm = predictor.predict(X_test[norm_idx], apply_zscore=False)
        print(f"\n[Known Normal Sample] Index {norm_idx}:")
        print(f"  -> True Label          : NORMAL (0)")
        print(f"  -> Predicted Label     : {res_norm['prediction']}")
        print(f"  -> Abnormal Probability: {res_norm['abnormal_probability']:.6f}")
        print(f"  -> Confidence          : {res_norm['confidence']:.6f} ({res_norm['confidence']*100:.2f}%)")
        print(f"  -> Threshold           : {res_norm['threshold']}")
        print(f"  -> Status              : {res_norm['status']}")
        print(f"  -> Match Result        : {'PASS' if res_norm['prediction'] == 'NORMAL' else 'FAIL'}")

        # Abnormal Sample (Index 6077 - PVC beat)
        abnorm_idx = 6077
        res_abnorm = predictor.predict(X_test[abnorm_idx], apply_zscore=False)
        print(f"\n[Known Abnormal Sample] Index {abnorm_idx}:")
        print(f"  -> True Label          : ABNORMAL (1)")
        print(f"  -> Predicted Label     : {res_abnorm['prediction']}")
        print(f"  -> Abnormal Probability: {res_abnorm['abnormal_probability']:.6f}")
        print(f"  -> Confidence          : {res_abnorm['confidence']:.6f} ({res_abnorm['confidence']*100:.2f}%)")
        print(f"  -> Threshold           : {res_abnorm['threshold']}")
        print(f"  -> Status              : {res_abnorm['status']}")
        print(f"  -> Match Result        : {'PASS' if res_abnorm['prediction'] == 'ABNORMAL' else 'FAIL'}")
