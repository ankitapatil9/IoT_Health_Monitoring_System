"""
HealthSense - ECG Inference Service
Delegates inference requests strictly to the verified ml/predict_ecg_optimized.py module.
Does NOT contain duplicate preprocessing or threshold logic.
"""

import os
import sys
from typing import Dict, Any, List

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.predict_ecg_optimized import get_predictor, DEFAULT_THRESHOLD, ECGOptimizedPredictor


class ECGInferenceService:
    """
    Singleton service wrapper around the verified ECGOptimizedPredictor.
    """
    def __init__(self):
        # Initializes singleton model instance
        self.predictor = get_predictor()

    def predict_single_beat(self, ecg_samples: List[float]) -> Dict[str, Any]:
        """
        Executes prediction on 256 ECG samples using the optimized predictor.
        
        Args:
            ecg_samples: 256 validated numeric ECG amplitude values.

        Returns:
            Structured prediction dictionary matching the HealthSense API contract:
            {
                "prediction": "NORMAL" | "ABNORMAL",
                "abnormal_probability": float,
                "confidence": float,
                "threshold": float,
                "class_id": int,
                "status": "SUCCESS"
            }
        """
        # Call the verified optimized predictor (single source of truth)
        raw_result = self.predictor.predict(ecg_samples, apply_zscore=True)
        
        return {
            "prediction": raw_result["prediction"],
            "abnormal_probability": raw_result["abnormal_probability"],
            "confidence": raw_result["confidence"],
            "threshold": raw_result["threshold"],
            "class_id": raw_result["class_id"],
            "status": "SUCCESS"
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Returns safe, verified model metadata and test benchmark performance.
        Does NOT expose sensitive filesystem paths.
        """
        return {
            "model": "Optimized 1D-CNN",
            "model_architecture": "1D Convolutional Neural Network (3 Conv Blocks + GlobalAveragePooling1D + Dense)",
            "input_length": 256,
            "sampling_frequency_hz": 360,
            "threshold": self.predictor.threshold,
            "preprocessing": "Per-beat Z-score normalization (mean 0, variance 1)",
            "performance_benchmark": {
                "accuracy": 0.7902,
                "precision": 0.5503,
                "recall_sensitivity": 0.5763,
                "specificity": 0.8558,
                "f1_score": 0.5630,
                "roc_auc": 0.8056,
                "test_dataset_beats": 58448,
                "unseen_test_patients": 26,
                "note": "Validated test-set benchmark on MIT-BIH Arrhythmia Database. Prototype for research and educational purposes."
            },
            "status": "LOADED" if self.predictor.model is not None else "ERROR"
        }


# Global service instance
_ecg_service_instance: ECGInferenceService = None


def get_ecg_service() -> ECGInferenceService:
    global _ecg_service_instance
    if _ecg_service_instance is None:
        _ecg_service_instance = ECGInferenceService()
    return _ecg_service_instance
