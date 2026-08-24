"""
HealthSense - ECG Prediction & Model Information Routes
Provides dedicated REST endpoints for single-beat ECG inference, model metadata,
and verified benchmark sample retrieval.
"""

import os
import json
import numpy as np
from flask import Blueprint, request
from backend.services.ecg_service import get_ecg_service
from backend.schemas.ecg_schemas import validate_ecg_payload
from backend.utils.response_utils import success_response, error_response

ecg_bp = Blueprint("ecg_bp", __name__)


@ecg_bp.route("/api/ecg/predict", methods=["POST"])
def predict_ecg():
    """
    POST /api/ecg/predict
    Accepts 256 numeric ECG samples and runs inference using the optimized 1D-CNN model.

    Request Format:
    {
        "ecg": [256 numeric float values]
    }

    Response Format (Normal):
    {
        "prediction": "NORMAL",
        "abnormal_probability": 0.008877,
        "confidence": 0.991123,
        "threshold": 0.30,
        "class_id": 0,
        "status": "SUCCESS"
    }

    Response Format (Abnormal):
    {
        "prediction": "ABNORMAL",
        "abnormal_probability": 0.991265,
        "confidence": 0.991265,
        "threshold": 0.30,
        "class_id": 1,
        "status": "SUCCESS"
    }
    """
    # 1. Verify JSON content type
    if not request.is_json:
        return error_response(
            "Content-Type must be application/json",
            error_code="INVALID_CONTENT_TYPE",
            status_code=400
        )

    # 2. Extract payload safely
    try:
        data = request.get_json(silent=True)
    except Exception as e:
        return error_response(
            f"Malformed JSON payload: {str(e)}",
            error_code="MALFORMED_JSON",
            status_code=400
        )

    # 3. Input Validation (Checks length == 256, numbers, NaN, Inf, None, empty)
    is_valid, cleaned_samples, err_msg = validate_ecg_payload(data)
    if not is_valid or cleaned_samples is None:
        return error_response(
            err_msg or "Invalid ECG payload.",
            error_code="VALIDATION_ERROR",
            status_code=400
        )

    # 4. Run prediction using the verified service (does not duplicate preprocessing)
    try:
        ecg_service = get_ecg_service()
        result = ecg_service.predict_single_beat(cleaned_samples)
        return success_response(result, status_code=200)
    except ValueError as val_err:
        return error_response(
            str(val_err),
            error_code="PREPROCESSING_VALIDATION_ERROR",
            status_code=400
        )
    except Exception as exc:
        return error_response(
            "An internal error occurred while evaluating the ECG sample.",
            error_code="INFERENCE_ERROR",
            status_code=500,
            details=str(exc)
        )


@ecg_bp.route("/api/ecg/model-info", methods=["GET"])
def get_model_info():
    """
    GET /api/ecg/model-info
    Returns safe, verified model metadata and previously validated test-set metrics.
    """
    ecg_service = get_ecg_service()
    metadata = ecg_service.get_model_metadata()
    
    return success_response({
        "model": "Optimized 1D-CNN",
        "input_length": 256,
        "threshold": metadata["threshold"],
        "accuracy": 0.7902,
        "precision": 0.5503,
        "recall": 0.5763,
        "specificity": 0.8558,
        "f1_score": 0.5630,
        "roc_auc": 0.8056,
        "preprocessing": metadata["preprocessing"],
        "test_dataset": "58,448 beats across 26 unseen patient records",
        "status": metadata["status"],
        "disclaimer": "Validated test-set metrics. Prototype for research and educational purposes; not for clinical diagnosis."
    })


@ecg_bp.route("/api/ecg/sample-beats", methods=["GET"])
def get_sample_beats():
    """
    GET /api/ecg/sample-beats
    Returns pre-extracted verified normal and abnormal ECG beats for dashboard demonstration.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data_path = os.path.join(base_dir, "ml", "data", "test_data.npz")
    
    if not os.path.exists(test_data_path):
        return error_response("Test dataset not found on server.", status_code=404)

    try:
        test_npz = np.load(test_data_path)
        X_test = test_npz['X']
        
        # Sample 0: Known Normal beat
        sample_norm = [round(float(v), 5) for v in X_test[0]]
        # Sample 6077: Known Abnormal (PVC) beat
        sample_abnorm = [round(float(v), 5) for v in X_test[6077]]
        
        return success_response({
            "normal_beat": {
                "sample_index": 0,
                "label": "NORMAL",
                "patient_id": "100",
                "samples": sample_norm
            },
            "abnormal_beat": {
                "sample_index": 6077,
                "label": "ABNORMAL (PVC)",
                "patient_id": "105",
                "samples": sample_abnorm
            }
        })
    except Exception as e:
        return error_response(f"Failed to load sample beats: {str(e)}", status_code=500)
