"""
HealthSense - Health Check & System Status Routes
"""

import time
from flask import Blueprint, jsonify
from backend.services.ecg_service import get_ecg_service
from backend.utils.response_utils import success_response

health_bp = Blueprint("health_bp", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Returns service operational status, model loaded state, and active decision threshold.
    """
    ecg_service = get_ecg_service()
    is_model_loaded = ecg_service.predictor.model is not None

    return success_response({
        "status": "healthy" if is_model_loaded else "degraded",
        "service": "HealthSense API",
        "model": "ecg_model_optimized.keras",
        "threshold": ecg_service.predictor.threshold
    })


@health_bp.route("/api/status", methods=["GET"])
def system_status():
    """
    Detailed system status check for telemetry and service health monitoring.
    """
    ecg_service = get_ecg_service()
    return success_response({
        "status": "healthy",
        "service": "HealthSense Real-Time & ML API",
        "model": "ecg_model_optimized.keras",
        "threshold": ecg_service.predictor.threshold,
        "input_dimension": 256,
        "model_loaded": ecg_service.predictor.model is not None,
        "timestamp": time.time()
    })
