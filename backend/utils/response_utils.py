"""
HealthSense - Response Utilities & Standard Response Formatters
"""

from typing import Dict, Any, Optional
from flask import jsonify, Response


def success_response(data: Dict[str, Any], status_code: int = 200) -> Response:
    """Returns a standardized JSON success response."""
    return jsonify(data), status_code


def error_response(message: str, error_code: str = "BAD_REQUEST", status_code: int = 400, details: Optional[Any] = None) -> Response:
    """
    Returns a standardized, safe JSON error response.
    Never exposes internal tracebacks or sensitive server information.
    """
    payload: Dict[str, Any] = {
        "error": message,
        "error_code": error_code,
        "status": "ERROR"
    }
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status_code
