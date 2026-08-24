"""
HealthSense - ECG Request and Response Validation Schemas
Validates ECG input parameters and structures output predictions safely.
"""

from typing import List, Dict, Any, Tuple, Optional
import math

EXPECTED_SAMPLE_COUNT = 256


def validate_ecg_payload(data: Any) -> Tuple[bool, Optional[List[float]], Optional[str]]:
    """
    Validates that the incoming payload contains exactly 256 numeric ECG samples.
    
    Checks:
    - payload is a dictionary
    - 'ecg' field is present and not None
    - 'ecg' is a list
    - len(ecg) == 256
    - all values are valid numeric floats/ints (not NaN, not Inf, not string)

    Returns:
        (is_valid: bool, cleaned_samples: Optional[List[float]], error_message: Optional[str])
    """
    if data is None:
        return False, None, "Request body is empty or None. Expected JSON object with 'ecg' field."

    if not isinstance(data, dict):
        return False, None, "Request payload must be a JSON object containing an 'ecg' key."

    if "ecg" not in data:
        return False, None, "Missing required field 'ecg' in JSON payload."

    ecg_list = data["ecg"]

    if ecg_list is None:
        return False, None, "Field 'ecg' cannot be None. Expected list of 256 numeric samples."

    if not isinstance(ecg_list, list):
        return False, None, f"Field 'ecg' must be a JSON array (list), got {type(ecg_list).__name__}."

    if len(ecg_list) == 0:
        return False, None, "Field 'ecg' is an empty array. Expected exactly 256 samples."

    if len(ecg_list) != EXPECTED_SAMPLE_COUNT:
        return False, None, f"Invalid ECG window length: {len(ecg_list)} samples. Expected exactly {EXPECTED_SAMPLE_COUNT} samples."

    # Validate individual numeric values
    cleaned: List[float] = []
    for i, val in enumerate(ecg_list):
        if val is None:
            return False, None, f"Sample at index {i} is None. All 256 samples must be numeric."

        if isinstance(val, bool):
            return False, None, f"Sample at index {i} is a boolean. All 256 samples must be numeric."

        if not isinstance(val, (int, float)):
            return False, None, f"Sample at index {i} has invalid type '{type(val).__name__}'. Expected float or int."

        # Check for NaN and Inf
        if math.isnan(val):
            return False, None, f"Sample at index {i} contains NaN (Not a Number)."

        if math.isinf(val):
            return False, None, f"Sample at index {i} contains Inf (Infinity)."

        cleaned.append(float(val))

    return True, cleaned, None
