"""
HealthSense - Multi-Parameter Health Assessment & Screening Engine
Evaluates physiological telemetry (Heart Rate, SpO2, Temperature) in conjunction
with real-time ML ECG arrhythmia classification to produce multi-level screening status.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class VitalThresholds:
    # Heart Rate (BPM)
    hr_bradycardia_critical: float = 45.0
    hr_bradycardia_warning: float = 60.0
    hr_tachycardia_warning: float = 100.0
    hr_tachycardia_critical: float = 130.0

    # Blood Oxygen SpO2 (%)
    spo2_critical: float = 90.0
    spo2_warning: float = 95.0

    # Temperature (°C)
    temp_hypothermia_critical: float = 35.0
    temp_hypothermia_warning: float = 36.0
    temp_fever_warning: float = 37.8
    temp_fever_critical: float = 39.0

    # ECG ML Abnormality Probability
    ecg_high_confidence_abnormal: float = 0.70
    ecg_borderline_abnormal: float = 0.50


class HealthAssessmentEngine:
    """
    Evaluates real-time patient telemetry and assigns multi-parameter health risk status:
    - NORMAL: All vitals and ECG are nominal.
    - NEEDS OBSERVATION: Mild vitals deviation or borderline ECG anomaly detected.
    - HIGH RISK: Critical vitals deviation or high-confidence arrhythmia detected.
    """

    def __init__(self, thresholds: Optional[VitalThresholds] = None):
        self.thresholds = thresholds or VitalThresholds()

    def evaluate_heart_rate(self, bpm: Optional[float]) -> Dict[str, Any]:
        """Evaluates heart rate telemetry in Beats Per Minute."""
        if bpm is None:
            return {"status": "UNKNOWN", "level": "UNKNOWN", "message": "No HR telemetry available"}

        bpm = float(bpm)
        t = self.thresholds

        if bpm < t.hr_bradycardia_critical:
            return {
                "status": "CRITICAL_LOW",
                "level": "HIGH_RISK",
                "message": f"Severe Bradycardia detected: {bpm:.1f} BPM (< {t.hr_bradycardia_critical} BPM)"
            }
        elif bpm < t.hr_bradycardia_warning:
            return {
                "status": "LOW",
                "level": "NEEDS_OBSERVATION",
                "message": f"Mild Bradycardia: {bpm:.1f} BPM (< {t.hr_bradycardia_warning} BPM)"
            }
        elif bpm > t.hr_tachycardia_critical:
            return {
                "status": "CRITICAL_HIGH",
                "level": "HIGH_RISK",
                "message": f"Severe Tachycardia detected: {bpm:.1f} BPM (> {t.hr_tachycardia_critical} BPM)"
            }
        elif bpm > t.hr_tachycardia_warning:
            return {
                "status": "HIGH",
                "level": "NEEDS_OBSERVATION",
                "message": f"Mild Tachycardia: {bpm:.1f} BPM (> {t.hr_tachycardia_warning} BPM)"
            }
        else:
            return {
                "status": "NORMAL",
                "level": "NORMAL",
                "message": f"Heart rate nominal: {bpm:.1f} BPM"
            }

    def evaluate_spo2(self, spo2: Optional[float]) -> Dict[str, Any]:
        """Evaluates blood oxygen saturation percentage."""
        if spo2 is None:
            return {"status": "UNKNOWN", "level": "UNKNOWN", "message": "No SpO2 telemetry available"}

        spo2 = float(spo2)
        t = self.thresholds

        if spo2 < t.spo2_critical:
            return {
                "status": "CRITICAL_LOW",
                "level": "HIGH_RISK",
                "message": f"Critical Hypoxemia detected: {spo2:.1f}% (< {t.spo2_critical}%)"
            }
        elif spo2 < t.spo2_warning:
            return {
                "status": "LOW",
                "level": "NEEDS_OBSERVATION",
                "message": f"Borderline Blood Oxygen: {spo2:.1f}% (< {t.spo2_warning}%)"
            }
        else:
            return {
                "status": "NORMAL",
                "level": "NORMAL",
                "message": f"SpO2 nominal: {spo2:.1f}%"
            }

    def evaluate_temperature(self, temp_c: Optional[float]) -> Dict[str, Any]:
        """Evaluates body temperature in degrees Celsius."""
        if temp_c is None:
            return {"status": "UNKNOWN", "level": "UNKNOWN", "message": "No Temperature telemetry available"}

        temp_c = float(temp_c)
        t = self.thresholds

        if temp_c < t.temp_hypothermia_critical:
            return {
                "status": "CRITICAL_LOW",
                "level": "HIGH_RISK",
                "message": f"Severe Hypothermia: {temp_c:.1f} °C (< {t.temp_hypothermia_critical} °C)"
            }
        elif temp_c < t.temp_hypothermia_warning:
            return {
                "status": "LOW",
                "level": "NEEDS_OBSERVATION",
                "message": f"Mild Hypothermia: {temp_c:.1f} °C (< {t.temp_hypothermia_warning} °C)"
            }
        elif temp_c >= t.temp_fever_critical:
            return {
                "status": "CRITICAL_HIGH",
                "level": "HIGH_RISK",
                "message": f"High Pyrexia (Severe Fever): {temp_c:.1f} °C (>= {t.temp_fever_critical} °C)"
            }
        elif temp_c >= t.temp_fever_warning:
            return {
                "status": "HIGH",
                "level": "NEEDS_OBSERVATION",
                "message": f"Elevated Temperature (Fever): {temp_c:.1f} °C (>= {t.temp_fever_warning} °C)"
            }
        else:
            return {
                "status": "NORMAL",
                "level": "NORMAL",
                "message": f"Temperature nominal: {temp_c:.1f} °C"
            }

    def evaluate_ecg_ml(self, ml_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates ML Arrhythmia model prediction."""
        if not ml_result or "prediction" not in ml_result:
            return {
                "status": "UNKNOWN",
                "level": "UNKNOWN",
                "message": "No ECG ML prediction available",
                "probability": None
            }

        prob = float(ml_result.get("abnormal_probability", 0.0))
        is_abnormal = bool(ml_result.get("is_abnormal", prob >= 0.5))
        t = self.thresholds

        if prob >= t.ecg_high_confidence_abnormal:
            return {
                "status": "ARRHYTHMIA_HIGH_CONFIDENCE",
                "level": "HIGH_RISK",
                "message": f"High-confidence Arrhythmia rhythm detected (Prob: {prob:.2f})",
                "probability": prob
            }
        elif is_abnormal or prob >= t.ecg_borderline_abnormal:
            return {
                "status": "ARRHYTHMIA_BORDERLINE",
                "level": "NEEDS_OBSERVATION",
                "message": f"Potential Arrhythmic/Ectopic beat pattern flagged (Prob: {prob:.2f})",
                "probability": prob
            }
        else:
            return {
                "status": "NORMAL_SINUS",
                "level": "NORMAL",
                "message": f"Normal sinus rhythm pattern (Abnormal Prob: {prob:.2f})",
                "probability": prob
            }

    def assess_health(
        self,
        heart_rate: Optional[float] = None,
        spo2: Optional[float] = None,
        temperature: Optional[float] = None,
        ecg_ml_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes multi-sensor telemetry and ML screening into a unified health report.
        """
        hr_eval = self.evaluate_heart_rate(heart_rate)
        spo2_eval = self.evaluate_spo2(spo2)
        temp_eval = self.evaluate_temperature(temperature)
        ecg_eval = self.evaluate_ecg_ml(ecg_ml_result)

        findings: List[str] = []
        levels = [hr_eval["level"], spo2_eval["level"], temp_eval["level"], ecg_eval["level"]]

        # Collect warning messages
        for item in [hr_eval, spo2_eval, temp_eval, ecg_eval]:
            if item["level"] in ("HIGH_RISK", "NEEDS_OBSERVATION"):
                findings.append(item["message"])

        # Determine overall category
        if "HIGH_RISK" in levels:
            overall_status = "HIGH RISK"
            status_code = "HIGH_RISK"
            status_color = "#ef4444"  # Red
            summary = "Significant physiological anomalies or high-confidence arrhythmia detected."
        elif "NEEDS_OBSERVATION" in levels:
            overall_status = "NEEDS OBSERVATION"
            status_code = "NEEDS_OBSERVATION"
            status_color = "#f59e0b"  # Amber
            summary = "Mild vitals deviation or potential rhythm anomaly requires monitoring."
        elif all(lvl == "NORMAL" for lvl in levels if lvl != "UNKNOWN"):
            overall_status = "NORMAL"
            status_code = "NORMAL"
            status_color = "#10b981"  # Emerald
            summary = "All physiological parameters and ECG screening within nominal ranges."
        else:
            overall_status = "AWAITING TELEMETRY"
            status_code = "UNKNOWN"
            status_color = "#94a3b8"  # Slate
            summary = "Streaming telemetry from sensors..."

        return {
            "overall_status": overall_status,
            "status_code": status_code,
            "status_color": status_color,
            "summary": summary,
            "findings": findings if findings else ["All monitored vitals are within target ranges."],
            "details": {
                "heart_rate": hr_eval,
                "spo2": spo2_eval,
                "temperature": temp_eval,
                "ecg_ml": ecg_eval
            }
        }
