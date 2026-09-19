"""
HealthSense - Medical Report Generation Endpoint
Generates printable/downloadable medical screening reports based on physical telemetry,
trained ML ECG model analysis, and multi-parameter health assessment rules.
"""

import time
import json
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, make_response, send_file
from backend.services.user_service import get_db_connection
from backend.utils.response_utils import success_response, error_response
from backend.health_engine import HealthAssessmentEngine

report_bp = Blueprint("report_bp", __name__)
health_engine = HealthAssessmentEngine()


@report_bp.route("/api/report/generate", methods=["POST"])
def generate_medical_report():
    """
    Generates structured HTML / printable PDF medical report data based on actual measurements.
    """
    data = request.get_json(silent=True) or {}
    patient_id = data.get("patient_id", "PNT-2026-0001")
    patient_name = data.get("patient_name", "Prajkta Patil")
    device_id = data.get("device_id", "HS-ESP32-HARDWARE")
    
    # Measured vitals (or None if sensor not connected)
    hr = data.get("heart_rate")
    spo2 = data.get("spo2")
    temp = data.get("temperature")
    ecg_result = data.get("ecg_result")
    sensor_status = data.get("sensor_status", {})

    # Generate multi-parameter assessment
    assessment = health_engine.assess_health(
        heart_rate=hr,
        spo2=spo2,
        temperature=temp,
        ecg_ml_result=ecg_result
    )

    # Format text labels strictly enforcing physical sensor rules
    hr_str = f"{float(hr):.0f} BPM" if (hr is not None and hr > 0) else "Not Available — MAX30102 sensor not connected or no finger contact"
    spo2_str = f"{float(spo2):.1f}%" if (spo2 is not None and spo2 > 0) else "Not Available — MAX30102 sensor not connected or no finger contact"
    temp_str = f"{float(temp):.1f} °C" if (temp is not None and temp > 0) else "Not Available — DS18B20 temperature sensor not connected"
    
    ecg_str = "No recording sample evaluated"
    if ecg_result:
        prob = ecg_result.get("abnormal_probability", 0.0) * 100.0
        pred = ecg_result.get("prediction", "UNKNOWN")
        ecg_str = f"{pred} (Arrhythmia Risk: {prob:.1f}%)"

    report_payload = {
        "report_id": f"RPT-{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "patient": {
            "patient_id": patient_id,
            "full_name": patient_name
        },
        "device": {
            "device_id": device_id,
            "status": "ONLINE"
        },
        "measured_vitals": {
            "heart_rate": hr_str,
            "spo2": spo2_str,
            "temperature": temp_str,
            "raw_heart_rate": hr,
            "raw_spo2": spo2,
            "raw_temperature": temp
        },
        "ecg_ml_analysis": {
            "summary": ecg_str,
            "details": ecg_result
        },
        "health_assessment": assessment,
        "disclaimer": "HealthSense is an experimental research software screening system. This report is for monitoring purposes only and does NOT constitute a clinical diagnosis or medical device advice."
    }

    # Format HTML string printable report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HealthSense Medical Screening Report - {report_payload['report_id']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #1e293b; padding: 40px; background: #fff; }}
        .header {{ border-bottom: 3px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; }}
        .title {{ font-size: 24px; font-weight: bold; color: #0f172a; margin: 0; }}
        .subtitle {{ font-size: 14px; color: #64748b; margin-top: 5px; }}
        .badge {{ display: inline-block; padding: 6px 12px; font-weight: bold; border-radius: 4px; color: #fff; background: {assessment['status_color']}; }}
        .section {{ margin-bottom: 25px; padding: 15px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; }}
        .section-title {{ font-size: 16px; font-weight: bold; border-bottom: 1px solid #cbd5e1; padding-bottom: 8px; margin-bottom: 12px; color: #0f172a; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; color: #475569; }}
        .disclaimer {{ font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 15px; margin-top: 40px; font-style: italic; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">HEALTHSENSE MEDICAL MONITORING REPORT</div>
            <div class="subtitle">Real-Time Physiological Telemetry & ML Screening Summary</div>
        </div>
        <div>
            <div class="badge">{assessment['overall_status']}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">PATIENT & RECORDING INFORMATION</div>
        <table>
            <tr><td><strong>Patient ID:</strong> {patient_id}</td><td><strong>Report ID:</strong> {report_payload['report_id']}</td></tr>
            <tr><td><strong>Patient Name:</strong> {patient_name}</td><td><strong>Generated Date:</strong> {report_payload['generated_at']}</td></tr>
            <tr><td><strong>Device ID:</strong> {device_id}</td><td><strong>Source:</strong> Physical IoT Sensors</td></tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">ACTUAL MEASURED VITALS (PHYSICAL SENSORS)</div>
        <table>
            <tr><th>Vital Parameter</th><th>Physical Sensor Source</th><th>Measured Value</th></tr>
            <tr><td><strong>Heart Rate (BPM)</strong></td><td>MAX30102 Optical Sensor</td><td>{hr_str}</td></tr>
            <tr><td><strong>Blood Oxygen (SpO2)</strong></td><td>MAX30102 Optical Sensor</td><td>{spo2_str}</td></tr>
            <tr><td><strong>Body Temperature</strong></td><td>DS18B20 1-Wire Digital Sensor</td><td>{temp_str}</td></tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">1D-CNN ECG ML MODEL ANALYSIS</div>
        <p><strong>Arrhythmia Model Classification:</strong> {ecg_str}</p>
        <p><em>Model configuration: 1D-CNN, 250 Hz sampling rate, 256-sample windowing, decision threshold 0.30.</em></p>
        <div style="background: #e0f2fe; padding: 10px 12px; border-left: 4px solid #0284c7; margin-top: 10px; border-radius: 4px; font-size: 13px;">
            <strong>OLED Hardware & Waveform Note:</strong> The live ECG graph shown on the physical ESP32 OLED display (and web dashboard) is recorded directly from the AD8232 sensor module. That same physical ECG waveform signal is what is evaluated by the AI model in this report.
        </div>
    </div>

    <div class="section">
        <div class="section-title">UNDERSTANDING ECG & ARRHYTHMIA (SIMPLE EXPLANATION)</div>
        <p><strong>What is an ECG?</strong> An Electrocardiogram (ECG) is a recording of the electrical signals produced by your heart every time it beats. It shows how fast your heart is beating and whether its rhythm is steady.</p>
        <p><strong>What is an Arrhythmia?</strong> An <em>Arrhythmia</em> means an irregular or abnormal heartbeat. It occurs when the electrical signals that coordinate heartbeats do not trigger properly, making the heart beat too fast (tachycardia), too slow (bradycardia), or irregularly (like premature ventricular contractions - PVCs).</p>
    </div>

    <div class="section">
        <div class="section-title">SOFTWARE HEALTH INTERPRETATION SUMMARY</div>
        <p><strong>Status:</strong> {assessment['summary']}</p>
        <ul>
            {"".join(f"<li>{finding}</li>" for finding in assessment['findings'])}
        </ul>
    </div>

    <div class="disclaimer">
        <strong>Medical Disclaimer:</strong> {report_payload['disclaimer']}
    </div>
</body>
</html>"""

    # If caller requested raw HTML for printing/download
    if request.args.get("format") == "html":
        resp = make_response(html_content)
        resp.headers["Content-Type"] = "text/html"
        return resp

    return success_response({
        "report": report_payload,
        "html": html_content
    })
