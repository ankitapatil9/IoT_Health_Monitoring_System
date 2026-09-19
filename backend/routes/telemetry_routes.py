"""
HealthSense - Real-Time Telemetry & Hardware Ingestion Gateway
Handles physical ESP32 sensor packet ingestion, Server-Sent Events (SSE) streaming,
and isolated simulation mode for testing.
"""

import time
import json
import queue
import threading
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, Response
from backend.services.stream_processor import ECGStreamProcessor
from backend.health_engine import HealthAssessmentEngine
from backend.simulator import TelemetrySimulator
from backend.utils.response_utils import success_response, error_response

telemetry_bp = Blueprint("telemetry_bp", __name__)

# State & Processing Engines
health_engine = HealthAssessmentEngine()
stream_processor = ECGStreamProcessor(buffer_size=1500, sampling_rate=250)

# Global Telemetry Cache - Starts with NULL vitals until physical ESP32 data arrives!
latest_telemetry: Dict[str, Any] = {
    "device_id": "HEALTHSENSE_NODE_01",
    "timestamp": time.time(),
    "heart_rate": None,
    "spo2": None,
    "temperature": None,
    "leads_off": True,
    "sensor_status": {
        "max30102": False,
        "ds18b20": False,
        "ad8232": False
    },
    "ml_prediction": None,
    "assessment": health_engine.assess_health(None, None, None, None),
    "screening_summary": stream_processor.get_screening_summary(),
    "source": "AWAITING_TELEMETRY"
}
telemetry_lock = threading.Lock()

# SSE Client Subscribers
sse_clients: List[queue.Queue] = []
sse_lock = threading.Lock()


def broadcast_sse_event(event_type: str, data: Dict[str, Any]):
    """Pushes a message to all connected SSE clients."""
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        dead_clients = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead_clients.append(q)
        for dead in dead_clients:
            if dead in sse_clients:
                sse_clients.remove(dead)


def process_incoming_packet(packet: Dict[str, Any]):
    """
    Central ingestion pipeline for physical ESP32 telemetry packets.
    Strictly handles null / unavailable sensor data.
    """
    global latest_telemetry

    ecg_samples = packet.get("ecg_samples") or packet.get("ecg") or []
    hr = packet.get("heart_rate") if packet.get("heart_rate") is not None else packet.get("heartRate")
    spo2 = packet.get("spo2")
    temp = packet.get("temperature")
    leads_off = packet.get("leads_off") if packet.get("leads_off") is not None else packet.get("ecgLeadOff", False)
    device_id = packet.get("device_id") or packet.get("deviceId") or "HS-ESP32-NODE"
    sensor_status = packet.get("sensor_status", {
        "max30102": (hr is not None and float(hr) > 0 if isinstance(hr, (int, float)) else False),
        "ds18b20": (temp is not None and float(temp) > 0 if isinstance(temp, (int, float)) else False),
        "ad8232": not leads_off
    })

    # Validate numbers (convert -127, negative numbers, or invalid inputs to None)
    if hr is not None:
        try:
            hr = float(hr)
            if hr <= 0 or hr > 250:
                hr = None
        except (ValueError, TypeError):
            hr = None

    if spo2 is not None:
        try:
            spo2 = float(spo2)
            if spo2 <= 0 or spo2 > 100:
                spo2 = None
        except (ValueError, TypeError):
            spo2 = None

    if temp is not None:
        try:
            temp = float(temp)
            if temp <= 0 or temp < 20.0 or temp > 50.0:  # -127°C or invalid DS18B20 reading
                temp = None
        except (ValueError, TypeError):
            temp = None

    # Ingest ECG samples if present and leads are connected
    ml_result = None
    if ecg_samples and not leads_off:
        stream_processor.ingest_samples(ecg_samples)
        ml_result = stream_processor.detect_and_classify_beats()

    if ml_result is None:
        ml_result = stream_processor.last_beat_prediction if stream_processor.total_beats_evaluated > 0 else None

    # Synthesize health assessment
    assessment = health_engine.assess_health(
        heart_rate=hr,
        spo2=spo2,
        temperature=temp,
        ecg_ml_result=ml_result
    )
    screening_summary = stream_processor.get_screening_summary()

    with telemetry_lock:
        latest_telemetry = {
            "device_id": device_id,
            "timestamp": time.time(),
            "heart_rate": hr,
            "spo2": spo2,
            "temperature": temp,
            "leads_off": leads_off,
            "sensor_status": sensor_status,
            "ml_prediction": ml_result,
            "assessment": assessment,
            "screening_summary": screening_summary,
            "source": packet.get("scenario", "PHYSICAL_DEVICE")
        }

    waveform_snapshot = stream_processor.get_latest_waveform_snapshot(max_points=500)
    broadcast_data = {
        "timestamp": time.time(),
        "device_id": device_id,
        "heart_rate": hr,
        "spo2": spo2,
        "temperature": temp,
        "leads_off": leads_off,
        "sensor_status": sensor_status,
        "ecg_points": ecg_samples,
        "waveform": waveform_snapshot,
        "ml_prediction": ml_result,
        "assessment": assessment,
        "screening_summary": screening_summary
    }
    broadcast_sse_event("telemetry", broadcast_data)


# Simulator instance available for isolated DEV mode testing only
simulator = TelemetrySimulator(sample_rate=250, on_packet_callback=process_incoming_packet)


@telemetry_bp.route("/api/telemetry", methods=["POST"])
@telemetry_bp.route("/api/device/data", methods=["POST"])
def ingest_telemetry():
    """Ingestion endpoint for physical ESP32 devices."""
    if not request.is_json:
        return error_response("Content-Type must be application/json", status_code=400)

    data = request.get_json(silent=True)
    if not data:
        return error_response("Empty payload", status_code=400)

    process_incoming_packet(data)
    return success_response({"status": "received", "timestamp": time.time()})


@telemetry_bp.route("/api/latest", methods=["GET"])
@telemetry_bp.route("/api/health/latest", methods=["GET"])
def get_latest_telemetry():
    """Returns recent physical telemetry snapshot, ML predictions, and assessment."""
    with telemetry_lock:
        snapshot = dict(latest_telemetry)
        snapshot["waveform"] = stream_processor.get_latest_waveform_snapshot(max_points=500)
    return success_response(snapshot)


@telemetry_bp.route("/api/stream", methods=["GET"])
def stream_sse():
    """Server-Sent Events (SSE) continuous stream for the frontend web dashboard."""
    def event_generator():
        client_queue = queue.Queue(maxsize=100)
        with sse_lock:
            sse_clients.append(client_queue)

        try:
            with telemetry_lock:
                init_data = dict(latest_telemetry)
                init_data["waveform"] = stream_processor.get_latest_waveform_snapshot(max_points=500)
            yield f"event: telemetry\ndata: {json.dumps(init_data)}\n\n"

            while True:
                try:
                    msg = client_queue.get(timeout=20.0)
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            with sse_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@telemetry_bp.route("/api/simulation/start", methods=["POST"])
def start_simulation():
    """Isolated simulation mode start trigger (for dev/test validation)."""
    req_json = request.get_json(silent=True) or {}
    scenario = req_json.get("scenario", "normal_sinus")
    simulator.set_scenario(scenario)
    simulator.start()
    return success_response({
        "status": "started",
        "scenario": simulator.get_current_scenario_info()
    })


@telemetry_bp.route("/api/simulation/stop", methods=["POST"])
def stop_simulation():
    simulator.stop()
    return success_response({"status": "stopped"})
