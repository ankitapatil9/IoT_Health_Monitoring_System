"""
HealthSense - Real-Time Telemetry & Simulation Routes
Handles physical sensor packet ingestion, Server-Sent Events (SSE) streaming,
and simulator controls.
"""

import time
import json
import queue
import threading
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, Response
from backend.services.stream_processor import ECGStreamProcessor
from backend.health_engine import HealthAssessmentEngine
from backend.simulator import TelemetrySimulator
from backend.utils.response_utils import success_response, error_response

telemetry_bp = Blueprint("telemetry_bp", __name__)

# State & Processing Engines
health_engine = HealthAssessmentEngine()
stream_processor = ECGStreamProcessor(buffer_size=1500, sampling_rate=250)

# Global Telemetry Cache
latest_telemetry: Dict[str, Any] = {
    "device_id": "HEALTHSENSE_NODE_01",
    "timestamp": time.time(),
    "heart_rate": 72.0,
    "spo2": 98.0,
    "temperature": 36.8,
    "leads_off": False,
    "ml_prediction": stream_processor.last_beat_prediction,
    "assessment": health_engine.assess_health(72.0, 98.0, 36.8, stream_processor.last_beat_prediction),
    "screening_summary": stream_processor.get_screening_summary(),
    "source": "INITIALIZING"
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
    Central processing pipeline for both physical ESP32 and simulator packets.
    """
    global latest_telemetry

    ecg_samples = packet.get("ecg_samples", [])
    hr = packet.get("heart_rate")
    spo2 = packet.get("spo2")
    temp = packet.get("temperature")
    leads_off = packet.get("leads_off", False)
    device_id = packet.get("device_id", "HEALTHSENSE_NODE")

    # Ingest ECG samples into sliding window buffer
    if ecg_samples:
        stream_processor.ingest_samples(ecg_samples)

    # Trigger beat detection & ML inference
    ml_result = stream_processor.detect_and_classify_beats()
    if ml_result is None:
        ml_result = stream_processor.last_beat_prediction

    # Multi-parameter assessment
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
        "ecg_points": ecg_samples,
        "waveform": waveform_snapshot,
        "ml_prediction": ml_result,
        "assessment": assessment,
        "screening_summary": screening_summary
    }
    broadcast_sse_event("telemetry", broadcast_data)


# Instantiate Simulator
simulator = TelemetrySimulator(sample_rate=250, on_packet_callback=process_incoming_packet)


@telemetry_bp.route("/api/telemetry", methods=["POST"])
def ingest_telemetry():
    """Ingestion endpoint for physical ESP32 or external sources."""
    if not request.is_json:
        return error_response("Content-Type must be application/json", status_code=400)

    data = request.get_json(silent=True)
    if not data:
        return error_response("Empty payload", status_code=400)

    process_incoming_packet(data)
    return success_response({"status": "received", "timestamp": time.time()})


@telemetry_bp.route("/api/latest", methods=["GET"])
def get_latest_telemetry():
    """Returns the most recent physiological readings, ML predictions, and assessment."""
    with telemetry_lock:
        snapshot = dict(latest_telemetry)
        snapshot["waveform"] = stream_processor.get_latest_waveform_snapshot(max_points=500)
    return success_response(snapshot)


@telemetry_bp.route("/api/stream", methods=["GET"])
def stream_sse():
    """Server-Sent Events (SSE) continuous stream for the real-time frontend dashboard."""
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


@telemetry_bp.route("/api/simulation/scenario", methods=["POST"])
def switch_simulation_scenario():
    req_json = request.get_json(silent=True) or {}
    scenario = req_json.get("scenario", "")
    if simulator.set_scenario(scenario):
        return success_response({
            "status": "success",
            "scenario": simulator.get_current_scenario_info()
        })
    return error_response(f"Invalid scenario '{scenario}'. Available: {list(simulator.SCENARIOS.keys())}", status_code=400)


@telemetry_bp.route("/api/simulation/status", methods=["GET"])
def simulation_status():
    return success_response({
        "is_running": simulator.is_running,
        "current_scenario": simulator.get_current_scenario_info(),
        "available_scenarios": simulator.SCENARIOS
    })
