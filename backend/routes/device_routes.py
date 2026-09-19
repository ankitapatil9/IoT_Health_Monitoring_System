"""
HealthSense - Device Registration & Real Hardware Telemetry API
Handles device authentication, registration, session management, and recording sessions.
"""

import time
import json
import sqlite3
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify
from backend.services.user_service import get_db_connection
from backend.utils.response_utils import success_response, error_response

device_bp = Blueprint("device_bp", __name__)


@device_bp.route("/api/device/register", methods=["POST"])
def register_device():
    """
    Registers or registers/updates a physical ESP32 device ID in the system.
    """
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    device_name = data.get("device_name", "HealthSense ESP32 Hardware Node")
    mac_address = data.get("mac_address", "")

    if not device_id:
        return error_response("Missing required field 'device_id'", status_code=400)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO devices (device_id, device_name, mac_address, status, last_seen)
                VALUES (?, ?, ?, 'ACTIVE', CURRENT_TIMESTAMP)
                ON CONFLICT(device_id) DO UPDATE SET
                    device_name = excluded.device_name,
                    mac_address = excluded.mac_address,
                    last_seen = CURRENT_TIMESTAMP
            """, (device_id, device_name, mac_address))
            conn.commit()

        return success_response({
            "status": "REGISTERED",
            "device_id": device_id,
            "device_name": device_name,
            "timestamp": time.time()
        })
    except Exception as e:
        return error_response(f"Database error registering device: {str(e)}", status_code=500)


@device_bp.route("/api/device/status", methods=["GET"])
def get_devices_status():
    """
    Lists registered devices and their connectivity status.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT device_id, device_name, mac_address, status, registered_at, last_seen FROM devices")
            rows = cursor.fetchall()
            devices = [dict(r) for r in rows]

        return success_response({"devices": devices, "count": len(devices)})
    except Exception as e:
        return error_response(f"Failed to fetch device status: {str(e)}", status_code=500)


@device_bp.route("/api/device/heartbeat", methods=["POST"])
def device_heartbeat():
    """
    Heartbeat endpoint for connected ESP32 hardware.
    """
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id", "HS-UNKNOWN")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE devices SET last_seen = CURRENT_TIMESTAMP WHERE device_id = ?", (device_id,))
            conn.commit()

        return success_response({"status": "ALIVE", "device_id": device_id, "timestamp": time.time()})
    except Exception as e:
        return error_response(f"Heartbeat failed: {str(e)}", status_code=500)
