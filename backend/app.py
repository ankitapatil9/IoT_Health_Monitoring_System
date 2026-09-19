"""
HealthSense - Backend Application Server
Entry point routing to modular architecture in backend/main.py.
Production mode strictly ingests real physical sensor data.
"""

import os
import sys

# Ensure project root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app, create_app

if __name__ == "__main__":
    print("=" * 80)
    print("HealthSense Physical IoT Backend Server Starting...")
    print("API Endpoint      : http://127.0.0.1:5000")
    print("Telemetry Ingest  : http://127.0.0.1:5000/api/telemetry")
    print("Interactive Docs  : http://127.0.0.1:5000/docs")
    print("Health Check      : http://127.0.0.1:5000/api/health")
    print("ECG Predict       : http://127.0.0.1:5000/api/ecg/predict")
    print("Report Generator  : http://127.0.0.1:5000/api/report/generate")
    print("=" * 80)
    print("[SERVER MODE] Real Hardware Telemetry Ingestion Active (No Auto-Simulation).")
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
