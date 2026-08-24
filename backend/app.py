"""
HealthSense - Backend Application Server
Entry point routing to modular architecture in backend/main.py.
"""

from backend.main import app, create_app

if __name__ == "__main__":
    from backend.routes.telemetry_routes import simulator
    print("=" * 80)
    print("HealthSense Backend Server Starting (via app.py)...")
    print("API Endpoint      : http://127.0.0.1:5000")
    print("Interactive Docs  : http://127.0.0.1:5000/docs")
    print("Health Check      : http://127.0.0.1:5000/api/health")
    print("ECG Predict       : http://127.0.0.1:5000/api/ecg/predict")
    print("=" * 80)
    simulator.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
