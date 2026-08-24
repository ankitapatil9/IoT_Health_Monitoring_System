"""
HealthSense - Core Backend Application Entry Point
Modular Flask API with CORS, Authentication, OpenAPI Swagger Documentation, and Verified ML Integration.
"""

import os
import sys
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.routes.health_routes import health_bp
from backend.routes.ecg_routes import ecg_bp
from backend.routes.docs_routes import docs_bp
from backend.routes.telemetry_routes import telemetry_bp, simulator
from backend.routes.auth_routes import auth_bp
from backend.services.user_service import init_user_db


def create_app() -> Flask:
    """Application factory for HealthSense API server."""
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
    
    # Configure secure session secret key
    app.secret_key = os.environ.get("HEALTHSENSE_SECRET_KEY", "healthsense-secure-session-key-2026-medical-portal")
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Initialize SQLite user database
    init_user_db()

    # Configure CORS for local development and dashboard access
    CORS(app, supports_credentials=True, resources={
        r"/api/*": {
            "origins": ["http://localhost:*", "http://127.0.0.1:*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Register Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(ecg_bp)
    app.register_blueprint(docs_bp)
    app.register_blueprint(telemetry_bp)
    app.register_blueprint(auth_bp)

    # Static frontend routing
    @app.route("/", methods=["GET"])
    def index():
        if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
            return send_from_directory(FRONTEND_DIR, "index.html")
        return jsonify({
            "service": "HealthSense ECG AI & Telemetry API",
            "status": "ONLINE",
            "docs": "/docs",
            "health": "/api/health",
            "predict": "/api/ecg/predict",
            "auth": "/api/auth/me"
        })

    @app.route("/<path:path>", methods=["GET"])
    def static_files(path):
        if os.path.exists(os.path.join(FRONTEND_DIR, path)):
            return send_from_directory(FRONTEND_DIR, path)
        return jsonify({"error": f"File '{path}' not found.", "status": "ERROR"}), 404

    # Global 404 & 500 error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "error": "Endpoint or resource not found.",
            "error_code": "NOT_FOUND",
            "status": "ERROR",
            "docs": "/docs"
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({
            "error": "HTTP method not allowed for this endpoint.",
            "error_code": "METHOD_NOT_ALLOWED",
            "status": "ERROR"
        }), 405

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({
            "error": "Internal server error occurred.",
            "error_code": "INTERNAL_ERROR",
            "status": "ERROR"
        }), 500

    return app


app = create_app()


if __name__ == "__main__":
    print("=" * 80)
    print("HealthSense Backend Server Starting...")
    print("API Endpoint      : http://127.0.0.1:5000")
    print("Interactive Docs  : http://127.0.0.1:5000/docs")
    print("Health Check      : http://127.0.0.1:5000/api/health")
    print("ECG Predict       : http://127.0.0.1:5000/api/ecg/predict")
    print("Auth Register     : http://127.0.0.1:5000/api/auth/register")
    print("Auth Login        : http://127.0.0.1:5000/api/auth/login")
    print("=" * 80)
    
    # Auto-start simulator in background
    simulator.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
