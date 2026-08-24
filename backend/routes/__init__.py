"""HealthSense Routes Package"""
from backend.routes.health_routes import health_bp
from backend.routes.ecg_routes import ecg_bp
from backend.routes.docs_routes import docs_bp
from backend.routes.telemetry_routes import telemetry_bp, simulator
from backend.routes.auth_routes import auth_bp

__all__ = ["health_bp", "ecg_bp", "docs_bp", "telemetry_bp", "auth_bp", "simulator"]
