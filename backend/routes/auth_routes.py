"""
HealthSense - Authentication & User Registration Routes
Provides secure session-based authentication endpoints:
- POST /api/auth/register (User registration)
- POST /api/auth/login (Credential authentication & signed session creation)
- POST /api/auth/logout (Session termination)
- GET /api/auth/me (Current user session state)
"""

from flask import Blueprint, request, session, jsonify
from backend.services.user_service import UserService
from backend.utils.response_utils import success_response, error_response

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    """
    POST /api/auth/register
    Creates a new user account with secure password hashing and unique Patient ID.
    
    Request:
    {
        "full_name": "John Doe",
        "email": "john@example.com",
        "password": "example-password",
        "confirm_password": "example-password",
        "phone": "+91 98765 43210",
        "age": 30,
        "gender": "Male"
    }

    Response (201 Created):
    {
        "status": "SUCCESS",
        "message": "Account created successfully",
        "user": {
            "full_name": "John Doe",
            "email": "john@example.com",
            "patient_id": "PNT-2026-0002"
        }
    }
    """
    if not request.is_json:
        return error_response("Content-Type must be application/json", error_code="INVALID_CONTENT_TYPE", status_code=400)

    data = request.get_json(silent=True)
    if not data:
        return error_response("Empty registration payload.", error_code="VALIDATION_ERROR", status_code=400)

    full_name = data.get("full_name")
    email = data.get("email")
    password = data.get("password")
    confirm_password = data.get("confirm_password")
    phone = data.get("phone")
    age = data.get("age")
    gender = data.get("gender")

    success, user, err_msg = UserService.register_user(
        full_name=full_name,
        email=email,
        password=password,
        confirm_password=confirm_password,
        phone=phone,
        age=age,
        gender=gender
    )

    if not success or user is None:
        return error_response(err_msg or "Registration failed.", error_code="REGISTRATION_ERROR", status_code=400)

    return success_response({
        "status": "SUCCESS",
        "message": "Account created successfully",
        "user": {
            "full_name": user["full_name"],
            "email": user["email"],
            "patient_id": user["patient_id"],
            "phone": user.get("phone"),
            "age": user.get("age"),
            "gender": user.get("gender")
        }
    }, status_code=201)


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """
    POST /api/auth/login
    Authenticates email and password, creates a cryptographically signed Flask session.

    Request:
    {
        "email": "john@example.com",
        "password": "example-password"
    }

    Response (200 OK):
    {
        "status": "SUCCESS",
        "message": "Login successful",
        "user": {
            "full_name": "John Doe",
            "email": "john@example.com",
            "patient_id": "PNT-2026-0002"
        }
    }
    """
    if not request.is_json:
        return error_response("Content-Type must be application/json", error_code="INVALID_CONTENT_TYPE", status_code=400)

    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return error_response("Please enter both email and password.", error_code="INVALID_CREDENTIALS", status_code=401)

    success, user, err_msg = UserService.authenticate_user(email, password)
    if not success or user is None:
        return error_response("Invalid email or password.", error_code="INVALID_CREDENTIALS", status_code=401)

    # Establish cryptographically signed session (Never store password in session!)
    session.clear()
    session["user_id"] = user["id"]
    session["email"] = user["email"]
    session["full_name"] = user["full_name"]
    session["patient_id"] = user["patient_id"]

    return success_response({
        "status": "SUCCESS",
        "message": "Login successful",
        "user": {
            "full_name": user["full_name"],
            "email": user["email"],
            "patient_id": user["patient_id"],
            "phone": user.get("phone"),
            "age": user.get("age"),
            "gender": user.get("gender")
        }
    }, status_code=200)


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    """
    POST /api/auth/logout
    Clears active user session.
    """
    session.clear()
    return success_response({
        "status": "SUCCESS",
        "message": "Logged out successfully"
    }, status_code=200)


@auth_bp.route("/api/auth/me", methods=["GET"])
def get_current_user():
    """
    GET /api/auth/me
    Returns current authenticated session user information.
    """
    user_id = session.get("user_id")
    if not user_id:
        return success_response({
            "authenticated": False
        }, status_code=200)

    user = UserService.get_user_by_id(user_id)
    if not user:
        session.clear()
        return success_response({
            "authenticated": False
        }, status_code=200)

    return success_response({
        "authenticated": True,
        "user": {
            "full_name": user["full_name"],
            "email": user["email"],
            "patient_id": user["patient_id"],
            "phone": user.get("phone"),
            "age": user.get("age"),
            "gender": user.get("gender")
        }
    }, status_code=200)
