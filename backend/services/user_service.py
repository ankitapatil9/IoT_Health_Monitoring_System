"""
HealthSense - User Database & Authentication Service
Provides secure local SQLite persistence, password hashing (Werkzeug),
and unique Patient ID generation.
"""

import os
import sqlite3
import re
from typing import Dict, Any, Optional, Tuple
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "healthsense.db")

# Simple email regex pattern
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db() -> None:
    """
    Initializes the SQLite database table for users and pre-seeds the demo account.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                patient_id TEXT UNIQUE NOT NULL,
                phone TEXT,
                age INTEGER,
                gender TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Check if any users exist; if not, seed default demo user
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        if count == 0:
            demo_hash = generate_password_hash("healthsense2026")
            cursor.execute("""
                INSERT INTO users (full_name, email, password_hash, patient_id, phone, age, gender)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "Prajkta Patil",
                "prajkta.patil@healthsense.org",
                demo_hash,
                "PNT-2026-0001",
                "+91 98765 43210",
                24,
                "Female"
            ))
            conn.commit()


def generate_unique_patient_id(conn: sqlite3.Connection) -> str:
    """
    Generates sequential, unique patient ID such as PNT-2026-0001, PNT-2026-0002.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    next_num = count + 1
    return f"PNT-2026-{next_num:04d}"


class UserService:
    """
    Authentication & User Management Service.
    """

    @staticmethod
    def validate_registration_input(
        full_name: str,
        email: str,
        password: str,
        confirm_password: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates registration fields and returns (is_valid, error_message).
        """
        if not full_name or not full_name.strip():
            return False, "Please enter your full name."

        if len(full_name.strip()) < 2:
            return False, "Full name must be at least 2 characters long."

        if not email or not email.strip():
            return False, "Please enter your email address."

        if not EMAIL_REGEX.match(email.strip()):
            return False, "Please enter a valid email address."

        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters long."

        if confirm_password is not None and password != confirm_password:
            return False, "Passwords do not match."

        return True, None

    @classmethod
    def register_user(
        cls,
        full_name: str,
        email: str,
        password: str,
        confirm_password: Optional[str] = None,
        phone: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Registers a new user into SQLite with hashed password.
        Returns: (success: bool, user_dict: Optional[Dict], error: Optional[str])
        """
        clean_name = full_name.strip() if full_name else ""
        clean_email = email.strip().lower() if email else ""

        is_valid, err_msg = cls.validate_registration_input(clean_name, clean_email, password, confirm_password)
        if not is_valid:
            return False, None, err_msg

        pw_hash = generate_password_hash(password)

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                # Check for existing email
                cursor.execute("SELECT id FROM users WHERE email = ?", (clean_email,))
                if cursor.fetchone():
                    return False, None, "An account with this email already exists."

                patient_id = generate_unique_patient_id(conn)

                cursor.execute("""
                    INSERT INTO users (full_name, email, password_hash, patient_id, phone, age, gender)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    clean_name,
                    clean_email,
                    pw_hash,
                    patient_id,
                    phone.strip() if phone else None,
                    int(age) if age else None,
                    gender.strip() if gender else None
                ))
                conn.commit()
                user_id = cursor.lastrowid

                safe_user = {
                    "id": user_id,
                    "full_name": clean_name,
                    "email": clean_email,
                    "patient_id": patient_id,
                    "phone": phone,
                    "age": age,
                    "gender": gender
                }
                return True, safe_user, None

        except sqlite3.IntegrityError:
            return False, None, "An account with this email already exists."
        except Exception as e:
            return False, None, f"Failed to create account: {str(e)}"

    @classmethod
    def authenticate_user(cls, email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Authenticates a user with email and password.
        Returns: (success: bool, user_dict: Optional[Dict], error: Optional[str])
        """
        if not email or not email.strip() or not password:
            return False, None, "Invalid email or password."

        clean_email = email.strip().lower()

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, full_name, email, password_hash, patient_id, phone, age, gender
                    FROM users WHERE email = ?
                """, (clean_email,))
                row = cursor.fetchone()

                if not row:
                    return False, None, "Invalid email or password."

                if not check_password_hash(row["password_hash"], password):
                    return False, None, "Invalid email or password."

                safe_user = {
                    "id": row["id"],
                    "full_name": row["full_name"],
                    "email": row["email"],
                    "patient_id": row["patient_id"],
                    "phone": row["phone"],
                    "age": row["age"],
                    "gender": row["gender"]
                }
                return True, safe_user, None

        except Exception:
            return False, None, "Authentication service error."

    @classmethod
    def get_user_by_id(cls, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches safe user dictionary by user ID.
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, full_name, email, patient_id, phone, age, gender
                    FROM users WHERE id = ?
                """, (user_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "full_name": row["full_name"],
                        "email": row["email"],
                        "patient_id": row["patient_id"],
                        "phone": row["phone"],
                        "age": row["age"],
                        "gender": row["gender"]
                    }
                return None
        except Exception:
            return None


# Initialize database automatically upon import
init_user_db()
