"""
HealthSense - Authentication & Registration Automated Test Suite
Tests registration, validation, SQLite storage, Werkzeug password hashing,
session creation, login, logout, me endpoint, and ECG regression.
"""

import sys
import os
import json
import unittest

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import create_app
from backend.services.user_service import UserService, get_db_connection


class TestAuthenticationSystem(unittest.TestCase):
    """Automated unit and integration tests for HealthSense Authentication."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def setUp(self):
        # Create unique test email prefix for each test
        self.test_email = f"test_{os.urandom(4).hex()}@healthsense.org"
        self.test_password = "SecurePassword2026!"
        self.test_name = "HealthSense Demo User"

    def test_01_index_html_serves_auth_elements(self):
        """Test that index.html serves both Login and Register structures."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")
        self.assertIn("login-card", html)
        self.assertIn("register-card", html)
        self.assertIn("Create Your HealthSense Account", html)
        self.assertIn("Welcome Back", html)
        print("\n[AUTH TEST 01] Login & Registration HTML delivery: PASS")

    def test_02_valid_registration(self):
        """Test valid user registration with auto-generated Patient ID."""
        payload = {
            "full_name": self.test_name,
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": self.test_password,
            "phone": "+91 99999 88888",
            "age": 28,
            "gender": "Female"
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["user"]["email"], self.test_email.lower())
        self.assertEqual(data["user"]["full_name"], self.test_name)
        self.assertTrue(data["user"]["patient_id"].startswith("PNT-2026-"))
        self.assertNotIn("password", data["user"])
        self.assertNotIn("password_hash", data["user"])
        print(f"[AUTH TEST 02] Valid Registration ({self.test_email}): PASS -> ID: {data['user']['patient_id']}")

    def test_03_duplicate_email_rejection(self):
        """Test rejection of duplicate email registration."""
        payload = {
            "full_name": "Duplicate User",
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": self.test_password
        }
        # First registration
        self.client.post("/api/auth/register", json=payload)
        # Duplicate registration attempt
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("already exists", data["error"])
        print("[AUTH TEST 03] Duplicate Email Rejection: PASS")

    def test_04_empty_name_rejection(self):
        """Test rejection of empty full name."""
        payload = {
            "full_name": "   ",
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": self.test_password
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("full name", data["error"])
        print("[AUTH TEST 04] Empty Name Rejection: PASS")

    def test_05_invalid_email_rejection(self):
        """Test rejection of malformed email address."""
        payload = {
            "full_name": self.test_name,
            "email": "not-a-valid-email",
            "password": self.test_password,
            "confirm_password": self.test_password
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("valid email", data["error"])
        print("[AUTH TEST 05] Invalid Email Rejection: PASS")

    def test_06_weak_password_rejection(self):
        """Test rejection of short password (less than 6 characters)."""
        payload = {
            "full_name": self.test_name,
            "email": self.test_email,
            "password": "123",
            "confirm_password": "123"
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("at least 6 characters", data["error"])
        print("[AUTH TEST 06] Weak Password Rejection: PASS")

    def test_07_password_mismatch_rejection(self):
        """Test rejection of non-matching passwords."""
        payload = {
            "full_name": self.test_name,
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": "DifferentPassword123"
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("Passwords do not match", data["error"])
        print("[AUTH TEST 07] Password Mismatch Rejection: PASS")

    def test_08_successful_login_and_session(self):
        """Test successful login with registered credentials and session establishment."""
        # 1. Register
        self.client.post("/api/auth/register", json={
            "full_name": self.test_name,
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": self.test_password
        })

        # 2. Login
        res = self.client.post("/api/auth/login", json={
            "email": self.test_email,
            "password": self.test_password
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["user"]["full_name"], self.test_name)
        self.assertNotIn("password", data["user"])

        # 3. Check /api/auth/me session
        me_res = self.client.get("/api/auth/me")
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.get_json()
        self.assertTrue(me_data["authenticated"])
        self.assertEqual(me_data["user"]["email"], self.test_email.lower())
        print(f"[AUTH TEST 08] Successful Login & Authenticated Session: PASS ({self.test_email})")

    def test_09_incorrect_password_rejected(self):
        """Test rejection of incorrect password."""
        # Register
        self.client.post("/api/auth/register", json={
            "full_name": self.test_name,
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": self.test_password
        })

        # Attempt login with wrong password
        res = self.client.post("/api/auth/login", json={
            "email": self.test_email,
            "password": "WrongPassword123!"
        })
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertEqual(data["error"], "Invalid email or password.")
        print("[AUTH TEST 09] Incorrect Password Rejection: PASS")

    def test_10_unknown_account_rejected(self):
        """Test rejection of non-existent email account."""
        res = self.client.post("/api/auth/login", json={
            "email": "nonexistent.user.2026@healthsense.org",
            "password": "AnyPassword123"
        })
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertEqual(data["error"], "Invalid email or password.")
        print("[AUTH TEST 10] Unknown Account Rejection: PASS")

    def test_11_logout_terminates_session(self):
        """Test that logout clears session and /api/auth/me returns authenticated: false."""
        # Login with demo user
        self.client.post("/api/auth/login", json={
            "email": "prajkta.patil@healthsense.org",
            "password": "healthsense2026"
        })

        # Verify logged in
        me_before = self.client.get("/api/auth/me").get_json()
        self.assertTrue(me_before["authenticated"])

        # Logout
        logout_res = self.client.post("/api/auth/logout")
        self.assertEqual(logout_res.status_code, 200)

        # Verify logged out
        me_after = self.client.get("/api/auth/me").get_json()
        self.assertFalse(me_after["authenticated"])
        print("[AUTH TEST 11] Logout & Session Termination: PASS")

    def test_12_default_demo_user_functional(self):
        """Test that pre-seeded default demo user logs in smoothly."""
        res = self.client.post("/api/auth/login", json={
            "email": "prajkta.patil@healthsense.org",
            "password": "healthsense2026"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["user"]["full_name"], "Prajkta Patil")
        print("[AUTH TEST 12] Default Demo User Authentication: PASS")

    def test_13_ecg_regression_normal_beat(self):
        """Regression test: verify Normal ECG inference works exactly as before."""
        res_sample = self.client.get("/api/ecg/sample-beats")
        samples = res_sample.get_json()["normal_beat"]["samples"]

        pred_res = self.client.post("/api/ecg/predict", json={"ecg": samples})
        self.assertEqual(pred_res.status_code, 200)
        data = pred_res.get_json()
        self.assertEqual(data["prediction"], "NORMAL")
        self.assertEqual(data["class_id"], 0)
        self.assertEqual(data["threshold"], 0.30)
        self.assertAlmostEqual(data["abnormal_probability"], 0.008877, places=4)
        print("[AUTH TEST 13] ECG Regression (Normal Beat): PASS -> NORMAL (0.89%)")

    def test_14_ecg_regression_abnormal_beat(self):
        """Regression test: verify Abnormal ECG inference works exactly as before."""
        res_sample = self.client.get("/api/ecg/sample-beats")
        samples = res_sample.get_json()["abnormal_beat"]["samples"]

        pred_res = self.client.post("/api/ecg/predict", json={"ecg": samples})
        self.assertEqual(pred_res.status_code, 200)
        data = pred_res.get_json()
        self.assertEqual(data["prediction"], "ABNORMAL")
        self.assertEqual(data["class_id"], 1)
        self.assertEqual(data["threshold"], 0.30)
        self.assertAlmostEqual(data["abnormal_probability"], 0.991265, places=4)
        print("[AUTH TEST 14] ECG Regression (Abnormal Beat): PASS -> ABNORMAL (99.13%)")


if __name__ == "__main__":
    unittest.main()
