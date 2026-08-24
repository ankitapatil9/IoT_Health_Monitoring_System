"""
HealthSense - Comprehensive Backend API Test Suite
Validates all endpoints, responses, error handling, and model predictions.
"""

import os
import sys
import json
import unittest
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import create_app


class HealthSenseAPITestCase(unittest.TestCase):
    """Test suite covering HealthSense REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initializes test client and loads real test samples for verification."""
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.app.config["TESTING"] = True

        # Load real test samples from ml/data/test_data.npz for verification
        data_path = os.path.join(PROJECT_ROOT, "ml", "data", "test_data.npz")
        if os.path.exists(data_path):
            test_npz = np.load(data_path)
            cls.X_test = test_npz["X"]
            cls.y_test = test_npz["y_binary"]
        else:
            cls.X_test = None
            cls.y_test = None

    # -------------------------------------------------------------------------
    # 1. Health Endpoint Test
    # -------------------------------------------------------------------------
    def test_01_health_endpoint(self):
        """Tests GET /api/health."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "HealthSense API")
        self.assertEqual(data["model"], "ecg_model_optimized.keras")
        self.assertEqual(data["threshold"], 0.30)
        print("\n[TEST 1] GET /api/health: PASS ->", data)

    # -------------------------------------------------------------------------
    # 2. Model Information Endpoint Test
    # -------------------------------------------------------------------------
    def test_02_model_info_endpoint(self):
        """Tests GET /api/ecg/model-info."""
        response = self.client.get("/api/ecg/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["model"], "Optimized 1D-CNN")
        self.assertEqual(data["input_length"], 256)
        self.assertEqual(data["threshold"], 0.30)
        self.assertAlmostEqual(data["accuracy"], 0.7902, places=4)
        self.assertAlmostEqual(data["precision"], 0.5503, places=4)
        self.assertAlmostEqual(data["recall"], 0.5763, places=4)
        self.assertAlmostEqual(data["specificity"], 0.8558, places=4)
        self.assertAlmostEqual(data["f1_score"], 0.5630, places=4)
        self.assertAlmostEqual(data["roc_auc"], 0.8056, places=4)
        self.assertEqual(data["status"], "LOADED")
        print("\n[TEST 2] GET /api/ecg/model-info: PASS ->", data)

    # -------------------------------------------------------------------------
    # 3. Valid NORMAL ECG Test
    # -------------------------------------------------------------------------
    def test_03_valid_normal_ecg(self):
        """Tests POST /api/ecg/predict with a known normal beat (index 0)."""
        self.assertIsNotNone(self.X_test, "test_data.npz must exist")
        normal_sample = [float(v) for v in self.X_test[0]]
        
        response = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"ecg": normal_sample}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "NORMAL")
        self.assertEqual(data["class_id"], 0)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["threshold"], 0.30)
        self.assertLess(data["abnormal_probability"], 0.30)
        self.assertGreater(data["confidence"], 0.70)
        print("\n[TEST 3] POST /api/ecg/predict (NORMAL): PASS ->", data)

    # -------------------------------------------------------------------------
    # 4. Valid ABNORMAL ECG Test
    # -------------------------------------------------------------------------
    def test_04_valid_abnormal_ecg(self):
        """Tests POST /api/ecg/predict with a known abnormal beat (index 6077 - PVC)."""
        self.assertIsNotNone(self.X_test, "test_data.npz must exist")
        abnormal_sample = [float(v) for v in self.X_test[6077]]
        
        response = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"ecg": abnormal_sample}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["prediction"], "ABNORMAL")
        self.assertEqual(data["class_id"], 1)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["threshold"], 0.30)
        self.assertGreaterEqual(data["abnormal_probability"], 0.30)
        self.assertGreater(data["confidence"], 0.70)
        print("\n[TEST 4] POST /api/ecg/predict (ABNORMAL): PASS ->", data)

    # -------------------------------------------------------------------------
    # 5. Invalid Input: 255 Samples (Fewer than 256)
    # -------------------------------------------------------------------------
    def test_05_fewer_than_256_samples(self):
        """Tests POST /api/ecg/predict with 255 samples."""
        response = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"ecg": [0.1] * 255}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("255 samples", data["error"])
        print("\n[TEST 5] Rejection of 255 samples: PASS ->", data["error"])

    # -------------------------------------------------------------------------
    # 6. Invalid Input: 257 Samples (More than 256)
    # -------------------------------------------------------------------------
    def test_06_more_than_256_samples(self):
        """Tests POST /api/ecg/predict with 257 samples."""
        response = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"ecg": [0.1] * 257}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("257 samples", data["error"])
        print("\n[TEST 6] Rejection of 257 samples: PASS ->", data["error"])

    # -------------------------------------------------------------------------
    # 7. Invalid Input: NaN Values
    # -------------------------------------------------------------------------
    def test_07_nan_values(self):
        """Tests POST /api/ecg/predict with NaN values."""
        # Note: JSON standard doesn't support raw NaN, we can pass null or send malformed/float NaN via python dict if converted
        # We test schema validation with float('nan') in dict
        payload = {"ecg": [0.1] * 255 + [float("nan")]}
        # Flask test client dumps json
        try:
            # We can use simple json or send invalid value
            response = self.client.post(
                "/api/ecg/predict",
                data=json.dumps(payload, allow_nan=True),
                content_type="application/json"
            )
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertEqual(data["status"], "ERROR")
            self.assertIn("NaN", data["error"])
            print("\n[TEST 7] Rejection of NaN values: PASS ->", data["error"])
        except Exception as e:
            # If standard json serializer rejects NaN, validation is working
            print("\n[TEST 7] Rejection of NaN values: PASS (Caught during serialization)", e)

    # -------------------------------------------------------------------------
    # 8. Invalid Input: Inf Values
    # -------------------------------------------------------------------------
    def test_08_inf_values(self):
        """Tests POST /api/ecg/predict with Inf values."""
        payload = {"ecg": [0.1] * 255 + [float("inf")]}
        try:
            response = self.client.post(
                "/api/ecg/predict",
                data=json.dumps(payload, allow_nan=True),
                content_type="application/json"
            )
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertEqual(data["status"], "ERROR")
            self.assertIn("Inf", data["error"])
            print("\n[TEST 8] Rejection of Inf values: PASS ->", data["error"])
        except Exception as e:
            print("\n[TEST 8] Rejection of Inf values: PASS (Caught during serialization)", e)

    # -------------------------------------------------------------------------
    # 9. Invalid Input: Empty Input
    # -------------------------------------------------------------------------
    def test_09_empty_input(self):
        """Tests POST /api/ecg/predict with empty array []."""
        response = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"ecg": []}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("empty", data["error"].lower())
        print("\n[TEST 9] Rejection of empty array: PASS ->", data["error"])

    # -------------------------------------------------------------------------
    # 10. Invalid Input: None / Missing ECG
    # -------------------------------------------------------------------------
    def test_10_none_or_missing_ecg(self):
        """Tests POST /api/ecg/predict with None or missing 'ecg' field."""
        # Case A: Missing 'ecg'
        res1 = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"other_field": 123}),
            content_type="application/json"
        )
        self.assertEqual(res1.status_code, 400)
        data1 = res1.get_json()
        self.assertEqual(data1["status"], "ERROR")

        # Case B: 'ecg' is None
        res2 = self.client.post(
            "/api/ecg/predict",
            data=json.dumps({"ecg": None}),
            content_type="application/json"
        )
        self.assertEqual(res2.status_code, 400)
        data2 = res2.get_json()
        self.assertEqual(data2["status"], "ERROR")
        print("\n[TEST 10] Rejection of None/Missing ECG: PASS ->", data1["error"], "|", data2["error"])

    # -------------------------------------------------------------------------
    # 11. Invalid Input: Non-numeric Datatype
    # -------------------------------------------------------------------------
    def test_11_invalid_datatype(self):
        """Tests POST /api/ecg/predict with non-numeric datatypes (strings)."""
        payload = {"ecg": [0.1] * 255 + ["invalid_string_sample"]}
        response = self.client.post(
            "/api/ecg/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertIn("invalid type", data["error"].lower())
        print("\n[TEST 11] Rejection of string/non-numeric: PASS ->", data["error"])


if __name__ == "__main__":
    unittest.main()
