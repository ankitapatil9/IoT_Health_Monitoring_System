"""
HealthSense - ESP32 Telemetry Simulator & Scenario Generator
Generates realistic multi-sensor telemetry (AD8232 ECG streaming, MAX30102 HR/SpO2, DS18B20 Temp)
and replays real clinical MIT-BIH Arrhythmia benchmark beats for hardware-less testing.
"""

import os
import sys
import time
import math
import random
import threading
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


class TelemetrySimulator:
    """
    Simulates an ESP32 hardware client streaming ECG waveform samples and vital signs.
    """

    SCENARIOS = {
        "normal_sinus": {
            "name": "Normal Sinus Rhythm (Healthy)",
            "description": "Resting normal sinus rhythm with nominal physiological parameters.",
            "target_bpm": 72.0,
            "target_spo2": 98.2,
            "target_temp": 36.8,
            "arrhythmia_prob": 0.0,
            "noise_level": 0.03
        },
        "arrhythmia_surge": {
            "name": "Arrhythmia Surge (PVCs / Ectopic Beats)",
            "description": "Frequent Premature Ventricular Contractions (PVCs) and irregular rhythm.",
            "target_bpm": 86.0,
            "target_spo2": 96.0,
            "target_temp": 36.9,
            "arrhythmia_prob": 0.70,
            "noise_level": 0.05
        },
        "hypoxemia": {
            "name": "Hypoxemia (Critical SpO2 Drop)",
            "description": "Acute respiratory distress with low blood oxygen and compensatory tachycardia.",
            "target_bpm": 108.0,
            "target_spo2": 87.5,
            "target_temp": 36.7,
            "arrhythmia_prob": 0.15,
            "noise_level": 0.04
        },
        "fever_pyrexia": {
            "name": "Severe Pyrexia / Fever Spike",
            "description": "Hyperthermia/infection with elevated temperature and physiological stress.",
            "target_bpm": 114.0,
            "target_spo2": 95.8,
            "target_temp": 38.9,
            "arrhythmia_prob": 0.20,
            "noise_level": 0.04
        },
        "bradycardia": {
            "name": "Severe Bradycardia",
            "description": "Abnormally low heart rate requiring immediate clinical observation.",
            "target_bpm": 43.0,
            "target_spo2": 94.5,
            "target_temp": 36.3,
            "arrhythmia_prob": 0.25,
            "noise_level": 0.03
        }
    }

    def __init__(self, sample_rate: int = 250, on_packet_callback=None):
        self.sample_rate = sample_rate
        self.on_packet_callback = on_packet_callback
        self.scenario_key = "normal_sinus"
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        # Telemetry State
        self.current_bpm = 72.0
        self.current_spo2 = 98.2
        self.current_temp = 36.8

        # Real Clinical Beat Repository from test_data.npz
        self.normal_beats: List[np.ndarray] = []
        self.abnormal_beats: List[np.ndarray] = []
        self._load_clinical_beats()

        # Synthesis index pointers
        self.sample_step = 0
        self.current_beat_buffer = collections_or_array = []
        self.beat_ptr = 0

    def _load_clinical_beats(self):
        """Loads preprocessed MIT-BIH beats from test_data.npz if available."""
        test_npz_path = os.path.join(PROJECT_ROOT, "ml", "data", "test_data.npz")
        if os.path.exists(test_npz_path):
            try:
                npz = np.load(test_npz_path)
                X = npz['X']
                y = npz['y_binary']
                norm_indices = np.where(y == 0)[0]
                abnorm_indices = np.where(y == 1)[0]
                # Keep subset in memory for speed
                self.normal_beats = [X[i].flatten() for i in norm_indices[:500]]
                self.abnormal_beats = [X[i].flatten() for i in abnorm_indices[:500]]
                print(f"[Simulator] Loaded {len(self.normal_beats)} Normal and {len(self.abnormal_beats)} Abnormal test beats.")
            except Exception as e:
                print(f"[Simulator] Error loading test_data.npz: {e}. Falling back to synthetic ECG generator.")
        else:
            print("[Simulator] test_data.npz not found; using pure synthetic ECG generation.")

    def set_scenario(self, scenario_key: str) -> bool:
        if scenario_key in self.SCENARIOS:
            self.scenario_key = scenario_key
            print(f"[Simulator] Switched scenario to: {self.SCENARIOS[scenario_key]['name']}")
            return True
        return False

    def get_current_scenario_info(self) -> Dict[str, Any]:
        info = dict(self.SCENARIOS.get(self.scenario_key, self.SCENARIOS["normal_sinus"]))
        info["key"] = self.scenario_key
        info["is_running"] = self.is_running
        info["current_vitals"] = {
            "heart_rate": round(self.current_bpm, 1),
            "spo2": round(self.current_spo2, 1),
            "temperature": round(self.current_temp, 1)
        }
        return info

    def _generate_synthetic_sinus_beat(self, length: int = 256) -> np.ndarray:
        """Generates standard P-Q-R-S-T wave mathematically if dataset unavailable."""
        t = np.linspace(0, 1, length)
        # Baseline + P wave + Q wave + R wave + S wave + T wave
        p_wave = 0.15 * np.exp(-((t - 0.25) / 0.04) ** 2)
        q_wave = -0.18 * np.exp(-((t - 0.36) / 0.015) ** 2)
        r_wave = 1.20 * np.exp(-((t - 0.39) / 0.02) ** 2)
        s_wave = -0.30 * np.exp(-((t - 0.42) / 0.018) ** 2)
        t_wave = 0.28 * np.exp(-((t - 0.65) / 0.08) ** 2)
        return p_wave + q_wave + r_wave + s_wave + t_wave

    def _generate_synthetic_pvc_beat(self, length: int = 256) -> np.ndarray:
        """Generates wide, bizarre QRS complex mimicking PVC."""
        t = np.linspace(0, 1, length)
        wide_r = 1.35 * np.exp(-((t - 0.37) / 0.06) ** 2)
        inverted_t = -0.55 * np.exp(-((t - 0.60) / 0.10) ** 2)
        return wide_r + inverted_t

    def _next_ecg_sample(self, config: Dict[str, Any]) -> float:
        """Pulls next sample from current active beat or transitions to next beat."""
        if self.beat_ptr >= len(self.current_beat_buffer):
            # Choose next beat (Normal vs Abnormal based on scenario probability)
            arrhythmia_prob = config.get("arrhythmia_prob", 0.0)
            pick_abnormal = random.random() < arrhythmia_prob

            if pick_abnormal:
                if self.abnormal_beats:
                    beat = random.choice(self.abnormal_beats)
                else:
                    beat = self._generate_synthetic_pvc_beat()
            else:
                if self.normal_beats:
                    beat = random.choice(self.normal_beats)
                else:
                    beat = self._generate_synthetic_sinus_beat()

            # Rescale beat according to target heart rate (BPM) interval
            # At 250 Hz, 60 BPM = 250 samples/beat; 120 BPM = 125 samples/beat
            bpm = max(40.0, min(160.0, self.current_bpm))
            samples_per_beat = int((60.0 / bpm) * self.sample_rate)
            
            # Interpolate beat to matching samples_per_beat
            orig_indices = np.linspace(0, 1, len(beat))
            new_indices = np.linspace(0, 1, samples_per_beat)
            scaled_beat = np.interp(new_indices, orig_indices, beat)

            self.current_beat_buffer = list(scaled_beat)
            self.beat_ptr = 0

        val = self.current_beat_buffer[self.beat_ptr]
        self.beat_ptr += 1

        # Add slight physiological baseline noise
        noise = random.gauss(0, config.get("noise_level", 0.02))
        return float(val + noise)

    def _simulation_loop(self):
        """High-precision streaming loop running in background thread."""
        packet_size = 10  # Stream 10 samples every 40 ms (250 Hz)
        interval = packet_size / float(self.sample_rate)  # 0.040s (40 ms)

        while self.is_running:
            start_time = time.time()
            config = self.SCENARIOS.get(self.scenario_key, self.SCENARIOS["normal_sinus"])

            # Smoothly drift vitals toward target
            self.current_bpm += (config["target_bpm"] - self.current_bpm) * 0.08 + random.uniform(-0.3, 0.3)
            self.current_spo2 += (config["target_spo2"] - self.current_spo2) * 0.08 + random.uniform(-0.1, 0.1)
            self.current_temp += (config["target_temp"] - self.current_temp) * 0.08 + random.uniform(-0.02, 0.02)

            # Clamp physical limits
            self.current_spo2 = max(70.0, min(100.0, self.current_spo2))
            self.current_temp = max(34.0, min(42.0, self.current_temp))
            self.current_bpm = max(30.0, min(200.0, self.current_bpm))

            # Generate ECG batch
            ecg_samples = [self._next_ecg_sample(config) for _ in range(packet_size)]

            packet = {
                "device_id": "HEALTHSENSE_ESP32_SIM",
                "timestamp": time.time(),
                "ecg_samples": ecg_samples,
                "heart_rate": round(float(self.current_bpm), 1),
                "spo2": round(float(self.current_spo2), 1),
                "temperature": round(float(self.current_temp), 2),
                "leads_off": False,
                "scenario": self.scenario_key
            }

            if self.on_packet_callback:
                try:
                    self.on_packet_callback(packet)
                except Exception as e:
                    print(f"[Simulator] Callback error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.thread.start()
            print(f"[Simulator] Started telemetry simulation thread on scenario '{self.scenario_key}'.")

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=1.0)
            print("[Simulator] Stopped telemetry simulation thread.")
