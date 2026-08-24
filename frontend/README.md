# HealthSense — Real-Time Web Dashboard

## 1. Overview
The HealthSense Frontend is a dark-theme, medical-grade real-time physiological monitoring dashboard built with Vanilla HTML5, CSS3, and JavaScript.

### Core Features:
- **60 FPS Canvas Oscilloscope**: Smooth, glowing neon cyan single-lead ECG trace with calibrated medical grid (5mm major / 1mm minor lines), phosphor sweep, and adjustable gain multipliers (0.75x, 1.0x, 1.5x).
- **Deep Learning Arrhythmia Classifier**: Real-time 1D-CNN classification banner, live abnormal risk percentage progress bar, model confidence gauge, and recent arrhythmia burden calculator.
- **Dynamic Multi-Vital Cards**:
  - Heart Rate (BPM) with animated cardiac pulse.
  - Blood Oxygen ($SpO_2$) with clinical range guidance.
  - Body Temperature ($^\circ C$) with fever thresholds.
- **Multi-Parameter Clinical Screening Banner**: Synthesizes combined status (`NORMAL`, `NEEDS OBSERVATION`, `HIGH RISK`) with active diagnostic finding chips.
- **Simulation Control Panel**: Instant switching between clinical scenarios (`Normal Sinus`, `Arrhythmia Surge / PVCs`, `Hypoxemia`, `Pyrexia / Fever`, `Severe Bradycardia`) and pause/resume stream toggle.
- **Live Activity Feed**: Real-time chronological audit trail of rhythm anomalies and threshold breaches.
