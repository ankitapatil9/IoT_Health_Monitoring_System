# HealthSense — Backend Service Architecture

## 1. Overview
The HealthSense backend is an asynchronous, high-performance physiological telemetry gateway and real-time machine learning inference server.

### Key Capabilities:
1. **Sensor Telemetry Ingestion (`POST /api/telemetry`)**: Receives high-frequency multi-sensor JSON packets from physical ESP32 devices or edge nodes.
2. **Sliding-Window Inference Engine (`ecg_stream_processor.py`)**: Manages continuous ECG circular FIFO ring buffers (250 Hz), digital Butterworth bandpass filtering, adaptive R-peak detection, and automatic 256-sample window extraction for the trained 1D-CNN Arrhythmia classification model.
3. **Multi-Parameter Health Rule Engine (`health_engine.py`)**: Synthesizes vital sign parameters (Heart Rate, SpO2, Body Temp) with the deep learning ECG classification to generate unified screening status (`NORMAL`, `NEEDS OBSERVATION`, `HIGH RISK`).
4. **Real-Time Streaming Gateway (`GET /api/stream`)**: Server-Sent Events (SSE) broadcast of synchronized waveforms and vitals to the frontend web dashboard.
5. **Hardware Telemetry Simulator (`simulator.py`)**: Replays clinical MIT-BIH benchmark test records and simulates synchronized vital sign scenarios for hardware-less testing.

---

## 2. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web dashboard application |
| `GET` | `/api/status` | System health, model state, and throughput counters |
| `GET` | `/api/latest` | Latest snapshot of physiological readings and AI status |
| `GET` | `/api/stream` | Server-Sent Events (SSE) live telemetry stream |
| `POST` | `/api/telemetry` | Ingestion endpoint for ESP32 hardware packets |
| `POST` | `/api/simulation/start` | Starts the background MIT-BIH telemetry simulator |
| `POST` | `/api/simulation/stop` | Stops the telemetry simulator |
| `POST` | `/api/simulation/scenario`| Switches scenario (`normal_sinus`, `arrhythmia_surge`, `hypoxemia`, `fever_pyrexia`, `bradycardia`) |

---

## 3. Running the Server

```bash
python backend/app.py
```
Default URL: `http://localhost:5000`
