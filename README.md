# HealthSense — IoT Multi-Parameter Health Monitoring System

> **Disclaimer**: HealthSense is an experimental research and educational IoT project currently under active development. It is **not** a certified medical device, is **not** clinically validated, and **does not** provide medical diagnosis or treatment advice. All classification labels and risk indicators are preliminary software screening categories only.

---

## 1. Project Objective
HealthSense is designed to collect physiological telemetry from wearable sensors (ECG, pulse oximetry, and body temperature) using an ESP32 microcontroller, stream data to a backend service, and provide real-time visualization and preliminary machine learning-based ECG rhythm screening on a web dashboard.

---

## 2. Hardware Components
The physical hardware setup consists of the following components:

1. **ESP32 Microcontroller** (Wi-Fi & Bluetooth enabled)
2. **AD8232 Single-Lead ECG Module** (with 3-lead electrode cable)
3. **MAX30102 Pulse Oximetry Sensor** (for Heart Rate and SpO2 measurement)
4. **DS18B20 Digital Temperature Sensor** (for body temperature measurement)
5. **0.96-inch I2C OLED Display** (for local device telemetry readout)
6. **ESP32 Base Board / Expansion Board**
7. **18650 Li-ion Battery (2000 mAh)**
8. **TP4056 Battery Charging Module**
9. **3.7V to 5V DC-DC Boost Converter**
10. **Jumper Wires**
11. **USB Data Cable**

---

## 3. System Architecture
```
+-------------------------------------------------------------+
|                     ESP32 Hardware Node                     |
|  - AD8232 (Single-Lead ECG via ADC)                         |
|  - MAX30102 (Heart Rate & SpO2 via I2C)                     |
|  - DS18B20 (Body Temperature via 1-Wire)                    |
|  - 0.96" OLED (Local Sensor Telemetry Display)              |
|  - Wi-Fi Telemetry Stream (JSON / Binary Packets)           |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       Backend Service                       |
|  - Ingestion Gateway (HTTP REST / WebSocket)                |
|  - Sensor Packet Parsing & Buffer Management                |
|  - ML Inference Pipeline Execution                          |
|  - Rule-Based Health Assessment Engine                      |
+---------------+-----------------------------+---------------+
                |                             |
                v                             v
+-------------------------------+   +-------------------------------+
|      ML Screening Engine      |   |   Frontend Web Application    |
|  - Preprocessing & Filtering  |   |  - Real-Time ECG Waveform     |
|  - Beat Window Segmentation   |   |  - Vital Signs Dashboard      |
|  - ML Model Inference         |   |    (BPM, SpO2, Temperature)   |
|  - Classification:            |   |  - Status Indicator:          |
|    NORMAL / ABNORMAL          |   |    NORMAL / OBSERVATION /     |
|                               |   |    HIGH RISK                  |
+-------------------------------+   +-------------------------------+
```

---

## 4. Machine Learning Pipeline
The machine learning pipeline analyzes single-lead ECG signals:

$$\text{ECG Signal} \longrightarrow \text{Preprocessing \& Filtering} \longrightarrow \text{ML Model} \longrightarrow \text{ECG Classification}$$

- **Training Dataset**: MIT-BIH Arrhythmia Database (benchmark single-lead MLII records).
- **Preprocessing**:
  - Noise suppression and baseline wander reduction (digital bandpass / notch filtering).
  - Beat segmentation into fixed-length sample windows centered around detected peaks.
  - Normalization (Z-score standard scaling).
- **Initial ECG Classification Target**:
  - **NORMAL** (Normal sinus rhythm beats)
  - **ABNORMAL** (Arrhythmic or ectopic beats, such as PVCs, PACs, bundle branch blocks)

---

## 5. Multi-Sensor Health Status Assessment (Software Screening)
The software layer combines the ML ECG classification with physiological vital readings to generate a unified screening status on the dashboard:

$$\text{Overall Status} = f(\text{ECG Classification}, \text{Heart Rate}, \text{SpO}_2, \text{Temperature})$$

### Software Screening Categories:
1. **NORMAL**: ECG classified as normal and all vital signs (BPM, SpO2, Temperature) reside within standard nominal ranges.
2. **NEEDS OBSERVATION**: Minor deviations detected (e.g., occasional abnormal beat, slightly elevated temperature, or borderline pulse rate).
3. **HIGH RISK**: Significant rhythm anomaly detected by the ML model or critical sensor threshold breaches (e.g., severe bradycardia/tachycardia, low SpO2).

> *Note*: These statuses are heuristic software screening indicators for research/monitoring purposes only and do not constitute clinical triage or medical diagnosis.

---

## 6. Backend & Frontend Architecture
- **Backend**:
  - Asynchronous web service receiving sensor telemetry streams.
  - ECG buffer management and sliding-window feed into the ML model.
  - Aggregation of multi-sensor readings for real-time dashboard broadcast.
- **Frontend Dashboard**:
  - Live continuous ECG waveform viewer.
  - Numerical telemetry cards for Heart Rate (BPM), Blood Oxygen (SpO2), and Temperature (°C).
  - Software screening status banner (`NORMAL`, `NEEDS OBSERVATION`, `HIGH RISK`).

---

## 7. Planned Development Roadmap
1. **Phase 1: Project Setup & Dataset Inspection** *(Completed)*
   - Workspace directory structure setup.
   - Verification and statistical analysis of MIT-BIH Arrhythmia Database.
2. **Phase 2: ECG Preprocessing & Beat Segmentation**
   - Filtering and noise removal routines.
   - Beat extraction and binary labeling (NORMAL vs. ABNORMAL).
   - Training, validation, and test dataset generation.
3. **Phase 3: Machine Learning Model Development**
   - Model architecture implementation (1D-CNN) and training.
   - Model evaluation (Precision, Recall, F1-Score, Confusion Matrix).
4. **Phase 4: Backend Service & ML Pipeline Integration**
   - Data ingestion endpoints and sliding-window inference pipeline.
   - Multi-sensor health status rule engine.
5. **Phase 5: Web Dashboard Development**
   - Real-time ECG plotting and multi-vital monitoring interface.
6. **Phase 6: ESP32 Hardware & Firmware Integration**
   - Sensor driver integration (AD8232, MAX30102, DS18B20, OLED).
   - Wi-Fi data streaming to backend.
7. **Phase 7: System Validation & Testing**
   - End-to-end telemetry and screening verification.
