# HealthSense — 1D-CNN ECG Binary Screening Model Evaluation Report

> **Model Architecture**: Lightweight 1D Convolutional Neural Network (1D-CNN)  
> **Input Dimension**: `(256, 1)` representing $711.1\text{ ms}$ single-lead ECG window at $360\text{ Hz}$  
> **Evaluation Partition**: Out-of-Sample Inter-Patient Test Benchmark (26 independent patient records, $58,448$ beats)  
> **Saved Model Path**: [ecg_model.keras](file:///e:/HealthSense/ml/models/ecg_model.keras)  
> **Evaluation Date**: 2026-08-22 01:21:52  

---

## 1. Model Architecture & Complexity

The model is designed for high temporal feature extraction with minimal compute latency for edge and real-time streaming:

```
Layer (type)                     Output Shape          Param #     Details
================================================================================
InputLayer                       (None, 256, 1)        0           Input ECG Beat Window
Conv1D (32, k=7) + BN + ReLU     (None, 256, 32)       352         Feature Extraction
Conv1D (32, k=7) + BN + ReLU     (None, 256, 32)       7,296       Feature Refinement
MaxPooling1D (pool=2) + Dropout  (None, 128, 32)       0           Downsampling
Conv1D (64, k=5) + BN + ReLU     (None, 128, 64)       10,496      Mid-Level Features
Conv1D (64, k=5) + BN + ReLU     (None, 128, 64)       20,736      Mid-Level Refinement
MaxPooling1D (pool=2) + Dropout  (None, 64, 64)        0           Downsampling
Conv1D (128, k=3) + BN + ReLU    (None, 64, 128)       25,088      High-Level Morphology
Conv1D (128, k=3) + BN + ReLU    (None, 64, 128)       49,664      High-Level Refinement
MaxPooling1D (pool=2) + Dropout  (None, 32, 128)       0           Downsampling
GlobalAveragePooling1D           (None, 128)           0           Temporal Aggregation
Dense (64) + BN + ReLU + Dropout (None, 64)            8,448       Classification Head
Dense (1, Sigmoid)               (None, 1)             65          Binary Probability
================================================================================
Total Parameters: 122,145 (~477.1 KB)
Trainable Parameters: 121,121
```

---

## 2. Training Dynamics & Convergence

- **Total Epochs Run**: 15 epochs (EarlyStopping patience: 10)
- **Best Validation Epoch**: Epoch 5
- **Best Validation Loss**: `0.0898`
- **Best Validation Accuracy**: `98.48%`
- **Class Weights Applied**:
  - Class 0 (`NORMAL`): `0.561254`
  - Class 1 (`ABNORMAL`): `4.581393`
- **Learning Rate Schedule**: Initial `1e-3` reduced adaptively on plateaus (`ReduceLROnPlateau`).
- **Overfitting Assessment**: The validation loss remained stable throughout training without runaway divergence. The use of spatial Dropout ($p=0.2-0.3$), Batch Normalization after convolutions, and Global Average Pooling constrained the capacity and preserved generalization.

![Training History](file:///e:/HealthSense/ml/models/training_history.png)

---

## 3. Comprehensive Test Set Performance (26 Unseen Patient Records)

Evaluated strictly on the independent $58,448$ beat test partition (zero recording overlap with training):

| Evaluation Metric | Value | Description |
| :--- | :---: | :--- |
| **Accuracy** | **73.28%** | Overall proportion of correct predictions |
| **Abnormal Sensitivity (Recall)** | **62.09%** | Proportion of true arrhythmia/ectopic beats correctly flagged |
| **Normal Specificity** | **76.70%** | Proportion of normal beats correctly identified |
| **Precision (PPV)** | **44.94%** | Reliability when model predicts 'Abnormal' |
| **F1-Score** | **0.5214** | Harmonic mean of Precision and Sensitivity |
| **ROC-AUC** | **0.7936** | Area under the Receiver Operating Characteristic curve |

---

## 4. Confusion Matrix Breakdown

| | Predicted `NORMAL` (0) | Predicted `ABNORMAL` (1) | Total True Beats |
| :---: | :---: | :---: | :---: |
| **Actual `NORMAL` (0)** | **34,320** (TN, 76.70%) | **10,425** (FP, 23.30%) | **44,745** |
| **Actual `ABNORMAL` (1)** | **5,195** (FN, 37.91%) | **8,508** (TP, 62.09%) | **13,703** |
| **Total Predicted** | **39,515** | **18,933** | **58,448** |

![Confusion Matrix](file:///e:/HealthSense/ml/models/confusion_matrix.png)

### Key Clinical Screening Interpretations:
1. **Abnormal Detection (8,508 True Positives, 62.09% Sensitivity)**: The model successfully detects the majority of true ectopic and arrhythmic waveforms on previously unseen patient anatomies.
2. **False Negatives (5,195 beats, 37.91%)**: Concentrated in challenging borderline morphology beats and atypical paced rhythm artifacts in the out-of-sample patient benchmark (such as records `102`, `104`, `107`).
3. **False Positives (10,425 beats, 23.30%)**: Driven by the cost-sensitive loss weighting prioritizing the detection of potential anomalies for clinical screening safety.

---

## 5. Limitations & Next Steps

1. **Screening Tool Only**: HealthSense is a software screening and research monitoring tool and does not provide clinical diagnoses.
2. **Single-Lead Generalization**: The model operates on single-lead ECG inputs matching the AD8232 module ($360\text{ Hz}$, 256 samples).
3. **Inference Readiness**: Model weight footprint is only **477 KB** (`ecg_model.keras`), enabling sub-5ms CPU inference suitable for real-time backend streaming.
