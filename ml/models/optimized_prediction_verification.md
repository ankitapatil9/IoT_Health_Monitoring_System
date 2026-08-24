# Optimized ECG Prediction Verification Report

> **Project**: HealthSense — IoT-Based Smart Health Monitoring System  
> **Phase**: Final ECG Prediction Testing & Verification  
> **Evaluation Model**: [ecg_model_optimized.keras](file:///e:/HealthSense/ml/models/ecg_model_optimized.keras)  
> **Inference Engine**: [predict_ecg_optimized.py](file:///e:/HealthSense/ml/predict_ecg_optimized.py)  
> **Verification Date**: 2026-08-24  

---

## 1. Model Used

- **Model Binary**: [ecg_model_optimized.keras](file:///e:/HealthSense/ml/models/ecg_model_optimized.keras)
- **Architecture**: 1D Convolutional Neural Network (3 Conv Blocks with BatchNorm, ReLU, MaxPool, Dropout + GlobalAveragePooling1D + Dense + Sigmoid Output)
- **Total Parameters**: 122,145 (121,121 trainable)
- **Trained Model State**: Completely untouched and unmodified
- **Original Baseline**: [ecg_model.keras](file:///e:/HealthSense/ml/models/ecg_model.keras)

---

## 2. Input Requirements

- **Signal Window Length**: Exactly 256 samples (~711 ms at 360 Hz sampling rate)
- **Tensor Input Shape**: `(1, 256, 1)` or batch `(N, 256, 1)`
- **Data Type**: `float32`
- **Lead / Channel**: Single-lead ECG (Lead MLII / Lead II)

---

## 3. Preprocessing

The preprocessing implemented in [predict_ecg_optimized.py](file:///e:/HealthSense/ml/predict_ecg_optimized.py) strictly matches the training and extraction pipeline in [preprocessing.py](file:///e:/HealthSense/ml/preprocessing.py):
1. **Dimensionality & Shape Validation**: Validates 1D array length == 256 samples.
2. **Quality & Boundary Validation**: Checks and safely rejects `NaN`, `Inf`, empty arrays, and invalid datatypes.
3. **Z-Score Normalization**:
   $$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$
   $$z_i = \frac{x_i - \mu}{\sigma + 10^{-8}}$$
4. **Channel Expansion**: Reshapes normalized 1D vector into `(1, 256, 1)` float32 tensor for 1D-CNN convolution.

---

## 4. Decision Threshold

- **Configured Threshold**: $\tau = 0.30$
- **Calibration Source**: Validation-set F1 grid search on 4 independent validation patients ($9,685$ beats) with zero test-set exposure.
- **Classification Rule**:
  $$\text{Prediction} = \begin{cases} \text{ABNORMAL (Class 1)}, & \text{if } P(\text{Abnormal}) \ge 0.30 \\ \text{NORMAL (Class 0)}, & \text{if } P(\text{Abnormal}) < 0.30 \end{cases}$$
- **Decision Threshold Programmatic Verification**: **PASS** ($\tau = 0.30$)

---

## 5. Normal ECG Prediction

Evaluation on a known normal ECG beat from the independent benchmark test partition:

| Parameter | Value |
| :--- | :--- |
| **Sample Index** | `0` (Patient Record `100`) |
| **True Clinical Label** | `NORMAL` (Class 0, AAMI Class `N`) |
| **Predicted Label** | `NORMAL` |
| **Predicted Class ID** | `0` |
| **Abnormal Probability** | `0.008877` |
| **Prediction Confidence** | `99.11%` (`0.991123`) |
| **Decision Threshold** | `0.30` |
| **Inference Status** | `SUCCESS` |
| **Verification Result** | **PASS** |

---

## 6. Abnormal ECG Prediction

Evaluation on a known abnormal ECG beat (PVC / Ventricular Ectopic Arrhythmia) from the independent benchmark test partition:

| Parameter | Value |
| :--- | :--- |
| **Sample Index** | `6077` (Patient Record `105`) |
| **True Clinical Label** | `ABNORMAL` (Class 1, AAMI Class `V` - PVC) |
| **Predicted Label** | `ABNORMAL` |
| **Predicted Class ID** | `1` |
| **Abnormal Probability** | `0.991265` |
| **Prediction Confidence** | `99.13%` (`0.991265`) |
| **Decision Threshold** | `0.30` |
| **Inference Status** | `SUCCESS` |
| **Verification Result** | **PASS** |

---

## 7. Complete Test Dataset Verification

The optimized predictor was run across all $58,448$ untouched ECG beats from the 26 unseen patient benchmark partition ([test_data.npz](file:///e:/HealthSense/ml/data/test_data.npz)):

| Metric | Verification Value | Target Reference | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Test Beats** | **58,448** | 58,448 | 0 | **MATCH** |
| **Test Accuracy** | **79.02%** (`0.790241`) | 79.02% (`0.790241`) | 0.00% | **PASS** |
| **Precision (PPV)** | **55.03%** (`0.550275`) | 55.03% (`0.550275`) | 0.00% | **PASS** |
| **Recall / Sensitivity** | **57.63%** (`0.576297`) | 57.63% (`0.576297`) | 0.00% | **PASS** |
| **Specificity** | **85.58%** (`0.855760`) | 85.58% (`0.855760`) | 0.00% | **PASS** |
| **F1-Score** | **0.5630** (`0.562986`) | 0.5630 (`0.562986`) | 0.0000 | **PASS** |
| **ROC-AUC** | **0.8056** (`0.805616`) | 0.8056 (`0.805616`) | 0.0000 | **PASS** |

### Confusion Matrix Breakdown

| Metric Cell | Verification Count | Optimization Baseline Reference | Match Result |
| :--- | :---: | :---: | :---: |
| **True Negatives (TN)** | **38,291** | 38,291 | **EXACT MATCH** |
| **False Positives (FP)** | **6,454** | 6,454 | **EXACT MATCH** |
| **False Negatives (FN)** | **5,806** | 5,806 | **EXACT MATCH** |
| **True Positives (TP)** | **7,897** | 7,897 | **EXACT MATCH** |

---

## 8. Error Handling

All edge cases and malformed inputs were tested against [predict_ecg_optimized.py](file:///e:/HealthSense/ml/predict_ecg_optimized.py). The predictor safely validates and rejects invalid input before reaching the model:

| Test Case | Input Tested | Expected Behavior | Actual Behavior | Result |
| :--- | :--- | :--- | :--- | :---: |
| **1. Too few samples** | Array with 255 samples | Reject with `ValueError` | Caught: `Invalid ECG window length: 255 samples. Expected exactly 256 samples.` | **PASS** |
| **2. Too many samples** | Array with 257 samples | Reject with `ValueError` | Caught: `Invalid ECG window length: 257 samples. Expected exactly 256 samples.` | **PASS** |
| **3. Contains NaN** | 256 samples with NaN | Reject with `ValueError` | Caught: `Input ECG window contains NaN values.` | **PASS** |
| **4. Contains Inf** | 256 samples with Inf | Reject with `ValueError` | Caught: `Input ECG window contains Inf values.` | **PASS** |
| **5. Empty input** | `[]` | Reject with `ValueError` | Caught: `Input ECG window is empty. Expected exactly 256 samples.` | **PASS** |
| **6. None input** | `None` | Reject with `ValueError` | Caught: `Input ECG window cannot be None.` | **PASS** |

---

## 9. Preprocessing Consistency

Consistency comparison between [predict_ecg_optimized.py](file:///e:/HealthSense/ml/predict_ecg_optimized.py), [preprocessing.py](file:///e:/HealthSense/ml/preprocessing.py), [preprocessing_config.json](file:///e:/HealthSense/ml/models/preprocessing_config.json), and [model_config.json](file:///e:/HealthSense/ml/models/model_config.json):

- **Input Length**: 256 samples (Window duration ~711 ms at 360 Hz) — **MATCH**
- **Reshaping**: `(1, 256, 1)` float32 — **MATCH**
- **Normalization**: Per-beat Z-Score `(x - mean) / (std + 1e-8)` — **MATCH** (Max absolute difference $< 10^{-6}$)
- **Channel Dimensions**: 1 input channel — **MATCH**
- **Datatype**: `np.float32` — **MATCH**
- **Overall Preprocessing Match**: **PASS**

---

## 10. Comparison With Previously Validated Results

| Metric | Validated Optimized Goal | Predictor Verification Results | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 79.02% | 79.02% | 0.00 pp | **MATCH** |
| **Precision** | 55.03% | 55.03% | 0.00 pp | **MATCH** |
| **Recall / Sensitivity** | 57.63% | 57.63% | 0.00 pp | **MATCH** |
| **Specificity** | 85.58% | 85.58% | 0.00 pp | **MATCH** |
| **F1-Score** | 0.5630 | 0.5630 | 0.0000 | **MATCH** |
| **ROC-AUC** | 0.8056 | 0.8056 | 0.0000 | **MATCH** |
| **Threshold** | 0.30 | 0.30 | 0.00 | **MATCH** |

There is **zero discrepancy** between the optimized training results in [optimization_results.json](file:///e:/HealthSense/ml/models/optimization_results.json) and the standalone inference engine in [predict_ecg_optimized.py](file:///e:/HealthSense/ml/predict_ecg_optimized.py).

---

## 11. Final Verification Status

```
================================================================================
FINAL VERIFICATION STATUS: PASS
================================================================================
MODEL MODIFICATION:             NOT PERFORMED
RETRAINING:                     NOT PERFORMED
THRESHOLD MODIFICATION:         NOT PERFORMED
TEST DATA USED FOR TRAINING:    NO
================================================================================
```
