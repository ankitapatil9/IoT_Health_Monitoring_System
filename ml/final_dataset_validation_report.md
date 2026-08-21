# HealthSense — Final Dataset Pre-Training Validation Report

> **Validation Date**: 2026-08-22 00:46:24  
> **Model Target**: Binary ECG Arrhythmia Screening (`0 = NORMAL`, `1 = ABNORMAL`)  
> **Validation Status**: **PASSED ALL INTEGRITY & LEAKAGE CHECKS**

---

## 1. Dataset Tensor Dimensions & Compatibility

| Partition | X Tensor Shape | y Label Shape | Sample Length (Samples) | Duration at 360 Hz | Input Compatible? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | `(41315, 256)` | `(41315,)` | 256 | 711.1 ms | **YES** |
| **Validation** | `(9685, 256)` | `(9685,)` | 256 | 711.1 ms | **YES** |
| **Test** | `(58448, 256)` | `(58448,)` | 256 | 711.1 ms | **YES** |

- **Sample Length Verification**: Confirmed that every single beat window across all partitions has exactly **256 samples** (100 samples pre-peak, 156 samples post-peak).
- **Batching & Conv1D Readiness**: All three tensors are fully shape-compatible for 1D-CNN input `(Batch, 256, 1)`.

---

## 2. Binary Class Distribution & Label Integrity

| Partition | Class 0 (`NORMAL`) | Class 1 (`ABNORMAL`) | % Normal | % Abnormal | Total Beats | Label Set Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36,806 | 4,509 | 89.09% | 10.91% | 41,315 | Strictly `{0, 1}` |
| **Validation** | 9,040 | 645 | 93.34% | 6.66% | 9,685 | Strictly `{0, 1}` |
| **Test** | 44,745 | 13,703 | 76.56% | 23.44% | 58,448 | Strictly `{0, 1}` |
| **Total** | **90,591** | **18,857** | **82.77%** | **17.23%** | **109,448** | **VERIFIED** |

---

## 3. Data Quality & Numerical Integrity Checks

Every single float value across all partitions was checked for `NaN`, positive/negative infinity, and numerical anomalies:

| Partition | Total Float Elements | NaN Count | Inf Count | Minimum | Maximum | Mean | Std Dev | Data Quality Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 10,576,640 | 0 | 0 | -6.6241 | 7.1832 | -5.4461e-11 | 1.0000 | **CLEAN (0 NaNs / 0 Infs)** |
| **Validation** | 2,479,360 | 0 | 0 | -5.1483 | 6.9638 | -1.6155e-11 | 1.0000 | **CLEAN (0 NaNs / 0 Infs)** |
| **Test** | 14,962,688 | 0 | 0 | -5.6498 | 7.4746 | 5.0989e-12 | 1.0000 | **CLEAN (0 NaNs / 0 Infs)** |

- **Z-Score Normalization Confirmation**: The global mean across segments is $pprox 0.0$ and global standard deviation is $pprox 1.0$, confirming successful per-beat normalization without dynamic range collapse.

---

## 4. Patient Isolation & Data Leakage Verification

### Patient Overlap Check:
- **Training Patient Records (18)**: `101, 106, 108, 109, 112, 114, 116, 118, 119, 124, 201, 203, 207, 208, 209, 215, 220, 230`
- **Validation Patient Records (4)**: `115, 122, 205, 223`
- **Test Patient Records (26)**: `100, 102, 103, 104, 105, 107, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233, 234`

| Check | Expected | Actual Result | Leakage Risk? |
| :--- | :---: | :---: | :---: |
| Train $\cap$ Validation overlap | 0 records | `[]` | **NO (Zero Overlap)** |
| Train $\cap$ Test overlap | 0 records | `[]` | **NO (Zero Overlap)** |
| Validation $\cap$ Test overlap | 0 records | `[]` | **NO (Zero Overlap)** |
| Total Unique Patient Records | 48 records | 48 records | **COMPLETE COVERAGE** |

- **Data Leakage Conclusion**: There is **zero patient data leakage**. The test set serves as a true out-of-sample inter-patient benchmark.

---

## 5. Original MIT-BIH Dataset Integrity Check

Verified that no files in `dataset/mitdb` were deleted, renamed, or modified:
- **`.dat` signal files present**: 48 (Expected: 48)
- **`.hea` header files present**: 48 (Expected: 48)
- **`.atr` annotation files present**: 49 (Expected: 49)
- **Dataset Modification Status**: **UNTOUCHED & FULLY PRESERVED**

---

## 6. Preprocessing Config & Real-Time Sensor Parity

Inspected [ml/models/preprocessing_config.json](file:///e:/HealthSense/ml/models/preprocessing_config.json):
- **Sampling Frequency ($f_s$)**: `360.0 Hz`
- **Bandpass Filter Settings**: `0.5 Hz - 45.0 Hz` (Order: `3`)
- **Window Length ($L$)**: `256 samples` (Pre-peak: `100`, Post-peak: `156`)
- **Normalization Algorithm**: `z_score`

> **Hardware / Live Streaming Alignment**: When streaming real-time ECG from the ESP32 + AD8232 module at 360 Hz (or resampled to 360 Hz), applying the stored bandpass filter parameters and peak-window Z-score scaler will produce identical input distributions for live inference.

---

## 7. Class Weight Verification

Re-calculated balanced inverse class frequency weights from training labels:
$$w_c = \frac{N_{\text{train}}}{2 \cdot N_c}$$

- **Calculated Weight for Class 0 (`NORMAL`)**: `0.561254` (Config: `0.561254`) $\rightarrow$ **MATCH**
- **Calculated Weight for Class 1 (`ABNORMAL`)**: `4.581393` (Config: `4.581393`) $\rightarrow$ **MATCH**

---

## 8. Summary of Findings & Readiness Assessment

1. **Problems Found**: **NONE**.
2. **Data Leakage**: **NONE**.
3. **Data Quality Issues (NaN/Inf/Shape)**: **NONE**.
4. **Recommended Next Step**: Ready to proceed with designing the 1D-CNN binary classification model architecture, defining the weighted loss function, and commencing training.
