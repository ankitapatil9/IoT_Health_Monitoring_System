# HealthSense — ECG Preprocessing & Training Dataset Report

> **Dataset Source**: MIT-BIH Arrhythmia Database (`dataset/mitdb`)  
> **Generation Date**: 2026-08-22 00:35:42  
> **Sampling Frequency**: 360.0 Hz  
> **Beat Window Length**: 256 samples (~711.1 ms, -100 to +156 samples around peak)  
> **Normalization**: Beat-level Z-score standard scaling  
> **Partitioning Strategy**: Strict Inter-Patient / Inter-Recording Division (zero patient overlap)

---

## 1. ECG Channel Selection & Filtering Pipeline

### Channel Selection
For each record, the pipeline automatically detects and prioritizes the **Modified Lead II (MLII)** channel when present. 
- In **45 of 48 records**, Lead MLII is used.
- In records where MLII is not present (e.g., Records 102, 104 with `V5, V2`), Lead 0 (`V5`) is utilized as the primary continuous single-lead channel.

### Digital Signal Filtering
To remove physiological and environmental noise without introducing phase distortion, a zero-phase digital filter is applied:
1. **Bandpass Filtering**: 3rd order zero-phase Butterworth bandpass filter with cutoff frequencies:
   - **Low Cutoff**: $0.5\text{ Hz}$ (eliminates respiratory baseline wander and motion artifacts)
   - **High Cutoff**: $45.0\text{ Hz}$ (suppresses high-frequency EMG muscle noise and 50/60 Hz powerline interference)
2. **Beat Normalization**: Every extracted beat segment $x$ is normalized using Z-score standardization:
   $$\hat{x} = \frac{x - \mu}{\sigma + 10^{-8}}$$

---

## 2. AAMI EC57 Beat Annotation Mapping & Exclusions

In accordance with clinical standards (AAMI EC57), ECG annotations are categorized into 5 major beat classes, and subsequently mapped to the primary **NORMAL** vs. **ABNORMAL** screening targets.

### Beat Class Mapping (5-Class & Binary)

| Target Class | AAMI Category | Included MIT-BIH Annotation Symbols | Description |
| :--- | :---: | :--- | :--- |
| **NORMAL (0)** | **N** | `N`, `L`, `R`, `e`, `j`, `B` | Normal beat, Left/Right bundle branch block, Atrial/Nodal escape |
| **ABNORMAL (1)** | **S** | `A`, `a`, `J`, `S` | Atrial premature beat, Aberrated APB, Nodal premature, Supraventricular premature |
| **ABNORMAL (1)** | **V** | `V`, `E` | Premature ventricular contraction (PVC), Ventricular escape |
| **ABNORMAL (1)** | **F** | `F` | Fusion of ventricular and normal beat |
| **ABNORMAL (1)** | **Q** | `/`, `f`, `Q`, `?` | Paced beat, Fusion of paced & normal, Unclassifiable beat |

### Excluded Non-Beat Annotation Markers
Non-beat markers (rhythm changes, noise artifacts, and comments) were systematically filtered out and **not** included as beat segments:

| Excluded Symbol | Total Count in Dataset | Reason for Exclusion |
| :---: | :---: | :--- |
| `+` | 1,291 | Rhythm change indicator (non-beat timing marker) |
| `~` | 616 | Signal quality / noise artifact change |
| `!` | 472 | Ventricular flutter wave |
| `"` | 437 | Annotator comment |
| `x` | 193 | Non-conducted P-wave (blocked APC) |
| `\|` | 132 | Isolated QRS-like artifact |
| `[` | 6 | Start of ventricular flutter/fibrillation |
| `]` | 6 | End of ventricular flutter/fibrillation |
| **Total Excluded** | **3,153** | **Non-beat clinical markers safely excluded** |

---

## 3. Inter-Patient Dataset Splits

To prevent data leakage and overly optimistic performance estimates, beats from the same recording/patient are **never** shared across splits.

- **Training Partition (18 records)**: `101, 106, 108, 109, 112, 114, 116, 118, 119, 124, 201, 203, 207, 208, 209, 215, 220, 230`
- **Validation Partition (4 records)**: `115, 122, 205, 223`
- **Test Benchmark Partition (26 records)**: `100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234, 102, 104, 107, 217`

### Split Statistics Summary

| Dataset Partition | Recordings | Total Beats | Input Shape $(N, L, 1)$ | Invalid Boundary Beats Discarded |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 18 | 41,315 | `(41315, 256, 1)` | 16 |
| **Validation Set** | 4 | 9,685 | `(9685, 256, 1)` | 5 |
| **Test Set** | 26 | 58,448 | `(58448, 256, 1)` | 25 |
| **Total Database** | **48** | **109,448** | — | **46** |

---

## 4. Class Distribution

### Binary Screening Targets (NORMAL vs. ABNORMAL)

| Split | NORMAL Beats (Class 0) | ABNORMAL Beats (Class 1) | % Normal | % Abnormal | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36,806 | 4,509 | 89.09% | 10.91% | 41,315 |
| **Validation** | 9,040 | 645 | 93.34% | 6.66% | 9,685 |
| **Test** | 44,745 | 13,703 | 76.56% | 23.44% | 58,448 |
| **Combined** | **90,591** | **18,857** | **82.77%** | **17.23%** | **109,448** |

### Detailed 5-Class AAMI Distribution

| Class | Name | Train Count | Val Count | Test Count | Total Count | Overall % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **N** | Normal | 36,806 | 9,040 | 44,745 | 90,591 | 82.77% |
| **S** | Supraventricular | 868 | 76 | 1,837 | 2,781 | 2.54% |
| **V** | Ventricular | 3,244 | 544 | 3,447 | 7,235 | 6.61% |
| **F** | Fusion | 389 | 25 | 388 | 802 | 0.73% |
| **Q** | Paced / Unknown | 8 | 0 | 8,031 | 8,039 | 7.35% |

---

## 5. Class Imbalance Strategy & Recommended Weights

The ECG beat distribution exhibits significant natural clinical imbalance (Normal beats represent ~82.8% of all beats).

### Recommended Handling Strategies:
1. **Cost-Sensitive Loss (Weighted Loss Function)** *(Recommended)*:
   Apply inverse class frequency weighting during training loss computation to penalize misclassification of minority arrhythmic beats without introducing synthetic distortion.
2. **Balanced Inverse Class Weights Computed from Training Set**:
   - **Binary Classification (NORMAL : ABNORMAL)**:
     - Weight for Class 0 (`NORMAL`): `0.5613`
     - Weight for Class 1 (`ABNORMAL`): `4.5814`
   - **5-Class Classification (N, S, V, F, Q)**:
     - Weight for Class `N`: `0.2245`
     - Weight for Class `S`: `9.5196`
     - Weight for Class `V`: `2.5472`
     - Weight for Class `F`: `21.2416`
     - Weight for Class `Q`: `1032.8750`
3. **Focal Loss / Balanced Batch Sampling**:
   Can be used optionally during 1D-CNN optimization to focus gradients on hard-to-classify borderline morphological beats.

---

## 6. Processed Data Artifacts

The generated data partitions and configuration files are saved at:
- Training Data: `ml/data/train_data.npz`
- Validation Data: `ml/data/val_data.npz`
- Test Data: `ml/data/test_data.npz`
- Preprocessing Config: `ml/models/preprocessing_config.json`
- Dataset Metadata: `ml/data/dataset_metadata.json`
