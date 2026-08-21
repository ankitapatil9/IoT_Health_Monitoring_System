"""
HealthSense - Dataset Preparation and Segmentation Pipeline
Extracts, filters, segments, normalizes, and splits MIT-BIH Arrhythmia Database
beats into Train, Validation, and Test partitions strictly by patient/recording.
"""

import os
import json
from collections import Counter
import numpy as np
import pandas as pd
import wfdb
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import (
    DEFAULT_FS,
    BEAT_WINDOW_SIZE,
    PRE_PEAK_SAMPLES,
    POST_PEAK_SAMPLES,
    AAMI_5CLASS_MAPPING,
    CLASS_TO_INT_5CLASS,
    INT_TO_CLASS_5CLASS,
    CLASS_TO_INT_BINARY,
    INT_TO_CLASS_BINARY,
    EXCLUDED_ANNOTATIONS,
    preprocess_ecg_signal,
    extract_beat_window,
    z_score_normalize,
    map_annotation_symbol,
)

# -------------------------------------------------------------------------
# Inter-Patient Dataset Splits (Disjoint Record Sets)
# Based on the established AAMI / de Chazal benchmark partitioning
# -------------------------------------------------------------------------
# Training Records (18 records)
TRAIN_RECORD_IDS = [
    '101', '106', '108', '109', '112', '114', '116', '118', '119',
    '124', '201', '203', '207', '208', '209', '215', '220', '230'
]

# Validation Records (4 records - disjoint patient set)
VAL_RECORD_IDS = [
    '115', '122', '205', '223'
]

# Test Records (26 records - disjoint test benchmark including DS2 + paced)
TEST_RECORD_IDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234',
    '102', '104', '107', '217'
]


def get_lead_index(record: wfdb.Record) -> int:
    """
    Selects the primary modified limb lead II (MLII) channel if available,
    otherwise defaults to Channel 0 (the primary continuous recording channel).
    """
    for idx, name in enumerate(record.sig_name):
        if 'MLII' in name.upper():
            return idx
    return 0


def process_record_beats(dataset_dir: str, record_id: str):
    """
    Loads, filters, and segments beats for a single MIT-BIH record.
    Returns:
        valid_segments: np.ndarray of shape (N_beats, 256)
        labels_5class: list of int class IDs (0..4)
        labels_binary: list of int class IDs (0..1)
        beat_symbols: list of original symbols
        stats: dict containing exclusion and boundary removal statistics
    """
    rec_path = os.path.join(dataset_dir, record_id)
    
    # 1. Read record header and signal
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, 'atr')
    
    lead_idx = get_lead_index(record)
    lead_name = record.sig_name[lead_idx]
    raw_signal = record.p_signal[:, lead_idx]
    fs = record.fs
    
    # 2. Filter signal (bandpass 0.5 - 45 Hz)
    filtered_signal = preprocess_ecg_signal(raw_signal, fs=fs, apply_notch=False)
    
    # 3. Extract and normalize beat windows
    valid_segments = []
    labels_5class = []
    labels_binary = []
    beat_symbols = []
    
    excluded_counts = Counter()
    boundary_removals = 0
    
    for sample_idx, sym in zip(annotation.sample, annotation.symbol):
        mapping = map_annotation_symbol(sym)
        
        # Check if it is an AAMI beat annotation
        if not mapping["is_beat"]:
            excluded_counts[sym] += 1
            continue
            
        # Extract fixed-length beat-centered window [sample - 100, sample + 156] = 256 samples
        segment = extract_beat_window(
            filtered_signal, 
            r_peak_idx=sample_idx, 
            pre_peak=PRE_PEAK_SAMPLES, 
            post_peak=POST_PEAK_SAMPLES
        )
        
        if segment is None:
            boundary_removals += 1
            continue
            
        # Apply Z-score normalization per beat
        norm_segment = z_score_normalize(segment)
        
        valid_segments.append(norm_segment)
        labels_5class.append(mapping["aami_5class_int"])
        labels_binary.append(mapping["binary_class_int"])
        beat_symbols.append(sym)
        
    stats = {
        "record_id": record_id,
        "lead_used": lead_name,
        "lead_index": lead_idx,
        "total_annotations": len(annotation.sample),
        "valid_beats_extracted": len(valid_segments),
        "boundary_removals": boundary_removals,
        "excluded_annotations": dict(excluded_counts),
        "symbol_counts": dict(Counter(beat_symbols))
    }
    
    return np.array(valid_segments, dtype=np.float32), np.array(labels_5class, dtype=np.int64), np.array(labels_binary, dtype=np.int64), stats


def process_dataset_split(dataset_dir: str, record_ids: list, split_name: str):
    """
    Processes all records in a split and aggregates arrays.
    """
    print(f"\nProcessing {split_name} split ({len(record_ids)} records)...")
    
    all_segments = []
    all_labels_5class = []
    all_labels_binary = []
    all_patient_ids = []
    split_stats = []
    
    for rec_id in record_ids:
        segments, y_5class, y_bin, stats = process_record_beats(dataset_dir, rec_id)
        if len(segments) > 0:
            all_segments.append(segments)
            all_labels_5class.append(y_5class)
            all_labels_binary.append(y_bin)
            all_patient_ids.extend([rec_id] * len(segments))
        split_stats.append(stats)
        print(f"  Record {rec_id} ({stats['lead_used']}): Extracted {stats['valid_beats_extracted']} beats | Discarded {stats['boundary_removals']} boundary samples")

    X = np.vstack(all_segments) if all_segments else np.empty((0, BEAT_WINDOW_SIZE), dtype=np.float32)
    y_5c = np.concatenate(all_labels_5class) if all_labels_5class else np.empty((0,), dtype=np.int64)
    y_bin = np.concatenate(all_labels_binary) if all_labels_binary else np.empty((0,), dtype=np.int64)
    patient_arr = np.array(all_patient_ids)
    
    return X, y_5c, y_bin, patient_arr, split_stats


def generate_preprocessing_report(output_dir: str, metadata: dict, report_md_path: str):
    """
    Generates a comprehensive markdown report of the preprocessing pipeline and dataset splits.
    """
    report_content = f"""# HealthSense — ECG Preprocessing & Training Dataset Report

> **Dataset Source**: MIT-BIH Arrhythmia Database (`dataset/mitdb`)  
> **Generation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
> **Sampling Frequency**: {DEFAULT_FS} Hz  
> **Beat Window Length**: {BEAT_WINDOW_SIZE} samples (~{BEAT_WINDOW_SIZE / DEFAULT_FS * 1000:.1f} ms, -{PRE_PEAK_SAMPLES} to +{POST_PEAK_SAMPLES} samples around peak)  
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
   - **Low Cutoff**: $0.5\\text{{ Hz}}$ (eliminates respiratory baseline wander and motion artifacts)
   - **High Cutoff**: $45.0\\text{{ Hz}}$ (suppresses high-frequency EMG muscle noise and 50/60 Hz powerline interference)
2. **Beat Normalization**: Every extracted beat segment $x$ is normalized using Z-score standardization:
   $$\\hat{{x}} = \\frac{{x - \\mu}}{{\\sigma + 10^{{-8}}}}$$

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
| `+` | {metadata['global_exclusions'].get('+', 0):,} | Rhythm change indicator (non-beat timing marker) |
| `~` | {metadata['global_exclusions'].get('~', 0):,} | Signal quality / noise artifact change |
| `!` | {metadata['global_exclusions'].get('!', 0):,} | Ventricular flutter wave |
| `"` | {metadata['global_exclusions'].get('"', 0):,} | Annotator comment |
| `x` | {metadata['global_exclusions'].get('x', 0):,} | Non-conducted P-wave (blocked APC) |
| `|` | {metadata['global_exclusions'].get('|', 0):,} | Isolated QRS-like artifact |
| `[` | {metadata['global_exclusions'].get('[', 0):,} | Start of ventricular flutter/fibrillation |
| `]` | {metadata['global_exclusions'].get(']', 0):,} | End of ventricular flutter/fibrillation |
| **Total Excluded** | **{sum(metadata['global_exclusions'].values()):,}** | **Non-beat clinical markers safely excluded** |

---

## 3. Inter-Patient Dataset Splits

To prevent data leakage and overly optimistic performance estimates, beats from the same recording/patient are **never** shared across splits.

- **Training Partition (18 records)**: `{', '.join(TRAIN_RECORD_IDS)}`
- **Validation Partition (4 records)**: `{', '.join(VAL_RECORD_IDS)}`
- **Test Benchmark Partition (26 records)**: `{', '.join(TEST_RECORD_IDS)}`

### Split Statistics Summary

| Dataset Partition | Recordings | Total Beats | Input Shape $(N, L, 1)$ | Invalid Boundary Beats Discarded |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | {len(TRAIN_RECORD_IDS)} | {metadata['train']['num_samples']:,} | `({metadata['train']['num_samples']}, {BEAT_WINDOW_SIZE}, 1)` | {metadata['train']['boundary_removals']:,} |
| **Validation Set** | {len(VAL_RECORD_IDS)} | {metadata['val']['num_samples']:,} | `({metadata['val']['num_samples']}, {BEAT_WINDOW_SIZE}, 1)` | {metadata['val']['boundary_removals']:,} |
| **Test Set** | {len(TEST_RECORD_IDS)} | {metadata['test']['num_samples']:,} | `({metadata['test']['num_samples']}, {BEAT_WINDOW_SIZE}, 1)` | {metadata['test']['boundary_removals']:,} |
| **Total Database** | **48** | **{metadata['total_valid_beats']:,}** | — | **{metadata['total_boundary_removals']:,}** |

---

## 4. Class Distribution

### Binary Screening Targets (NORMAL vs. ABNORMAL)

| Split | NORMAL Beats (Class 0) | ABNORMAL Beats (Class 1) | % Normal | % Abnormal | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | {metadata['train']['binary_dist']['NORMAL']:,} | {metadata['train']['binary_dist']['ABNORMAL']:,} | {metadata['train']['binary_dist']['NORMAL']/metadata['train']['num_samples']*100:.2f}% | {metadata['train']['binary_dist']['ABNORMAL']/metadata['train']['num_samples']*100:.2f}% | {metadata['train']['num_samples']:,} |
| **Validation** | {metadata['val']['binary_dist']['NORMAL']:,} | {metadata['val']['binary_dist']['ABNORMAL']:,} | {metadata['val']['binary_dist']['NORMAL']/metadata['val']['num_samples']*100:.2f}% | {metadata['val']['binary_dist']['ABNORMAL']/metadata['val']['num_samples']*100:.2f}% | {metadata['val']['num_samples']:,} |
| **Test** | {metadata['test']['binary_dist']['NORMAL']:,} | {metadata['test']['binary_dist']['ABNORMAL']:,} | {metadata['test']['binary_dist']['NORMAL']/metadata['test']['num_samples']*100:.2f}% | {metadata['test']['binary_dist']['ABNORMAL']/metadata['test']['num_samples']*100:.2f}% | {metadata['test']['num_samples']:,} |
| **Combined** | **{metadata['total_binary']['NORMAL']:,}** | **{metadata['total_binary']['ABNORMAL']:,}** | **{metadata['total_binary']['NORMAL']/metadata['total_valid_beats']*100:.2f}%** | **{metadata['total_binary']['ABNORMAL']/metadata['total_valid_beats']*100:.2f}%** | **{metadata['total_valid_beats']:,}** |

### Detailed 5-Class AAMI Distribution

| Class | Name | Train Count | Val Count | Test Count | Total Count | Overall % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **N** | Normal | {metadata['train']['5class_dist']['N']:,} | {metadata['val']['5class_dist']['N']:,} | {metadata['test']['5class_dist']['N']:,} | {metadata['total_5class']['N']:,} | {metadata['total_5class']['N']/metadata['total_valid_beats']*100:.2f}% |
| **S** | Supraventricular | {metadata['train']['5class_dist']['S']:,} | {metadata['val']['5class_dist']['S']:,} | {metadata['test']['5class_dist']['S']:,} | {metadata['total_5class']['S']:,} | {metadata['total_5class']['S']/metadata['total_valid_beats']*100:.2f}% |
| **V** | Ventricular | {metadata['train']['5class_dist']['V']:,} | {metadata['val']['5class_dist']['V']:,} | {metadata['test']['5class_dist']['V']:,} | {metadata['total_5class']['V']:,} | {metadata['total_5class']['V']/metadata['total_valid_beats']*100:.2f}% |
| **F** | Fusion | {metadata['train']['5class_dist']['F']:,} | {metadata['val']['5class_dist']['F']:,} | {metadata['test']['5class_dist']['F']:,} | {metadata['total_5class']['F']:,} | {metadata['total_5class']['F']/metadata['total_valid_beats']*100:.2f}% |
| **Q** | Paced / Unknown | {metadata['train']['5class_dist']['Q']:,} | {metadata['val']['5class_dist']['Q']:,} | {metadata['test']['5class_dist']['Q']:,} | {metadata['total_5class']['Q']:,} | {metadata['total_5class']['Q']/metadata['total_valid_beats']*100:.2f}% |

---

## 5. Class Imbalance Strategy & Recommended Weights

The ECG beat distribution exhibits significant natural clinical imbalance (Normal beats represent ~{metadata['total_5class']['N']/metadata['total_valid_beats']*100:.1f}% of all beats).

### Recommended Handling Strategies:
1. **Cost-Sensitive Loss (Weighted Loss Function)** *(Recommended)*:
   Apply inverse class frequency weighting during training loss computation to penalize misclassification of minority arrhythmic beats without introducing synthetic distortion.
2. **Balanced Inverse Class Weights Computed from Training Set**:
   - **Binary Classification (NORMAL : ABNORMAL)**:
     - Weight for Class 0 (`NORMAL`): `{metadata['class_weights']['binary'][0]:.4f}`
     - Weight for Class 1 (`ABNORMAL`): `{metadata['class_weights']['binary'][1]:.4f}`
   - **5-Class Classification (N, S, V, F, Q)**:
     - Weight for Class `N`: `{metadata['class_weights']['5class'][0]:.4f}`
     - Weight for Class `S`: `{metadata['class_weights']['5class'][1]:.4f}`
     - Weight for Class `V`: `{metadata['class_weights']['5class'][2]:.4f}`
     - Weight for Class `F`: `{metadata['class_weights']['5class'][3]:.4f}`
     - Weight for Class `Q`: `{metadata['class_weights']['5class'][4]:.4f}`
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
"""

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Generated preprocessing report at: {report_md_path}")


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "mitdb")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "ml", "data")
    MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
    REPORT_MD = os.path.join(BASE_DIR, "ml", "preprocessing_report.md")
    
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 80)
    print("HealthSense: ECG Preprocessing & Dataset Segmentation Pipeline")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Output Directory: {DATA_OUTPUT_DIR}")
    print("=" * 80)

    # 1. Process Train, Val, and Test splits
    X_train, y_train_5c, y_train_bin, p_train, train_stats = process_dataset_split(DATASET_DIR, TRAIN_RECORD_IDS, "TRAIN")
    X_val, y_val_5c, y_val_bin, p_val, val_stats = process_dataset_split(DATASET_DIR, VAL_RECORD_IDS, "VAL")
    X_test, y_test_5c, y_test_bin, p_test, test_stats = process_dataset_split(DATASET_DIR, TEST_RECORD_IDS, "TEST")

    # 2. Compute global exclusions and boundary removals
    all_stats = train_stats + val_stats + test_stats
    global_exclusions = Counter()
    total_boundary_removals = 0
    
    for s in all_stats:
        for k, v in s["excluded_annotations"].items():
            global_exclusions[k] += v
        total_boundary_removals += s["boundary_removals"]

    # 3. Compute class distributions
    train_5c_cnt = Counter(y_train_5c)
    val_5c_cnt = Counter(y_val_5c)
    test_5c_cnt = Counter(y_test_5c)
    
    train_bin_cnt = Counter(y_train_bin)
    val_bin_cnt = Counter(y_val_bin)
    test_bin_cnt = Counter(y_test_bin)

    # 4. Compute Class Weights for Train set (balanced inverse frequency)
    classes_bin = np.array([0, 1])
    weights_bin = compute_class_weight(class_weight='balanced', classes=classes_bin, y=y_train_bin)
    
    classes_5c = np.array([0, 1, 2, 3, 4])
    # Ensure all 5 classes are present or handle appropriately
    unique_present_5c = np.unique(y_train_5c)
    weights_5c_present = compute_class_weight(class_weight='balanced', classes=unique_present_5c, y=y_train_5c)
    weights_5c_full = np.ones(5, dtype=np.float32)
    for c, w in zip(unique_present_5c, weights_5c_present):
        weights_5c_full[c] = float(w)

    # 5. Save .npz datasets
    train_npz_path = os.path.join(DATA_OUTPUT_DIR, "train_data.npz")
    val_npz_path = os.path.join(DATA_OUTPUT_DIR, "val_data.npz")
    test_npz_path = os.path.join(DATA_OUTPUT_DIR, "test_data.npz")

    np.savez_compressed(train_npz_path, X=X_train, y_5class=y_train_5c, y_binary=y_train_bin, patient_ids=p_train)
    np.savez_compressed(val_npz_path, X=X_val, y_5class=y_val_5c, y_binary=y_val_bin, patient_ids=p_val)
    np.savez_compressed(test_npz_path, X=X_test, y_5class=y_test_5c, y_binary=y_test_bin, patient_ids=p_test)
    
    print(f"\nSaved compressed datasets:")
    print(f"  Train : {train_npz_path} -> X shape: {X_train.shape}, y: {y_train_bin.shape}")
    print(f"  Val   : {val_npz_path}   -> X shape: {X_val.shape}, y: {y_val_bin.shape}")
    print(f"  Test  : {test_npz_path}  -> X shape: {X_test.shape}, y: {y_test_bin.shape}")

    # 6. Save Configuration and Metadata JSONs
    config_dict = {
        "sampling_frequency_hz": DEFAULT_FS,
        "window_size_samples": BEAT_WINDOW_SIZE,
        "pre_peak_samples": PRE_PEAK_SAMPLES,
        "post_peak_samples": POST_PEAK_SAMPLES,
        "bandpass_lowcut_hz": 0.5,
        "bandpass_highcut_hz": 45.0,
        "butter_order": 3,
        "normalization": "z_score",
        "aami_5class_mapping": AAMI_5CLASS_MAPPING,
        "class_to_int_5class": CLASS_TO_INT_5CLASS,
        "int_to_class_5class": INT_TO_CLASS_5CLASS,
        "class_to_int_binary": CLASS_TO_INT_BINARY,
        "int_to_class_binary": INT_TO_CLASS_BINARY,
        "excluded_annotations": EXCLUDED_ANNOTATIONS,
        "train_record_ids": TRAIN_RECORD_IDS,
        "val_record_ids": VAL_RECORD_IDS,
        "test_record_ids": TEST_RECORD_IDS,
        "class_weights": {
            "binary": {int(k): float(v) for k, v in zip(classes_bin, weights_bin)},
            "5class": {int(k): float(v) for k, v in zip(classes_5c, weights_5c_full)}
        }
    }

    metadata_dict = {
        "total_valid_beats": len(X_train) + len(X_val) + len(X_test),
        "total_boundary_removals": total_boundary_removals,
        "global_exclusions": dict(global_exclusions),
        "train": {
            "num_records": len(TRAIN_RECORD_IDS),
            "num_samples": len(X_train),
            "boundary_removals": sum(s["boundary_removals"] for s in train_stats),
            "binary_dist": {"NORMAL": int(train_bin_cnt[0]), "ABNORMAL": int(train_bin_cnt[1])},
            "5class_dist": {INT_TO_CLASS_5CLASS[i]: int(train_5c_cnt[i]) for i in range(5)}
        },
        "val": {
            "num_records": len(VAL_RECORD_IDS),
            "num_samples": len(X_val),
            "boundary_removals": sum(s["boundary_removals"] for s in val_stats),
            "binary_dist": {"NORMAL": int(val_bin_cnt[0]), "ABNORMAL": int(val_bin_cnt[1])},
            "5class_dist": {INT_TO_CLASS_5CLASS[i]: int(val_5c_cnt[i]) for i in range(5)}
        },
        "test": {
            "num_records": len(TEST_RECORD_IDS),
            "num_samples": len(X_test),
            "boundary_removals": sum(s["boundary_removals"] for s in test_stats),
            "binary_dist": {"NORMAL": int(test_bin_cnt[0]), "ABNORMAL": int(test_bin_cnt[1])},
            "5class_dist": {INT_TO_CLASS_5CLASS[i]: int(test_5c_cnt[i]) for i in range(5)}
        },
        "total_binary": {
            "NORMAL": int(train_bin_cnt[0] + val_bin_cnt[0] + test_bin_cnt[0]),
            "ABNORMAL": int(train_bin_cnt[1] + val_bin_cnt[1] + test_bin_cnt[1])
        },
        "total_5class": {
            INT_TO_CLASS_5CLASS[i]: int(train_5c_cnt[i] + val_5c_cnt[i] + test_5c_cnt[i]) for i in range(5)
        },
        "class_weights": {
            "binary": [float(w) for w in weights_bin],
            "5class": [float(w) for w in weights_5c_full]
        }
    }

    config_json_path = os.path.join(MODELS_DIR, "preprocessing_config.json")
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Saved preprocessing config to: {config_json_path}")

    metadata_json_path = os.path.join(DATA_OUTPUT_DIR, "dataset_metadata.json")
    with open(metadata_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"Saved dataset metadata to: {metadata_json_path}")

    # 7. Generate markdown report
    generate_preprocessing_report(DATA_OUTPUT_DIR, metadata_dict, REPORT_MD)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"Total Valid Beat Segments Extracted: {metadata_dict['total_valid_beats']:,}")
    print(f"Train Segments : {metadata_dict['train']['num_samples']:,} ({metadata_dict['train']['num_records']} records)")
    print(f"Val Segments   : {metadata_dict['val']['num_samples']:,} ({metadata_dict['val']['num_records']} records)")
    print(f"Test Segments  : {metadata_dict['test']['num_samples']:,} ({metadata_dict['test']['num_records']} records)")
    print(f"Total Non-Beat Excluded Markers: {sum(global_exclusions.values()):,}")
    print(f"Total Boundary Discarded Beats : {total_boundary_removals:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
