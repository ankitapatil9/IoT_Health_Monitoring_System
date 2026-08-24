"""
HealthSense - Comprehensive Optimized ECG Prediction Verification Suite
Executes Steps 3, 4, 5, 6, 7 and produces full verification metrics.
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from predict_ecg_optimized import ECGOptimizedPredictor, predict_ecg_beat, DEFAULT_THRESHOLD, DEFAULT_MODEL_PATH
import preprocessing

def run_suite():
    print("=" * 80)
    print("HEALTHSENSE OPTIMIZED ECG PREDICTION VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1 & 7: THRESHOLD & MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    print("\n--- STEP 1 & 7: PREDICTOR INITIALIZATION & THRESHOLD VERIFICATION ---")
    predictor = ECGOptimizedPredictor()
    threshold_val = predictor.threshold
    print(f"Model Path Loaded  : {predictor.model_path}")
    print(f"Decision Threshold : {threshold_val}")
    
    threshold_pass = (threshold_val == 0.30)
    print(f"DECISION THRESHOLD VERIFICATION: {'PASS' if threshold_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 3: KNOWN TEST DATA SAMPLES
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: TEST KNOWN TEST-DATA SAMPLES ---")
    test_data_path = os.path.join(BASE_DIR, "data", "test_data.npz")
    test_npz = np.load(test_data_path)
    X_test = test_npz['X']
    y_test = test_npz['y_binary']
    y_5class = test_npz['y_5class']
    patient_ids = test_npz['patient_ids']

    # Find known NORMAL beats (class 0) and known ABNORMAL beats (class 1)
    normal_indices = np.where(y_test == 0)[0]
    abnormal_indices = np.where(y_test == 1)[0]

    print(f"Total Test Samples : {len(y_test):,}")
    print(f"Total Normal Beats : {len(normal_indices):,}")
    print(f"Total Abnormal Beats: {len(abnormal_indices):,}")

    sample_checks = []

    # Sample 1: Normal beat (index 0)
    idx_norm = normal_indices[0]
    res_norm = predictor.predict(X_test[idx_norm], apply_zscore=False)
    pass_norm = (res_norm['prediction'] == 'NORMAL' and res_norm['class_id'] == 0)
    sample_checks.append({
        "sample_index": int(idx_norm),
        "patient_id": str(patient_ids[idx_norm]),
        "5class_label": int(y_5class[idx_norm]),
        "true_label": "NORMAL (0)",
        "predicted_label": res_norm['prediction'],
        "class_id": res_norm['class_id'],
        "abnormal_probability": res_norm['abnormal_probability'],
        "confidence": res_norm['confidence'],
        "threshold": res_norm['threshold'],
        "status": res_norm['status'],
        "result": "PASS" if pass_norm else "FAIL"
    })

    # Sample 2: Another Normal beat (index 100)
    idx_norm2 = normal_indices[100]
    res_norm2 = predictor.predict(X_test[idx_norm2], apply_zscore=False)
    pass_norm2 = (res_norm2['prediction'] == 'NORMAL' and res_norm2['class_id'] == 0)
    sample_checks.append({
        "sample_index": int(idx_norm2),
        "patient_id": str(patient_ids[idx_norm2]),
        "5class_label": int(y_5class[idx_norm2]),
        "true_label": "NORMAL (0)",
        "predicted_label": res_norm2['prediction'],
        "class_id": res_norm2['class_id'],
        "abnormal_probability": res_norm2['abnormal_probability'],
        "confidence": res_norm2['confidence'],
        "threshold": res_norm2['threshold'],
        "status": res_norm2['status'],
        "result": "PASS" if pass_norm2 else "FAIL"
    })

    # Sample 3: Known Abnormal beat (index 6077 - PVC/Ventricular beat)
    idx_abnorm = 6077
    res_abnorm = predictor.predict(X_test[idx_abnorm], apply_zscore=False)
    pass_abnorm = (res_abnorm['prediction'] == 'ABNORMAL' and res_abnorm['class_id'] == 1)
    sample_checks.append({
        "sample_index": int(idx_abnorm),
        "patient_id": str(patient_ids[idx_abnorm]),
        "5class_label": int(y_5class[idx_abnorm]),
        "true_label": "ABNORMAL (1)",
        "predicted_label": res_abnorm['prediction'],
        "class_id": res_abnorm['class_id'],
        "abnormal_probability": res_abnorm['abnormal_probability'],
        "confidence": res_abnorm['confidence'],
        "threshold": res_abnorm['threshold'],
        "status": res_abnorm['status'],
        "result": "PASS" if pass_abnorm else "FAIL"
    })

    # Sample 4: Known Abnormal beat (first occurrence in abnormal indices)
    idx_abnorm2 = abnormal_indices[0]
    res_abnorm2 = predictor.predict(X_test[idx_abnorm2], apply_zscore=False)
    pass_abnorm2 = (res_abnorm2['prediction'] == 'ABNORMAL' and res_abnorm2['class_id'] == 1)
    sample_checks.append({
        "sample_index": int(idx_abnorm2),
        "patient_id": str(patient_ids[idx_abnorm2]),
        "5class_label": int(y_5class[idx_abnorm2]),
        "true_label": "ABNORMAL (1)",
        "predicted_label": res_abnorm2['prediction'],
        "class_id": res_abnorm2['class_id'],
        "abnormal_probability": res_abnorm2['abnormal_probability'],
        "confidence": res_abnorm2['confidence'],
        "threshold": res_abnorm2['threshold'],
        "status": res_abnorm2['status'],
        "result": "PASS" if pass_abnorm2 else "FAIL"
    })

    for s in sample_checks:
        print(f"Sample Index       : {s['sample_index']}")
        print(f"  Patient ID       : {s['patient_id']}")
        print(f"  True Label       : {s['true_label']}")
        print(f"  Predicted Label  : {s['predicted_label']} (class_id: {s['class_id']})")
        print(f"  Abnormal Prob    : {s['abnormal_probability']:.6f}")
        print(f"  Confidence       : {s['confidence']:.6f} ({s['confidence']*100:.2f}%)")
        print(f"  Threshold        : {s['threshold']}")
        print(f"  Status           : {s['status']}")
        print(f"  Verification     : {s['result']}\n")

    # -------------------------------------------------------------------------
    # STEP 4: COMPLETE TEST DATASET VERIFICATION
    # -------------------------------------------------------------------------
    print("--- STEP 4: COMPLETE UNTOUCHED TEST DATASET VERIFICATION ---")
    print(f"Running inference across all {len(X_test):,} test beats...")
    
    batch_results = predictor.predict_batch(X_test, apply_zscore=False)
    y_pred_binary = np.array([r['class_id'] for r in batch_results])
    y_pred_probs = np.array([r['abnormal_probability'] for r in batch_results])

    test_acc = float(accuracy_score(y_test, y_pred_binary))
    test_prec = float(precision_score(y_test, y_pred_binary, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred_binary, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))
    test_roc_auc = float(roc_auc_score(y_test, y_pred_probs))

    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    test_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print("\nVerification Metrics on Complete Test Dataset:")
    print(f"  Accuracy            : {test_acc*100:.4f}% ({test_acc:.6f})")
    print(f"  Precision           : {test_prec*100:.4f}% ({test_prec:.6f})")
    print(f"  Recall/Sensitivity  : {test_rec*100:.4f}% ({test_rec:.6f})")
    print(f"  Specificity         : {test_spec*100:.4f}% ({test_spec:.6f})")
    print(f"  F1-Score            : {test_f1:.6f}")
    print(f"  ROC-AUC             : {test_roc_auc:.6f}")
    print(f"  Confusion Matrix    : TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    # Load baseline/optimization results to compare
    with open(os.path.join(BASE_DIR, "models", "optimization_results.json"), "r") as f:
        opt_res = json.load(f)
    exp_m = opt_res["test_metrics"]
    exp_cm = exp_m["confusion_matrix"]

    match_acc = np.isclose(test_acc, exp_m['accuracy'], atol=1e-4)
    match_prec = np.isclose(test_prec, exp_m['precision'], atol=1e-4)
    match_rec = np.isclose(test_rec, exp_m['recall_sensitivity'], atol=1e-4)
    match_spec = np.isclose(test_spec, exp_m['specificity'], atol=1e-4)
    match_f1 = np.isclose(test_f1, exp_m['f1_score'], atol=1e-4)
    match_auc = np.isclose(test_roc_auc, exp_m['roc_auc'], atol=1e-4)
    match_cm = (tn == exp_cm['tn'] and fp == exp_cm['fp'] and fn == exp_cm['fn'] and tp == exp_cm['tp'])

    all_metrics_match = (match_acc and match_prec and match_rec and match_spec and match_f1 and match_auc and match_cm)
    print(f"\nMETRICS EXACT REPRODUCIBILITY MATCH: {'PASS' if all_metrics_match else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 5: ERROR HANDLING TESTS
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: ERROR HANDLING VALIDATION ---")
    error_tests = []

    # 1. ECG with incorrect length (255 samples)
    try:
        predictor.predict([0.5] * 255)
        error_tests.append(("1. Too few samples (255 samples)", "FAIL", "Did not raise ValueError"))
    except ValueError as e:
        error_tests.append(("1. Too few samples (255 samples)", "PASS", str(e)))
    except Exception as e:
        error_tests.append(("1. Too few samples (255 samples)", "FAIL", f"Wrong exception: {type(e).__name__}"))

    # 2. ECG with too many samples (257 samples)
    try:
        predictor.predict([0.5] * 257)
        error_tests.append(("2. Too many samples (257 samples)", "FAIL", "Did not raise ValueError"))
    except ValueError as e:
        error_tests.append(("2. Too many samples (257 samples)", "PASS", str(e)))
    except Exception as e:
        error_tests.append(("2. Too many samples (257 samples)", "FAIL", f"Wrong exception: {type(e).__name__}"))

    # 3. ECG containing NaN
    try:
        sample_nan = [0.0] * 255 + [float('nan')]
        predictor.predict(sample_nan)
        error_tests.append(("3. Contains NaN", "FAIL", "Did not raise ValueError"))
    except ValueError as e:
        error_tests.append(("3. Contains NaN", "PASS", str(e)))
    except Exception as e:
        error_tests.append(("3. Contains NaN", "FAIL", f"Wrong exception: {type(e).__name__}"))

    # 4. ECG containing Inf
    try:
        sample_inf = [0.0] * 255 + [float('inf')]
        predictor.predict(sample_inf)
        error_tests.append(("4. Contains Inf", "FAIL", "Did not raise ValueError"))
    except ValueError as e:
        error_tests.append(("4. Contains Inf", "PASS", str(e)))
    except Exception as e:
        error_tests.append(("4. Contains Inf", "FAIL", f"Wrong exception: {type(e).__name__}"))

    # 5. Empty ECG input
    try:
        predictor.predict([])
        error_tests.append(("5. Empty input []", "FAIL", "Did not raise ValueError"))
    except ValueError as e:
        error_tests.append(("5. Empty input []", "PASS", str(e)))
    except Exception as e:
        error_tests.append(("5. Empty input []", "FAIL", f"Wrong exception: {type(e).__name__}"))

    # 6. None input
    try:
        predictor.predict(None)
        error_tests.append(("6. None input", "FAIL", "Did not raise ValueError"))
    except ValueError as e:
        error_tests.append(("6. None input", "PASS", str(e)))
    except Exception as e:
        error_tests.append(("6. None input", "FAIL", f"Wrong exception: {type(e).__name__}"))

    for test_name, status, msg in error_tests:
        print(f"  [{status}] {test_name}: {msg}")

    all_error_tests_pass = all(st == "PASS" for _, st, _ in error_tests)
    print(f"\nERROR HANDLING VERIFICATION: {'PASS' if all_error_tests_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 6: PREPROCESSING CONSISTENCY CHECK
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: PREPROCESSING CONSISTENCY VERIFICATION ---")
    # Verify normalization output
    raw_test_beat = np.random.randn(256)
    norm_from_prep = preprocessing.z_score_normalize(raw_test_beat)
    tensor_from_pred = predictor.preprocess_input(raw_test_beat, apply_zscore=True)
    norm_from_pred = tensor_from_pred.squeeze()

    zscore_diff = np.max(np.abs(norm_from_prep - norm_from_pred))
    shape_match = (tensor_from_pred.shape == (1, 256, 1))
    dtype_match = (tensor_from_pred.dtype == np.float32)

    print(f"  Shape Check       : {tensor_from_pred.shape} (Expected (1, 256, 1)) -> {'PASS' if shape_match else 'FAIL'}")
    print(f"  Dtype Check       : {tensor_from_pred.dtype} (Expected float32) -> {'PASS' if dtype_match else 'FAIL'}")
    print(f"  Z-Score Max Diff  : {zscore_diff:.10e} -> {'PASS' if zscore_diff < 1e-6 else 'FAIL'}")

    preprocessing_match = shape_match and dtype_match and (zscore_diff < 1e-6)
    print(f"PREPROCESSING MATCH: {'PASS' if preprocessing_match else 'FAIL'}")

    # Return summary dict for report generation
    return {
        "threshold_pass": threshold_pass,
        "sample_checks": sample_checks,
        "complete_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "specificity": test_spec,
            "f1_score": test_f1,
            "roc_auc": test_roc_auc,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        },
        "expected_metrics": exp_m,
        "metrics_match": all_metrics_match,
        "error_tests": error_tests,
        "all_error_pass": all_error_tests_pass,
        "preprocessing_match": preprocessing_match
    }

if __name__ == "__main__":
    run_suite()
