"""
HealthSense - Dataset Inspection Script
MIT-BIH Arrhythmia Database Verification and Analysis using WFDB
"""

import os
import glob
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

def inspect_dataset(dataset_dir: str, output_csv: str, plot_output_path: str):
    print("=" * 80)
    print("HealthSense: MIT-BIH Arrhythmia Database Inspection")
    print(f"Dataset Directory: {os.path.abspath(dataset_dir)}")
    print("=" * 80)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # 1. Scan directory files
    all_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    
    dat_files = [f for f in all_files if f.endswith('.dat')]
    hea_files = [f for f in all_files if f.endswith('.hea')]
    atr_files = [f for f in all_files if f.endswith('.atr')]

    # Read RECORDS if available
    records_file = os.path.join(dataset_dir, 'RECORDS')
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            official_record_ids = [line.strip() for line in f if line.strip()]
    else:
        official_record_ids = sorted(list(set([os.path.splitext(f)[0] for f in hea_files])))

    # Find all unique record IDs discovered in files
    all_record_stems = sorted(list(set([os.path.splitext(f)[0] for f in (dat_files + hea_files + atr_files)])))

    print(f"\n--- File Inventory ---")
    print(f"Total .dat files : {len(dat_files)}")
    print(f"Total .hea files : {len(hea_files)}")
    print(f"Total .atr files : {len(atr_files)}")
    print(f"Official records in RECORDS file: {len(official_record_ids)}")
    print(f"Unique record IDs from file stems: {len(all_record_stems)}")

    # 2. Check for missing required files (.dat, .hea, .atr) for each record
    missing_files_report = {}
    valid_records = []
    
    for rec_id in official_record_ids:
        missing = []
        if f"{rec_id}.dat" not in dat_files:
            missing.append(".dat")
        if f"{rec_id}.hea" not in hea_files:
            missing.append(".hea")
        if f"{rec_id}.atr" not in atr_files:
            missing.append(".atr")
        
        if missing:
            missing_files_report[rec_id] = missing
        else:
            valid_records.append(rec_id)

    print(f"\n--- Missing Files Check ---")
    if missing_files_report:
        print(f"WARNING: The following {len(missing_files_report)} records have missing files:")
        for rec_id, missing in missing_files_report.items():
            print(f"  Record {rec_id}: Missing {', '.join(missing)}")
    else:
        print(f"All {len(valid_records)} official records have all required files (.dat, .hea, .atr) intact.")

    # 3. Read header and annotation information for each valid record
    summary_rows = []
    global_symbol_counter = Counter()

    print(f"\n--- Reading Headers & Annotations ---")
    for rec_id in valid_records:
        rec_path = os.path.join(dataset_dir, rec_id)
        
        # Read header / record info
        header = wfdb.rdheader(rec_path)
        fs = header.fs
        n_sig = header.n_sig
        sig_len = header.sig_len
        duration_sec = sig_len / fs if fs else 0
        duration_min = duration_sec / 60.0
        sig_names = header.sig_name
        units = header.units
        
        # Read annotations
        annotation = wfdb.rdann(rec_path, 'atr')
        num_annotations = len(annotation.sample)
        symbols = annotation.symbol
        symbol_counts = Counter(symbols)
        global_symbol_counter.update(symbols)

        symbol_breakdown_str = "; ".join([f"{sym}:{cnt}" for sym, cnt in sorted(symbol_counts.items())])

        summary_rows.append({
            "record_id": rec_id,
            "sampling_frequency_hz": fs,
            "num_channels": n_sig,
            "channel_names": ", ".join(sig_names),
            "units": ", ".join(units),
            "num_samples": sig_len,
            "duration_seconds": round(duration_sec, 2),
            "duration_minutes": round(duration_min, 2),
            "num_annotations": num_annotations,
            "annotation_breakdown": symbol_breakdown_str
        })

    summary_df = pd.DataFrame(summary_rows)
    
    # Save CSV summary
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved dataset summary CSV to: {output_csv}")

    # 4. Dataset Overview Aggregations
    print(f"\n" + "=" * 80)
    print("DATASET AGGREGATION SUMMARY")
    print("=" * 80)
    print(f"Total Valid Complete Records : {len(summary_df)}")
    print(f"Sampling Frequency (Hz)      : {summary_df['sampling_frequency_hz'].unique().tolist()}")
    print(f"ECG Channels per Record      : {summary_df['num_channels'].unique().tolist()}")
    print(f"Channel Configurations       : {summary_df['channel_names'].value_counts().to_dict()}")
    print(f"Total ECG Samples across DB  : {summary_df['num_samples'].sum():,}")
    print(f"Total Recording Time (hours) : {summary_df['duration_minutes'].sum() / 60.0:.2f} hours")
    print(f"Total Beat/Event Annotations : {summary_df['num_annotations'].sum():,}")

    # 5. Annotation Symbols Breakdown
    print(f"\n--- Annotation Symbol Distribution Across Dataset ---")
    print(f"{'Symbol':<8} {'Count':<10} {'Percentage':<12} {'AAMI/WFDB Description'}")
    print("-" * 65)

    symbol_descriptions = {
        'N': 'Normal beat',
        'L': 'Left bundle branch block beat',
        'R': 'Right bundle branch block beat',
        'B': 'Bundle branch block beat (unspecified)',
        'A': 'Atrial premature beat',
        'a': 'Aberrated atrial premature beat',
        'J': 'Nodal (junctional) premature beat',
        'S': 'Supraventricular premature or ectopic beat',
        'V': 'Premature ventricular contraction (PVC)',
        'F': 'Fusion of ventricular and normal beat',
        '[ ': 'Start of ventricular flutter/fibrillation',
        '!': 'Ventricular flutter wave',
        ']': 'End of ventricular flutter/fibrillation',
        'e': 'Atrial escape beat',
        'j': 'Nodal (junctional) escape beat',
        'E': 'Ventricular escape beat',
        '/': 'Paced beat',
        'f': 'Fusion of paced and normal beat',
        'Q': 'Unclassifiable beat',
        '?': 'Beat not classified by annotator',
        '+': 'Rhythm change',
        '~': 'Signal quality change',
        '|': 'Isolated QRS-like artifact',
        'x': 'Non-conducted P-wave (blocked APC)',
        '"': 'Comment annotation',
    }

    total_ann = sum(global_symbol_counter.values())
    for sym, count in global_symbol_counter.most_common():
        pct = (count / total_ann) * 100
        desc = symbol_descriptions.get(sym, 'Other rhythm/auxiliary marker')
        print(f"{sym:<8} {count:<10} {pct:<10.2f}%  {desc}")

    # 6. Plotting Record 100 for ~10 seconds with Annotations
    plot_record_id = "100"
    if plot_record_id in valid_records:
        print(f"\n--- Generating ECG Plot for Record {plot_record_id} ---")
        rec_path = os.path.join(dataset_dir, plot_record_id)
        
        # Read 10 seconds of signal
        # At 360 Hz, 10 seconds = 3600 samples
        samp_from = 0
        samp_to = 360 * 10  # 10 seconds
        
        record = wfdb.rdrecord(rec_path, sampfrom=samp_from, sampto=samp_to)
        annotation = wfdb.rdann(rec_path, 'atr', sampfrom=samp_from, sampto=samp_to)
        
        time_axis = [i / record.fs for i in range(record.sig_len)]
        
        fig, axes = plt.subplots(record.n_sig, 1, figsize=(14, 7), sharex=True)
        if record.n_sig == 1:
            axes = [axes]
            
        for ch_idx in range(record.n_sig):
            ax = axes[ch_idx]
            sig_name = record.sig_name[ch_idx]
            sig_data = record.p_signal[:, ch_idx]
            unit = record.units[ch_idx]
            
            ax.plot(time_axis, sig_data, label=f"Channel {ch_idx}: {sig_name}", color="#1f77b4" if ch_idx == 0 else "#2ca02c", linewidth=1.2)
            ax.set_ylabel(f"Amplitude ({unit})", fontsize=11, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            
            # Add annotation markers
            if ch_idx == 0:
                ax.set_title(f"HealthSense — MIT-BIH Arrhythmia Database: Record {plot_record_id} (First 10 Seconds with Annotations)", fontsize=13, fontweight='bold', pad=12)
                
                # Annotate peaks
                ann_samples = annotation.sample - samp_from
                ann_symbols = annotation.symbol
                
                for s, sym in zip(ann_samples, ann_symbols):
                    if 0 <= s < len(sig_data):
                        val = sig_data[s]
                        t = s / record.fs
                        ax.scatter(t, val, color='red', s=45, zorder=5)
                        ax.annotate(
                            sym, 
                            (t, val), 
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center', 
                            fontsize=10, 
                            fontweight='bold', 
                            color='#d62728',
                            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#d62728", lw=1.2, alpha=0.85)
                        )

        axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
        plt.savefig(plot_output_path, dpi=300)
        plt.close()
        print(f"Saved annotated ECG plot to: {plot_output_path}")

    return {
        "total_dat": len(dat_files),
        "total_hea": len(hea_files),
        "total_atr": len(atr_files),
        "total_valid_records": len(valid_records),
        "missing_files_report": missing_files_report,
        "csv_path": output_csv,
        "plot_path": plot_output_path
    }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "mitdb")
    CSV_OUTPUT = os.path.join(BASE_DIR, "ml", "mitdb_dataset_summary.csv")
    PLOT_OUTPUT = os.path.join(BASE_DIR, "ml", "ecg_record_100_annotated.png")
    
    inspect_dataset(DATASET_DIR, CSV_OUTPUT, PLOT_OUTPUT)
