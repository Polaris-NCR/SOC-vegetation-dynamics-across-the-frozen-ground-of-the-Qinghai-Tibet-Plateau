"""
Extract Landsat 8 data from GEE at QTP SOC point locations,
run local super-resolution model to generate Hyperion 57-band spectra,
and build the paired (spectra, SOC) dataset for Random Forest.

Pipeline:
  1. Read QTP SOC points (lat/lon) from Excel
  2. For each point, fetch Landsat 8 SR (6 bands) via GEE/geemap
     -> 64x64 patch centered on point
  3. Run trained UNet/Diffusion model -> Hyperion 57-band 64x64 patch
  4. Extract center pixel spectrum + surround statistics
  5. Output paired CSV for rf_soc_prediction.py

Dependencies: geemap, earthengine-api, torch, joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ========== Configuration ==========
MODEL_DIR       = "./regression_checkpoints"          # trained UNet models
OUTPUT_CSV_FMT  = "data2026/QTP_boreholes_hyperion_0_{depth}cm.csv"
PATCH_SIZE      = 64     # model input patch size
TARGET_DEPTHS   = [30, 50, 100]

# Landsat band names (GEE export order, unified across LT5/LE7/LC8)
LANDSAT_BANDS   = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']


# ========== 2. GEE Landsat multi-sensor extraction ==========

def create_landsat_gee_script():
    """
    Returns the path to the standalone GEE extraction script.
    The script is maintained as a separate file: code/extract_landsat_gee.py
    (no longer embedded/generated — directly editable and version-controlled).
    """
    script_path = os.path.join('code', 'extract_landsat_gee.py')
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"GEE script not found: {script_path}\n"
            f"It should be in the code/ directory alongside this file.")
    print(f"\nGEE extraction script: {script_path}")
    print(f"Run it with:  python {script_path}")
    return script_path


# ========== 3. Local model inference: Landsat -> Hyperion ==========

def load_trained_models(model_dir, device='cuda'):
    """
    Load the 6 trained UNet regression models (one per band group).
    Uses RegressionUNet from the project's Regression_Baseline module.
    """
    sys.path.insert(0, 'code')
    from Regression_Baseline import RegressionUNet, BAND_HYPERION_COUNTS, BAND_ORDER

    models = {}
    for band_name in BAND_ORDER:
        n_hyper = BAND_HYPERION_COUNTS[band_name]
        model = RegressionUNet(in_channels=n_hyper, cond_channels=1,
                               out_channels=n_hyper, base_channels=32)
        ckpt_path = os.path.join(model_dir, f'{band_name}_best.pth')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            model.eval()
            models[band_name] = model
            print(f"  Loaded {band_name}: {n_hyper} bands")
        else:
            print(f"  WARNING: {ckpt_path} not found, skipping {band_name}")
    return models


def extract_window_stats(patch, window_size=5):
    """
    Extract mean and std from a centered window of size window_size x window_size.

    Args:
        patch: numpy array [C, H, W]
        window_size: int, must be odd

    Returns:
        mean:  [C,]  — window mean
        std:   [C,]  — window standard deviation
    """
    _, h, w = patch.shape
    half = window_size // 2
    cy, cx = h // 2, w // 2
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    window = patch[:, y0:y1, x0:x1]  # [C, W, W]

    # Mask zero/NaN pixels (as in training data cleaning)
    means, stds = [], []
    for c in range(window.shape[0]):
        vals = window[c].ravel()
        # Exclude NaN and near-zero (< 1e-8) values
        valid = vals[(~np.isnan(vals)) & (np.abs(vals) > 1e-8)]
        if len(valid) >= 3:  # need at least 3 valid pixels
            means.append(valid.mean())
            stds.append(valid.std())
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def landsat_to_hyperion_single(models, landsat_patch, cond_idx,
                               device='cuda'):
    """
    Convert Landsat 6-band patch [6, H, W] -> Hyperion 57-band full patch.

    Returns:
        hyperion_patch: numpy array [57, H, W]
    """
    from Regression_Baseline import BAND_ORDER

    all_bands = []
    for band_name in BAND_ORDER:
        if band_name not in models:
            continue
        idx = cond_idx[band_name]
        cond = torch.from_numpy(
            landsat_patch[idx:idx+1]).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = models[band_name](cond)  # [1, N_hyper, H, W]

        all_bands.append(pred.squeeze(0).cpu().numpy())  # [N_hyper, H, W]

    hyperion_patch = np.concatenate(all_bands, axis=0)  # [57, H, W]
    return hyperion_patch


def batch_inference(models, landsat_patches_dir, soc_df, cond_idx,
                    window_size=5, device='cuda'):
    """
    Run model inference for all SOC points.

    For each point:
      1. Landsat patch -> Hyperion patch (57, 64, 64)
      2. Extract 5x5 window mean + std from Hyperion center
      3. Extract 5x5 window mean + std from Landsat center
      4. Output paired CSV with ~(57+57)*2 + 6*2 = 240 spectral features
    """
    records = []
    status_path = os.path.join(landsat_patches_dir, 'extraction_status.csv')

    # Read extraction status — maps patch_id -> lat/lon/path
    if os.path.exists(status_path):
        status_df = pd.read_csv(status_path)
        ok_entries = status_df[status_df['status'] == 'ok'].copy()
    else:
        raise FileNotFoundError(f"Status file not found: {status_path}")

    # Build lat/lon -> [list of SOC rows] lookup (1:N: keep duplicate locations)
    soc_lookup = {}
    for _, row in soc_df.iterrows():
        key = (round(row['Longitude'], 6), round(row['Latitude'], 6))
        if key not in soc_lookup:
            soc_lookup[key] = []
        soc_lookup[key].append(row)

    total = len(ok_entries)
    total_profiles = 0

    for i, (_, entry) in enumerate(ok_entries.iterrows()):
        if i % 100 == 0:
            print(f"  Inference: {i}/{total} (profiles: {total_profiles})")

        patch_path = entry['patch_path']
        if not os.path.exists(patch_path):
            # Try fallback: {patch_id}.npy in landsat_patches_dir
            pid = int(entry['patch_id'])
            patch_path = os.path.join(landsat_patches_dir, f'{pid}.npy')
            if not os.path.exists(patch_path):
                continue

        landsat_patch = np.load(patch_path)

        # Mask: skip patches with zero/NaN values (as in training data cleaning)
        # Training removed zero-value or empty Landsat pixels
        if np.any(np.isnan(landsat_patch)):
            continue
        # Skip if all bands are near-zero (empty/unvegetated/sensor artifact)
        band_means = landsat_patch.mean(axis=(1, 2))  # [6,]
        if np.all(band_means < 0.001):
            continue
        # Skip if any band is entirely zero
        if np.any(np.all(landsat_patch == 0, axis=(1, 2))):
            continue

        try:
            hyperion_patch = landsat_to_hyperion_single(
                models, landsat_patch, cond_idx, device=device)
        except Exception as e:
            print(f"  Error on patch {entry['patch_id']}: {e}")
            continue

        # --- Extract window statistics ---
        hyp_mean, hyp_std = extract_window_stats(hyperion_patch, window_size)
        lnd_mean, lnd_std = extract_window_stats(landsat_patch, window_size)

        # --- Build record ---
        record = {'patch_id': int(entry['patch_id'])}

        for b in range(56):
            record[f'Hyperion_B{b+1:03d}'] = hyp_mean[b]
            record[f'Hyperion_B{b+1:03d}_std'] = hyp_std[b]

        for b_idx, band_name in enumerate(LANDSAT_BANDS):
            record[f'Landsat_{band_name}'] = lnd_mean[b_idx]
            record[f'Landsat_{band_name}_std'] = lnd_std[b_idx]

        # --- Build one row per SOC profile at this location ---
        lon, lat = entry['Longitude'], entry['Latitude']
        key = (round(lon, 6), round(lat, 6))

        if key in soc_lookup:
            for src in soc_lookup[key]:  # 1:N — keep all profiles at same lat/lon
                row = dict(record)
                for d in TARGET_DEPTHS:
                    col = f'SOC_0_{d}_cm_kg_m2'
                    if col in src.index and pd.notna(src[col]):
                        row[col] = src[col]
                for col in ['Longitude', 'Latitude', 'DEM_m', 'MAT_C', 'MAP_mm',
                            'PET_mm', 'AI', 'CLCD', 'Time',
                            'BD_0_30_cm', 'BD_0_50_cm', 'BD_0_100_cm',
                            'Sand_0_30_cm', 'Sand_0_50_cm', 'Sand_0_100_cm',
                            'Silt_0_30_cm', 'Silt_0_50_cm', 'Silt_0_100_cm',
                            'Clay_0_30_cm', 'Clay_0_50_cm', 'Clay_0_100_cm',
                            'Profile_ID', 'Slope_deg', 'Aspect_deg', 'TWI',
                            'NDVI', 'OSAVI', 'EVI', 'SAVI', 'RVI', 'TVI', 'DVI', 'MSAVI']:
                    if col in src.index and pd.notna(src[col]):
                        row[col] = src[col]
                records.append(row)
                total_profiles += 1
        # else: no SOC at this depth for this location → skip

    print(f"  Total profiles: {total_profiles}")
    return pd.DataFrame(records)


# ========== 4. Main pipeline ==========5


def load_soc_for_depth(depth_cm):
    """
    Load SOC profiles for a specific depth interval (30, 50, or 100 cm).

    Reads the filled Excel (with BD/Sand/Silt/Clay + MAT/MAP already completed).
    Time column is already present in the filled files — no CSV merge needed.

    Returns:
        soc_df: DataFrame with all columns from the filled Excel.
        Only profiles where SOC > 0 at this depth.
    """
    soc_path = f"data2026/QTP-boreholes-{depth_cm}cm.xlsx"

    if not os.path.exists(soc_path):
        raise FileNotFoundError(
            f"SOC depth file not found: {soc_path}\n"
            f"Run extract_qtp_full_by_depth.py and fill_soil_props_openlandmap.py first.")

    soc_df = pd.read_excel(soc_path)

    soc_col = f"SOC_0_{depth_cm}_cm_kg_m2"
    if soc_col not in soc_df.columns:
        raise KeyError(f"Column {soc_col} not found in {soc_path}")

    soc_df = soc_df.dropna(subset=[soc_col])
    soc_df = soc_df[soc_df[soc_col] > 0].copy()

    return soc_df


def main(target_depth=30, model_dir=MODEL_DIR, device='cuda'):
    """
    Complete extraction pipeline for ONE target depth.

    Args:
        target_depth: 30, 50, or 100 (cm)

    Steps:
      1. Load depth-specific SOC profiles (only those with SOC > 0)
      2. Generate GEE extraction script (manual, shared across depths)
      3. Load local super-resolution models
      4. Run inference on Landsat patches -> Hyperion 5x5 window stats
      5. Save depth-specific paired CSV
    """
    output_csv = OUTPUT_CSV_FMT.format(depth=target_depth)
    landsat_dir = './landsat_patches'

    # Step 1: Load SOC data for this depth
    print(f"\n{'='*60}")
    print(f"Target depth: 0-{target_depth}cm")
    print(f"{'='*60}")
    soc_df = load_soc_for_depth(target_depth)

    print(f"Loaded {len(soc_df)} SOC profiles (SOC > 0 at 0-{target_depth}cm)")
    print(f"  Lon: {soc_df['Longitude'].min():.2f} - {soc_df['Longitude'].max():.2f}")
    print(f"  Lat: {soc_df['Latitude'].min():.2f} - {soc_df['Latitude'].max():.2f}")
    if 'Time' in soc_df.columns:
        has_time = soc_df['Time'].notna().sum()
        print(f"  Time: {has_time}/{len(soc_df)} have sampling year")

    # Step 2: Generate GEE extraction script
    # The GEE script uses ALL profiles from the raw CSV (so patches are
    # shared across depths). We pass soc_df only to seed the script path.
    os.makedirs(landsat_dir, exist_ok=True)
    gee_script = create_landsat_gee_script()

    # Step 3: Check existing Landsat patches
    existing = len([f for f in os.listdir(landsat_dir)
                    if f.endswith('.npy')]) if os.path.exists(landsat_dir) else 0
    needed = len(soc_df)
    print(f"\nLandsat patches: {existing} on disk, {needed} needed for 0-{target_depth}cm")

    if existing == 0:
        print(f"\nNo Landsat patches found. Run the GEE extraction script:")
        print(f"  python {gee_script}")
        print("Then re-run this script to complete the pipeline.")
        return None

    # Step 4: Load local super-resolution models
    print(f"\nLoading models from {model_dir}...")
    sys.path.insert(0, 'code')
    from Regression_Baseline import BAND_COND_IDX
    models = load_trained_models(model_dir, device)

    if not models:
        print("No models loaded. Train models first (Regression_Baseline.py).")
        return None

    # Step 5: Run Hyperion inference (only for profiles with valid SOC
    #         at this depth — batch_inference filters by soc_df)
    print(f"\nRunning Hyperion inference...")
    paired_df = batch_inference(models, landsat_dir, soc_df, BAND_COND_IDX, device=device)

    # Step 6: Save depth-specific paired dataset
    paired_df.to_csv(output_csv, index=False)
    n_hyperion = len([c for c in paired_df.columns if 'Hyperion' in c])
    n_soc = len([c for c in paired_df.columns if 'SOC_0' in c])
    print(f"\nSaved: {output_csv}")
    print(f"  {len(paired_df)} samples, {n_hyperion} spectral features, {n_soc} SOC columns")
    print(f"\nReady for Random Forest:")
    print(f"  python code/rf_soc_prediction.py  (set TARGET_DEPTH={target_depth})")

    return paired_df


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    import argparse
    parser = argparse.ArgumentParser(
        description='Build Hyperion-SOC paired dataset for one depth')
    parser.add_argument('--depth', type=int, default=30,
                        choices=[30, 50, 100],
                        help='Target soil depth (cm)')
    args = parser.parse_args()
    main(target_depth=args.depth, device=device)
