import pandas as pd
import numpy as np
from tqdm import tqdm
import os

DATA_ROOT = 'data'
MAX_SEQ_LEN = 200
OUTPUT_DIR = 'processed_data'
TIME_SCALE = 100.0  # <--- FIX: Scale time xuống để tránh nổ gradient

# Extinction Coeffs
R_WAVELENGTH = {'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31}
BAND_MAP = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_arcsinh(x):
    """Xử lý an toàn cho arcsinh: thay NaN/Inf bằng 0"""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.arcsinh(x)


def process_flux(flux, flux_err, ebv, band):
    # 1. Sanitize Inputs
    if pd.isna(flux) or np.isinf(flux): flux = 0.0
    if pd.isna(flux_err) or np.isinf(flux_err): flux_err = 0.0
    if pd.isna(ebv) or np.isinf(ebv): ebv = 0.0

    # 2. De-extinction
    r_val = R_WAVELENGTH.get(band, 0.0)
    correction = np.power(10, 0.4 * r_val * ebv)

    true_flux = flux * correction
    true_err = flux_err * correction

    # 3. Log-transform an toàn
    return safe_arcsinh(true_flux), safe_arcsinh(true_err)


def process_dataset(mode='train'):
    print(f"🛡️ Đang xử lý an toàn tập {mode} (Robust Mode)...")

    log_file = os.path.join(DATA_ROOT, f'{mode}_log.csv')
    meta_df = pd.read_csv(log_file)

    # Containers
    all_numeric, all_bands, all_masks = [], [], []
    all_metas, all_targets, all_ids = [], [], []

    splits = meta_df['split'].unique()

    for sp in tqdm(splits):
        lc_path = os.path.join(DATA_ROOT, sp, f'{mode}_full_lightcurves.csv')
        if not os.path.exists(lc_path): continue

        lc_df = pd.read_csv(lc_path)
        lc_groups = lc_df.groupby('object_id')
        objs_in_split = meta_df[meta_df['split'] == sp]

        for _, row in objs_in_split.iterrows():
            obj_id = row['object_id']
            # --- META HANDLING ---
            ebv = row.get('EBV', 0.0)
            z = row.get('Z', 0.0)
            z_err = row.get('Z_err', 0.0)

            # Clean metadata
            meta_vec = np.nan_to_num([z, z_err, ebv], nan=0.0).astype(np.float32)

            # Target
            target = row['target'] if 'target' in row else -1

            # --- LIGHTCURVE HANDLING ---
            if obj_id in lc_groups.groups:
                group = lc_groups.get_group(obj_id).sort_values('Time (MJD)')

                times = group['Time (MJD)'].values
                fluxes = group['Flux'].values
                errors = group['Flux_err'].values
                bands = group['Filter'].values

                # 1. Time Normalization & Scaling
                t0 = times[0]
                rel_time = (times - t0) / TIME_SCALE  # <--- FIX QUAN TRỌNG

                # 2. Process Flux
                proc_flux, proc_err, proc_band = [], [], []
                for f, e, b in zip(fluxes, errors, bands):
                    pf, pe = process_flux(f, e, ebv, b)
                    proc_flux.append(pf)
                    proc_err.append(pe)
                    proc_band.append(BAND_MAP.get(b, 6))  # 6 is unknown

                feat_numeric = np.stack([proc_flux, proc_err, rel_time], axis=1)
                feat_band = np.array(proc_band)

            else:
                feat_numeric = np.zeros((1, 3))
                feat_band = np.zeros((1,))

            # 3. Padding/Truncating
            L = len(feat_numeric)
            if L >= MAX_SEQ_LEN:
                feat_numeric = feat_numeric[:MAX_SEQ_LEN]
                feat_band = feat_band[:MAX_SEQ_LEN]
                mask = np.ones(MAX_SEQ_LEN)
            else:
                pad_len = MAX_SEQ_LEN - L
                feat_numeric = np.pad(feat_numeric, ((0, pad_len), (0, 0)), 'constant')
                feat_band = np.pad(feat_band, (0, pad_len), 'constant', constant_values=6)
                mask = np.concatenate([np.ones(L), np.zeros(pad_len)])

            all_numeric.append(feat_numeric)
            all_bands.append(feat_band)
            all_masks.append(mask)
            all_metas.append(meta_vec)
            all_targets.append(target)
            all_ids.append(obj_id)

    # Save
    print(f"💾 Saving clean {mode} data...")
    np.save(f'{OUTPUT_DIR}/{mode}_numeric.npy', np.array(all_numeric, dtype=np.float32))
    np.save(f'{OUTPUT_DIR}/{mode}_bands.npy', np.array(all_bands, dtype=np.int16))
    np.save(f'{OUTPUT_DIR}/{mode}_mask.npy', np.array(all_masks, dtype=bool))
    np.save(f'{OUTPUT_DIR}/{mode}_meta.npy', np.array(all_metas, dtype=np.float32))
    np.save(f'{OUTPUT_DIR}/{mode}_target.npy', np.array(all_targets, dtype=np.float32))
    print("✅ Done.")


if __name__ == "__main__":
    process_dataset('train')
    process_dataset('test')