import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

# 0.4580

# ======================================================================================
# 1. HÀM TÍNH TOÁN SHAPE FEATURES (ĐƯỢC TỐI ƯU HÓA VECTOR)
# ======================================================================================

def calculate_shape_features(df_group):
    """
    Hàm này tính toán các đặc trưng hình học cho một nhóm (1 object - 1 filter).
    Được viết để apply lên GroupBy object.
    """
    # Lấy dữ liệu flux và time
    flux = df_group['Flux_corr'].values
    time = df_group['Time (MJD)'].values

    if len(flux) < 3:
        # Nếu quá ít điểm dữ liệu, trả về mặc định
        return pd.Series({
            'rise_time': 0, 'decay_time': 0,
            'fwd_half_time': 0, 'peak_to_median': 0
        })

    # Tìm đỉnh (Peak)
    idx_max = np.argmax(flux)
    t_max = time[idx_max]
    f_max = flux[idx_max]

    # 1. Rise Time: Thời gian từ lần đầu tiên đạt 20% Flux max đến khi đạt đỉnh
    # Lọc các điểm trước đỉnh
    mask_pre = time < t_max
    if np.any(mask_pre):
        # Tìm điểm gần nhất trước đỉnh mà flux < 0.2 * max
        threshold = 0.2 * f_max
        # Các điểm thỏa mãn
        candidates = np.where((flux < threshold) & mask_pre)[0]
        if len(candidates) > 0:
            t_start = time[candidates[-1]]  # Lấy điểm gần đỉnh nhất
            rise_time = t_max - t_start
        else:
            # Nếu không có điểm nào < 20%, lấy điểm đầu tiên
            rise_time = t_max - time[0]
    else:
        rise_time = 0

    # 2. Decay Time: Thời gian từ đỉnh về 20% Flux max
    mask_post = time > t_max
    if np.any(mask_post):
        threshold = 0.2 * f_max
        candidates = np.where((flux < threshold) & mask_post)[0]
        if len(candidates) > 0:
            t_end = time[candidates[0]]  # Lấy điểm đầu tiên thỏa mãn sau đỉnh
            decay_time = t_end - t_max
        else:
            decay_time = time[-1] - t_max
    else:
        decay_time = 0

    # 3. Peak to Median (Độ nhọn của sự kiện)
    median_flux = np.median(flux)
    peak_to_med = f_max / (median_flux + 1e-6)  # Tránh chia cho 0

    return pd.Series({
        'rise_time': rise_time,
        'decay_time': decay_time,
        'peak_to_median': peak_to_med
    })


# ======================================================================================
# 2. PIPELINE XỬ LÝ MỚI (FIXED - SỬA LỖI SERIES ATTRIBUTE)
# ======================================================================================

def process_dataset_advanced(log_path, is_train=True):
    print(f"🔄 Advanced Processing (Fixed): {'TRAIN' if is_train else 'TEST'}...")
    df_log = pd.read_csv(log_path)

    # Chỉ định các band quan trọng cho TDE (u, g, r, i)
    target_bands = ['u', 'g', 'r', 'i']

    unique_splits = df_log['split'].unique()
    all_features = []

    # Cấu hình Extinction
    EXTINCTION_COEFFS = {'u': 4.81, 'g': 3.64, 'r': 2.70, 'i': 2.06, 'z': 1.58, 'y': 1.31}

    for split_name in tqdm(unique_splits):
        lc_path = os.path.join('data', split_name,
                               'train_full_lightcurves.csv' if is_train else 'test_full_lightcurves.csv')
        if not os.path.exists(lc_path): continue

        df_lc = pd.read_csv(lc_path)

        # --- DE-EXTINCTION ---
        df_meta_split = df_log[df_log['split'] == split_name]
        df_lc = df_lc.merge(df_meta_split[['object_id', 'EBV']], on='object_id', how='left')
        df_lc['R_factor'] = df_lc['Filter'].map(EXTINCTION_COEFFS)
        correction = 10 ** (0.4 * df_lc['R_factor'] * df_lc['EBV'])
        df_lc['Flux_corr'] = df_lc['Flux'] * correction

        # --- BASIC STATS ---
        basic_aggs = df_lc.groupby(['object_id', 'Filter'])['Flux_corr'].agg(['mean', 'max', 'std', 'count']).unstack()
        basic_aggs.columns = [f"{c[1]}_{c[0]}" for c in basic_aggs.columns]

        # --- ADVANCED SHAPE FEATURES (SỬA LỖI Ở ĐÂY) ---
        df_shape_input = df_lc[df_lc['Filter'].isin(target_bands)]

        shape_dfs = []
        for band in target_bands:
            band_data = df_shape_input[df_shape_input['Filter'] == band]
            if band_data.empty: continue

            # Group by object và apply
            s_feat = band_data.groupby('object_id').apply(calculate_shape_features)

            # --- [FIX QUAN TRỌNG] ---
            # Nếu Pandas trả về Series (MultiIndex), unstack để thành DataFrame
            if isinstance(s_feat, pd.Series):
                s_feat = s_feat.unstack()
            # ------------------------

            s_feat.columns = [f"{band}_{c}" for c in s_feat.columns]
            shape_dfs.append(s_feat)

        if shape_dfs:
            shape_features = pd.concat(shape_dfs, axis=1)
        else:
            shape_features = pd.DataFrame()

        # Merge Basic và Shape
        combined = basic_aggs.join(shape_features, how='left')

        # Reset index để lấy object_id
        combined = combined.reset_index()
        all_features.append(combined)

        del df_lc, shape_dfs
        gc.collect()

    # Nối tất cả features
    full_feats = pd.concat(all_features, axis=0).fillna(0)

    # --- COLOR RATIOS ---
    full_feats['u_g_ratio'] = full_feats['u_max'] / (full_feats['g_max'] + 1e-5)
    full_feats['g_r_ratio'] = full_feats['g_max'] / (full_feats['r_max'] + 1e-5)
    full_feats['blue_slope'] = (full_feats['u_max'] - full_feats['g_max']) / (
                full_feats['u_std'] + full_feats['g_std'] + 1e-5)

    # Merge lại với Log gốc
    final_df = df_log.merge(full_feats, on='object_id', how='left')
    return final_df


# ======================================================================================
# 3. CHẠY VÀ TRAIN LẠI
# ======================================================================================

# B1: Tạo Features Mới
# Chú ý: Bước này sẽ mất thời gian hơn bước trước do hàm calculate_shape_features
# Nếu máy bạn yếu, hãy thử trên 1 split trước
train_df_adv = process_dataset_advanced('data/train_log.csv', is_train=True)
test_df_adv = process_dataset_advanced('data/test_log.csv', is_train=False)

# Lưu lại ngay lập tức
train_df_adv.to_csv('processed/train_advanced.csv', index=False)
test_df_adv.to_csv('processed/test_advanced.csv', index=False)

# B2: Train LightGBM với thông số tinh chỉnh hơn
print("🚀 Training Advanced Model...")

features = [c for c in train_df_adv.columns if
            c not in ['object_id', 'split', 'target', 'SpecType', 'English Translation', 'Z_err']]
X = train_df_adv[features]
y = train_df_adv['target']
X_test = test_df_adv[features]

# Class Weight cũ
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Params mạnh hơn (Tăng độ sâu một chút để bắt pattern phức tạp)
lgb_params_adv = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,  # Tăng số lượng cây
    'learning_rate': 0.02,  # Giảm LR để học kỹ hơn
    'num_leaves': 40,  # Tăng nhẹ độ phức tạp
    'max_depth': 8,  # Giới hạn độ sâu để tránh overfit với dữ liệu nhiễu
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.6,  # Giảm feature sampling để cây đa dạng hơn
    'reg_alpha': 0.5,  # Tăng L1 Regularization (quan trọng khi thêm nhiều feature)
    'reg_lambda': 0.5,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# K-Fold Loop (Giống bước trước nhưng dùng params mới)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    clf = lgb.LGBMClassifier(**lgb_params_adv)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )

    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    test_preds += clf.predict_proba(X_test)[:, 1] / 5
    print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.5f}")

# Tìm Best Threshold (Rất quan trọng)
thresholds = np.arange(0.1, 0.95, 0.01)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    f1 = f1_score(y, (oof_preds >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n🏆 New CV F1-Score: {best_f1:.5f} at threshold {best_thresh:.2f}")

# Tạo Submission
sub = pd.DataFrame({'object_id': test_df_adv['object_id'], 'prediction': (test_preds >= best_thresh).astype(int)})
sample = pd.read_csv('data/sample_submission.csv')
sub = sample[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
sub['prediction'] = sub['prediction'].astype(int)
sub.to_csv('submit/submission_advanced.csv', index=False)
print("💾 Done: submit/submission_advanced.csv")