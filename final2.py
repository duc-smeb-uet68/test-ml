import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import os

# -------------------------------------------------------------------------------------
# 1. RETRAIN MÔ HÌNH TỐT NHẤT (LIGHTGBM + NN) - KHÔNG PSEUDO
# -------------------------------------------------------------------------------------
print("📥 Loading Advanced Data (Back to Safety)...")
train_df = pd.read_csv('train_advanced.csv')
test_df = pd.read_csv('test_advanced.csv')

DROP_COLS = ['object_id', 'split', 'target', 'SpecType', 'English Translation', 'Z_err']
features = [c for c in train_df.columns if c not in DROP_COLS]

X = train_df[features]
y = train_df['target']
X_test = test_df[features]

# --- CHUẨN BỊ CHO NN ---
X_nn = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_nn = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nn)
X_test_scaled = scaler.transform(X_test_nn)

# --- MODEL 1: LIGHTGBM ---
print("\n🔥 Training LightGBM Base...")
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'num_leaves': 40,
    'max_depth': 8,
    'scale_pos_weight': scale_pos_weight,
    'colsample_bytree': 0.6,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_test_preds += clf.predict_proba(X_test)[:, 1] / 5

# --- MODEL 2: NEURAL NET ---
print("\n🧠 Training Neural Network Base...")


def build_mlp(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(128), BatchNormalization(), Activation('relu'), Dropout(0.2),
        Dense(64), BatchNormalization(), Activation('relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model


nn_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X_scaled, y)):
    model = build_mlp(X_scaled.shape[1])
    es = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, verbose=0)

    model.fit(X_scaled[train_idx], y.iloc[train_idx],
              validation_data=(X_scaled[val_idx], y.iloc[val_idx]),
              epochs=50, batch_size=128, callbacks=[es, rlr], verbose=0)

    nn_test_preds += model.predict(X_test_scaled, verbose=0).flatten() / 5

# -------------------------------------------------------------------------------------
# 2. RANK BLENDING (CHIẾN THUẬT QUAN TRỌNG)
# -------------------------------------------------------------------------------------
print("\n⚗️ Performing Rank Blending...")

# Chuyển xác suất thành thứ hạng (Rank)
# Ví dụ: [0.1, 0.9, 0.5] -> Rank [1, 3, 2] (chia cho len để chuẩn hóa về 0-1)
rank_lgb = rankdata(lgb_test_preds) / len(lgb_test_preds)
rank_nn = rankdata(nn_test_preds) / len(nn_test_preds)

# Blend theo Rank (Ưu tiên LightGBM 60% - NN 40%)
final_rank_score = 0.6 * rank_lgb + 0.4 * rank_nn

# -------------------------------------------------------------------------------------
# 3. TOP-K THRESHOLD STRATEGY
# -------------------------------------------------------------------------------------
# Chiến thuật: TDE rất hiếm. Chúng ta sẽ submit các file với số lượng TDE cố định.
# Dựa trên kích thước tập test, giả sử tỉ lệ TDE là p%.
total_test_samples = len(X_test)
print(f"📊 Total Test Samples: {total_test_samples}")

# Tạo 3 mức độ "liều lĩnh"
top_k_candidates = [50, 100, 200]

# Load sample submission để giữ đúng format
sample_sub = pd.read_csv('data/sample_submission.csv')

for k in top_k_candidates:
    # Chọn ra k object có điểm rank cao nhất
    # Sắp xếp index theo điểm giảm dần
    sorted_indices = np.argsort(final_rank_score)[::-1]
    top_k_indices = sorted_indices[:k]

    # Tạo mảng dự đoán toàn số 0
    preds = np.zeros(total_test_samples, dtype=int)
    # Gán 1 cho top k
    preds[top_k_indices] = 1

    # Save
    sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': preds})
    sub = sample_sub[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
    sub['prediction'] = sub['prediction'].astype(int)

    filename = f'submission_rank_top{k}.csv'
    sub.to_csv(filename, index=False)
    print(f"💾 Saved: {filename} (Selected Top {k} objects as TDE)")

print("\n💡 HƯỚNG DẪN SUBMIT:")
print("Bạn hãy submit lần lượt 3 file này.")
print("- Top 50: Precision cao (ít sai) nhưng có thể sót (Recall thấp).")
print("- Top 100: Cân bằng (Khuyên dùng).")
print("- Top 200: Recall cao (bắt hết TDE) nhưng có thể dính nhiều nhiễu.")