import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as pd_tf  # Trick import name to avoid confusion
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import f1_score, roc_auc_score
import gc
import os

# 0.44

# Tắt warning của Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------------------------------------------------------------------
# 1. CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# -------------------------------------------------------------------------------------
print("📥 Loading Advanced Data...")
# Load lại dữ liệu từ bước features advanced (bước đạt 0.458)
train_df = pd.read_csv('train_advanced.csv')
test_df = pd.read_csv('test_advanced.csv')

DROP_COLS = ['object_id', 'split', 'target', 'SpecType', 'English Translation', 'Z_err']
features = [c for c in train_df.columns if c not in DROP_COLS]

X = train_df[features]
y = train_df['target']
X_test = test_df[features]

# --- QUAN TRỌNG CHO NEURAL NET: XỬ LÝ SẠCH SẼ ---
# 1. Thay thế inf/-inf bằng NaN
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# 2. Fill NaN (NN không chịu được NaN)
# Fill bằng 0 hoặc mean. Với dữ liệu này, fill 0 an toàn hơn vì nhiều feature là "độ lệch"
X = X.fillna(0)
X_test = X_test.fillna(0)

# 3. Scaling (BẮT BUỘC cho NN)
# Dùng StandardScaler để đưa về mean=0, std=1
print("⚖️ Scaling data for Neural Network...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------------------------
# 2. XÂY DỰNG MODEL 1: LIGHTGBM (Best Version from 0.458 run)
# -------------------------------------------------------------------------------------
print("\n🔥 Retraining LightGBM (The Strongest One)...")
# Dùng lại params tốt nhất
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'num_leaves': 40,
    'max_depth': 8,
    'scale_pos_weight': scale_pos_weight,
    'colsample_bytree': 0.6,
    'subsample': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))

# Train LightGBM
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
            callbacks=[lgb.early_stopping(150, verbose=False)])

    lgb_oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    lgb_test_preds += clf.predict_proba(X_test)[:, 1] / 5

print(f"✅ LightGBM AUC: {roc_auc_score(y, lgb_oof):.5f}")

# -------------------------------------------------------------------------------------
# 3. XÂY DỰNG MODEL 2: NEURAL NETWORK (MLP)
# -------------------------------------------------------------------------------------
print("\n🧠 Training Neural Network (MLP)...")


def build_mlp(input_dim):
    model = Sequential()
    # Layer 1: Mở rộng
    model.add(Dense(256, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))  # Chống overfit

    # Layer 2: Thu hẹp
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Layer 3: Cô đọng
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['AUC'])
    return model


nn_oof = np.zeros(len(X))
nn_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X_scaled, y)):
    X_tr, y_tr = X_scaled[train_idx], y.iloc[train_idx]
    X_val, y_val = X_scaled[val_idx], y.iloc[val_idx]

    model = build_mlp(X_tr.shape[1])

    # Callbacks quan trọng
    es = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True, verbose=0)
    # Thêm mode='max'
    rlr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    # Train NN
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=128, callbacks=[es, rlr], verbose=0)

    # Predict (NN trả về mảng 2D (n,1), cần flatten)
    val_preds = model.predict(X_val, verbose=0).flatten()
    nn_oof[val_idx] = val_preds

    test_preds = model.predict(X_test_scaled, verbose=0).flatten()
    nn_test_preds += test_preds / 5

    print(f"   -> NN Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.5f}")

print(f"✅ Neural Network AUC: {roc_auc_score(y, nn_oof):.5f}")

# -------------------------------------------------------------------------------------
# 4. HỢP NHẤT THÔNG MINH (SMART BLENDING)
# -------------------------------------------------------------------------------------
print("\n⚗️ Optimizing Ensemble...")

best_score = 0
best_w = 0
best_t = 0.5

# Grid Search trọng số blend
# Chúng ta ưu tiên LightGBM (vì nó đã chứng minh hiệu quả), NN chỉ đóng vai trò sửa lỗi phụ trợ
# Nên weight của LGBM sẽ chạy từ 0.5 đến 1.0
weights = np.arange(0.5, 1.0, 0.05)
thresholds = np.arange(0.1, 0.9, 0.01)

for w in weights:
    blend_oof = w * lgb_oof + (1 - w) * nn_oof
    for t in thresholds:
        score = f1_score(y, (blend_oof >= t).astype(int))
        if score > best_score:
            best_score = score
            best_w = w
            best_t = t

print(f"🏆 BEST ENSEMBLE CONFIG:")
print(f"   Mix: {best_w:.2f} LightGBM + {1 - best_w:.2f} NeuralNet")
print(f"   Threshold: {best_t:.2f}")
print(f"   Max F1-Score (CV): {best_score:.5f}")

# -------------------------------------------------------------------------------------
# 5. SUBMISSION
# -------------------------------------------------------------------------------------
final_test_preds = best_w * lgb_test_preds + (1 - best_w) * nn_test_preds
final_labels = (final_test_preds >= best_t).astype(int)

sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_labels})
sample = pd.read_csv('data/sample_submission.csv')
sub = sample[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
sub['prediction'] = sub['prediction'].astype(int)

sub.to_csv('submission_lgbm_nn.csv', index=False)
print("\n💾 Saved: submission_lgbm_nn.csv")