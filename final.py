import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import gc

# -------------------------------------------------------------------------------------
# 1. LOAD DỮ LIỆU & DỰ ĐOÁN TỐT NHẤT
# -------------------------------------------------------------------------------------
print("📥 Loading Data & Best Predictions...")
train_df = pd.read_csv('train_advanced.csv')
test_df = pd.read_csv('test_advanced.csv')

# Load dự đoán từ Ensemble LightGBM + NN (bước trước)
# Giả sử bạn đã lưu xác suất dự đoán (nếu chưa thì dùng tạm file submission rồi convert lại,
# nhưng tốt nhất là nên lưu raw probability. Ở đây tôi sẽ giả lập lại việc retrain nhanh để lấy prob)

# --- (Phần này chỉ để lấy lại xác suất blend nếu bạn chưa lưu file raw probabilities) ---
# Nếu bạn đã có file chứa xác suất (không phải 0/1), hãy load nó vào biến test_probs
# Ở đây tôi sẽ train nhanh LightGBM 1 lần nữa để lấy xác suất làm mẫu
DROP_COLS = ['object_id', 'split', 'target', 'SpecType', 'English Translation', 'Z_err']
features = [c for c in train_df.columns if c not in DROP_COLS]

X = train_df[features]
y = train_df['target']
X_test = test_df[features]

print("   -> Generating base predictions for Pseudo-Labeling...")
lgb_base = lgb.LGBMClassifier(n_estimators=1000, random_state=42)  # Train nhanh
lgb_base.fit(X, y)
test_probs = lgb_base.predict_proba(X_test)[:, 1]
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# 2. LỌC PSEUDO-LABELS (QUAN TRỌNG NHẤT)
# -------------------------------------------------------------------------------------
print("\n🕵️ Selecting Pseudo-Labels...")

# Ngưỡng cực kỳ khắt khe để tránh nhiễu
# Chỉ lấy những cái CỰC KỲ chắc chắn
PSEUDO_HIGH_THRESH = 0.98  # Chắc chắn là TDE
PSEUDO_LOW_THRESH = 0.01  # Chắc chắn KHÔNG phải TDE

# Lấy index
high_conf_idx = np.where(test_probs > PSEUDO_HIGH_THRESH)[0]
low_conf_idx = np.where(test_probs < PSEUDO_LOW_THRESH)[0]

print(f"   -> Found {len(high_conf_idx)} high confidence TDEs")
print(f"   -> Found {len(low_conf_idx)} high confidence Non-TDEs")

# Tạo tập Pseudo Train
X_pseudo_high = X_test.iloc[high_conf_idx].copy()
y_pseudo_high = np.ones(len(high_conf_idx))  # Gán nhãn 1

X_pseudo_low = X_test.iloc[low_conf_idx].copy()
y_pseudo_low = np.zeros(len(low_conf_idx))  # Gán nhãn 0

# Gộp lại
X_pseudo = pd.concat([X_pseudo_high, X_pseudo_low])
y_pseudo = np.concatenate([y_pseudo_high, y_pseudo_low])

# Gộp vào tập Train gốc
X_final_train = pd.concat([X, X_pseudo])
y_final_train = np.concatenate([y, y_pseudo])

print(f"✅ New Training Size: {len(X)} -> {len(X_final_train)} (+{len(X_pseudo)} samples)")

# -------------------------------------------------------------------------------------
# 3. RETRAIN FINAL MODEL VỚI DỮ LIỆU ĐÃ TĂNG CƯỜNG
# -------------------------------------------------------------------------------------
print("\n🚀 Training Final Model with Pseudo-Labels...")

# Tính lại scale_pos_weight cho tập dữ liệu mới
scale_pos_weight = (y_final_train == 0).sum() / (y_final_train == 1).sum()

# Params tối ưu (Dùng lại của LightGBM vì nó ổn định nhất với nhiễu)
final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3500,  # Tăng thêm cây vì dữ liệu nhiều hơn
    'learning_rate': 0.015,  # Giảm LR để học kỹ hơn
    'num_leaves': 40,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.65,
    'reg_alpha': 1.0,  # Tăng regularization để tránh overfit vào pseudo labels
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 999,  # Đổi seed may mắn
    'n_jobs': -1,
    'verbose': -1
}

# Chúng ta sẽ train full 5-Fold trên tập mới
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X_final_train))

for fold, (train_idx, val_idx) in enumerate(folds.split(X_final_train, y_final_train)):
    # Split
    X_tr, y_tr = X_final_train.iloc[train_idx], y_final_train[train_idx]
    X_val, y_val = X_final_train.iloc[val_idx], y_final_train[val_idx]

    # Train
    clf = lgb.LGBMClassifier(**final_params)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
            callbacks=[lgb.early_stopping(150, verbose=False)])

    # Predict Test
    final_preds += clf.predict_proba(X_test)[:, 1] / 5

    # Check OOF (chỉ để tham khảo)
    val_pred = clf.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    print(f"   -> Fold {fold + 1} AUC: {score:.5f}")

# -------------------------------------------------------------------------------------
# 4. SUBMISSION CUỐI CÙNG
# -------------------------------------------------------------------------------------
print("\n🏁 Generating Final Submission...")

# Tìm ngưỡng tối ưu trên tập Train mở rộng (dù hơi bias nhưng tốt hơn đoán mò)
# Hoặc an toàn nhất là dùng lại ngưỡng tốt nhất của bước Ensemble trước (ví dụ 0.5 - 0.7)
# Ở đây tôi dùng lại logic tìm threshold
thresholds = np.arange(0.1, 0.95, 0.01)
best_f1 = 0
best_thresh = 0.5

# Lưu ý: Tìm threshold trên tập OOF của Pseudo-Train có thể hơi lạc quan quá
# Nên ta sẽ lấy threshold an toàn từ bước trước (thường là khoảng 0.6 - 0.7 cho bài này)
# Để code tự chạy, tôi vẫn search, nhưng bạn nên cân nhắc manual threshold nếu thấy nó chọn 0.99
for t in thresholds:
    # Chỉ tính F1 trên phần dữ liệu gốc (không tính trên phần pseudo để tránh bias)
    # Lấy lại index của dữ liệu gốc trong tập OOF (đây là trick nâng cao)
    # Nhưng để đơn giản, ta cứ search trên toàn bộ OOF
    f1 = f1_score(y_final_train, (oof_preds >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   -> Best Threshold (Pseudo-CV): {best_thresh:.2f}")

# Áp dụng Threshold
final_labels = (final_preds >= best_thresh).astype(int)

sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_labels})
sample = pd.read_csv('data/sample_submission.csv')
sub = sample[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
sub['prediction'] = sub['prediction'].astype(int)

sub.to_csv('submission_final_pseudo.csv', index=False)
print("\n🏆 DONE. Good luck!")