import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import re  # <--- Nhớ import thư viện này ở đầu file

# 0.


train_df_adv = pd.read_csv('processed_data/processed_train_features.csv')
test_df_adv = pd.read_csv('processed_data/processed_test_features.csv')

print("🧹 Đang làm sạch tên cột cho XGBoost...")
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

train_df_adv.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_df_adv.columns.values]
test_df_adv.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test_df_adv.columns.values]

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
xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'max_depth': 6,             # Tương đương num_leaves=35 (2^6 = 64, nhưng XGB thường cần depth nhỏ hơn)
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 30,     # Tương đương min_child_samples
    'gamma': 0.5,               # Tương tự reg_alpha/lambda (Lagrangian multiplier)
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'scale_pos_weight': scale_pos_weight,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',      # Quan trọng để chạy nhanh
    'device': 'cuda',           # Dùng GPU nếu có (đổi thành 'cpu' nếu không có)
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds':100
}

# K-Fold Loop (Giống bước trước nhưng dùng params mới)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(X))
xgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]


    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    xgb_test_preds += model.predict_proba(X_test)[:, 1] / 5
    print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, xgb_oof[val_idx]):.5f}")

oof_df = pd.DataFrame()
oof_df['object_id'] = train_df_adv['object_id']
oof_df['target'] = y
oof_df['xgb_prob'] = xgb_oof # Tên biến trong code XGBoost của mày

pred_df = pd.DataFrame()
pred_df['object_id'] = test_df_adv['object_id']
pred_df['xgb_prob'] = xgb_test_preds # Tên biến test

oof_df.to_csv('oof_xgb.csv', index=False)
pred_df.to_csv('pred_xgb.csv', index=False)
print("Đã lưu oof_xgb.csv và pred_xgb.csv")

# Tìm Best Threshold (Rất quan trọng)
thresholds = np.arange(0.1, 1.0, 0.01)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    f1 = f1_score(y, (xgb_oof >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n🏆 New CV F1-Score: {best_f1:.5f} at threshold {best_thresh:.2f}")

# Tạo Submission
sub = pd.DataFrame({'object_id': test_df_adv['object_id'], 'prediction': (xgb_test_preds >= best_thresh).astype(int)})
sample = pd.read_csv('data/sample_submission.csv')
sub = sample[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
sub['prediction'] = sub['prediction'].astype(int)
sub.to_csv('submit/submission_XG.csv', index=False)
print("💾 Done: submit/submission_XG.csv")