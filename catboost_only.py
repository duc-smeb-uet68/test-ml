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

# 0.4580


train_df_adv = pd.read_csv('processed_data/processed_train_features.csv')
test_df_adv = pd.read_csv('processed_data/processed_test_features.csv')

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
cat_params = {
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 6,                 # CatBoost dùng Symmetric Tree, depth 6-8 là chuẩn
    'l2_leaf_reg': 5,           # Tương đương reg_lambda
    'subsample': 0.8,
    'colsample_bylevel': 0.6,   # Tương đương colsample_bytree
    'scale_pos_weight': scale_pos_weight,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'bootstrap_type': 'Bernoulli',
    'random_seed': 42,
    'allow_writing_files': False, # Tắt tạo file log rác
    'verbose': False,
    'task_type': 'CPU' # Đổi thành 'GPU' nếu dùng Colab/Kaggle GPU
}

# K-Fold Loop (Giống bước trước nhưng dùng params mới)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_oof = np.zeros(len(X))
cat_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(**cat_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        verbose=False
    )

    cat_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    cat_test_preds += model.predict_proba(X_test)[:, 1] / 5
    print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, cat_oof[val_idx]):.5f}")

oof_df = pd.DataFrame()
oof_df['object_id'] = train_df_adv['object_id']
oof_df['target'] = y
oof_df['cat_prob'] = cat_oof # Biến chứa xác suất OOF mày đã tính trong vòng lặp

# Lưu Test predictions
pred_df = pd.DataFrame()
pred_df['object_id'] = test_df_adv['object_id']
pred_df['cat_prob'] = cat_test_preds # Biến chứa xác suất Test mày đã tính

# Xuất file
oof_df.to_csv('oof_catboost.csv', index=False)
pred_df.to_csv('pred_catboost.csv', index=False)
print("Đã lưu oof_catboost.csv và pred_catboost.csv")

# Tìm Best Threshold (Rất quan trọng)
thresholds = np.arange(0.1, 1.0, 0.01)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    f1 = f1_score(y, (cat_oof >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n🏆 New CV F1-Score: {best_f1:.5f} at threshold {best_thresh:.2f}")

# Tạo Submission
sub = pd.DataFrame({'object_id': test_df_adv['object_id'], 'prediction': (cat_test_preds >= best_thresh).astype(int)})
sample = pd.read_csv('data/sample_submission.csv')
sub = sample[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
sub['prediction'] = sub['prediction'].astype(int)
sub.to_csv('submit/submission_CAT.csv', index=False)
print("💾 Done: submit/submission_CAT.csv")