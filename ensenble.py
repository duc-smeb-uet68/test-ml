import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import gc

# 0.4311

# -------------------------------------------------------------------------------------
# 1. LOAD DATA (Dữ liệu đã tạo ở bước trước)
# -------------------------------------------------------------------------------------
print("📥 Loading Advanced Data...")
train_df = pd.read_csv('train_advanced.csv')
test_df = pd.read_csv('test_advanced.csv')

# Loại bỏ các cột không train
DROP_COLS = ['object_id', 'split', 'target', 'SpecType', 'English Translation', 'Z_err']
features = [c for c in train_df.columns if c not in DROP_COLS]

X = train_df[features]
y = train_df['target']
X_test = test_df[features]

# Xử lý NaN cho CatBoost (CatBoost thích số cực nhỏ thay vì NaN đôi khi)
X = X.fillna(-999)
X_test = X_test.fillna(-999)

print(f"Features: {len(features)}")
print(f"Train shape: {X.shape}")

# Tỷ lệ scale pos weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# -------------------------------------------------------------------------------------
# 2. MODEL 1: LIGHTGBM (Retrain)
# -------------------------------------------------------------------------------------
print("\n🔥 Training LightGBM...")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'num_leaves': 40,
    'max_depth': 8,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)])

    lgb_oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    lgb_test_preds += clf.predict_proba(X_test)[:, 1] / 5

print(f"✅ LightGBM OOF AUC: {roc_auc_score(y, lgb_oof):.5f}")

# -------------------------------------------------------------------------------------
# 3. MODEL 2: CATBOOST (New)
# -------------------------------------------------------------------------------------
print("\n🐱 Training CatBoost...")
# CatBoost tự xử lý imbalance tốt qua tham số auto_class_weights hoặc scale_pos_weight
cat_params = {
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'scale_pos_weight': scale_pos_weight,
    'od_type': 'Iter',  # Overfitting Detector
    'od_wait': 100,
    'random_seed': 42,
    'verbose': False,
    'allow_writing_files': False
}

cat_oof = np.zeros(len(X))
cat_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    clf = CatBoostClassifier(**cat_params)
    clf.fit(train_pool, eval_set=val_pool)

    cat_oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    cat_test_preds += clf.predict_proba(X_test)[:, 1] / 5
    print(f"   -> Fold {fold + 1} done.")

print(f"✅ CatBoost OOF AUC: {roc_auc_score(y, cat_oof):.5f}")

# -------------------------------------------------------------------------------------
# 4. ENSEMBLE (BLENDING) & OPTIMIZATION
# -------------------------------------------------------------------------------------
print("\n⚗️ Blending Models...")

# Thử nghiệm các tỷ lệ blend khác nhau trên tập OOF
best_blend_score = 0
best_w = 0.5
best_thresh = 0.5

# Grid search tỷ lệ trọng số (w) và threshold
for w in np.arange(0.1, 1.0, 0.1):
    blended_oof = (w * lgb_oof) + ((1 - w) * cat_oof)

    # Tìm threshold tốt nhất cho tỷ lệ này
    for t in np.arange(0.1, 0.95, 0.01):
        score = f1_score(y, (blended_oof >= t).astype(int))
        if score > best_blend_score:
            best_blend_score = score
            best_w = w
            best_thresh = t

print(f"🏆 BEST ENSEMBLE RESULT:")
print(f"Weight: {best_w:.1f} LightGBM + {1 - best_w:.1f} CatBoost")
print(f"Threshold: {best_thresh:.2f}")
print(f"Max F1-Score (OOF): {best_blend_score:.5f}")

# -------------------------------------------------------------------------------------
# 5. SUBMISSION
# -------------------------------------------------------------------------------------
# Áp dụng tỷ lệ tối ưu vào tập Test
final_test_preds = (best_w * lgb_test_preds) + ((1 - best_w) * cat_test_preds)
final_labels = (final_test_preds >= best_thresh).astype(int)

sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_labels})
sample = pd.read_csv('data/sample_submission.csv')
sub = sample[['object_id']].merge(sub, on='object_id', how='left').fillna(0)
sub['prediction'] = sub['prediction'].astype(int)

sub.to_csv('submission_ensemble.csv', index=False)
print("\n💾 Done: submission_ensemble.csv saved!")