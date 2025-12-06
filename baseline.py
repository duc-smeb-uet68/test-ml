import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

# 0.4203 f1

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------------------
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# -------------------------------------------------------------------------------------
# Load dữ liệu đã xử lý ở bước trước
print("📥 Đang load dữ liệu features...")
train_df = pd.read_csv('processed_data/processed_train_features.csv')
test_df = pd.read_csv('processed_data/processed_test_features.csv')

# Xác định các cột không dùng cho training
# Z_err bị drop vì lý do Covariate Shift (Train=NaN, Test=Value)
DROP_COLS = ['object_id', 'split', 'target', 'SpecType', 'English Translation', 'Z_err']

# Xác định features
features = [col for col in train_df.columns if col not in DROP_COLS]
print(f"✅ Số lượng features sử dụng: {len(features)}")

X = train_df[features]
y = train_df['target']
X_test = test_df[features]

# Tính toán scale_pos_weight cho Imbalance
# Công thức: Tổng số mẫu Negative / Tổng số mẫu Positive
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"⚖️ Tỷ lệ Imbalance: 1 TDE : {neg_count / pos_count:.2f} Non-TDE")
print(f"⚖️ Sử dụng scale_pos_weight = {scale_pos_weight:.2f}")

# -------------------------------------------------------------------------------------
# 2. THIẾT LẬP MÔ HÌNH LIGHTGBM
# -------------------------------------------------------------------------------------
# Hyperparameters tối ưu cho bài toán mất cân bằng và dữ liệu nhiễu
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',  # Dùng AUC để training ổn định hơn logloss
    'boosting_type': 'gbdt',
    'n_estimators': 2000,  # Số lượng cây tối đa
    'learning_rate': 0.03,  # Learning rate nhỏ để hội tụ tốt
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,  # Row subsampling
    'colsample_bytree': 0.7,  # Feature subsampling
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'scale_pos_weight': scale_pos_weight,  # Xử lý mất cân bằng
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# -------------------------------------------------------------------------------------
# 3. VÒNG LẶP TRAINING (STRATIFIED K-FOLD)
# -------------------------------------------------------------------------------------
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Mảng lưu kết quả
oof_preds = np.zeros(len(X))  # Out-of-Fold predictions (để validate)
test_preds = np.zeros(len(X_test))  # Test set predictions (để submit)
feature_importance_df = pd.DataFrame()

print(f"\n🚀 Bắt đầu training {FOLDS}-Fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")

    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    # Khởi tạo mô hình
    clf = lgb.LGBMClassifier(**lgb_params)

    # Train với Early Stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    clf.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
        eval_names=['train', 'valid'],
        eval_metric='auc',
        callbacks=callbacks
    )

    # Predict (lấy xác suất class 1)
    val_probs = clf.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_probs

    # Predict Test set (cộng dồn để lấy trung bình sau này)
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds += test_probs / FOLDS

    # Log score cơ bản của fold (AUC)
    fold_auc = roc_auc_score(y_val_fold, val_probs)
    print(f"Fold {fold + 1} AUC: {fold_auc:.5f} | Best Iteration: {clf.best_iteration_}")

    # Feature Importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = clf.feature_importances_
    fold_importance["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

    # Dọn dẹp RAM
    del X_train_fold, X_val_fold, y_train_fold, y_val_fold, clf
    gc.collect()

# -------------------------------------------------------------------------------------
# 4. TỐI ƯU HÓA NGƯỠNG (THRESHOLD OPTIMIZATION)
# -------------------------------------------------------------------------------------
print("\n🔍 Đang tìm Threshold tối ưu cho F1-Score...")

thresholds = np.arange(0.01, 1.00, 0.01)
f1_scores = []
precisions = []
recalls = []

best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    # Chuyển xác suất thành nhãn 0/1 dựa trên ngưỡng
    y_pred_bin = (oof_preds >= thresh).astype(int)

    f1 = f1_score(y, y_pred_bin)
    f1_scores.append(f1)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        # Lưu lại precision/recall tại điểm tốt nhất để tham khảo
        best_prec = precision_score(y, y_pred_bin)
        best_rec = recall_score(y, y_pred_bin)

print(f"\n🏆 KẾT QUẢ VALIDATION:")
print(f"Best Threshold: {best_thresh:.2f}")
print(f"Max F1-Score (OOF): {best_f1:.5f}")
print(f"Precision: {best_prec:.5f}")
print(f"Recall: {best_rec:.5f}")
print(f"Overall AUC: {roc_auc_score(y, oof_preds):.5f}")

# Vẽ biểu đồ F1 theo Threshold
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.axvline(best_thresh, color='r', linestyle='--', label=f'Best Threshold {best_thresh:.2f}')
plt.title('F1 Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------------------
# 5. TẠO FILE SUBMISSION
# -------------------------------------------------------------------------------------
# Áp dụng ngưỡng tối ưu vào tập Test
final_test_predictions = (test_preds >= best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_df['object_id'],
    'prediction': final_test_predictions
})

# Kiểm tra định dạng sample submission
sample_sub = pd.read_csv('data/sample_submission.csv')
# Đảm bảo thứ tự object_id giống sample (quan trọng để tránh lỗi chấm điểm)
submission = sample_sub[['object_id']].merge(submission, on='object_id', how='left')
# Fill 0 nếu có lỗi null (phòng hờ)
submission['prediction'] = submission['prediction'].fillna(0).astype(int)

submission.to_csv('submit/submission.csv', index=False)
print("\n💾 Đã lưu file: submission.csv")

# -------------------------------------------------------------------------------------
# 6. HIỂN THỊ FEATURE IMPORTANCE
# -------------------------------------------------------------------------------------
print("\n📊 Top 20 Features quan trọng nhất:")
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:20].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(10, 8))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.show()