import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# 1. Load các file OOF (Validation)
oof_cat = pd.read_csv('oof_catboost.csv')
oof_lgbm = pd.read_csv('oof_lgbm.csv')
oof_xgb = pd.read_csv('oof_xgb.csv')

# Merge lại thành 1 bảng duy nhất theo object_id
# Lưu ý: Thứ tự dòng có thể khác nhau nếu mày không sort, nên merge là an toàn nhất
df_oof = oof_cat.merge(oof_lgbm[['object_id', 'lgbm_prob']], on='object_id')
df_oof = df_oof.merge(oof_xgb[['object_id', 'xgb_prob']], on='object_id')

y_true = df_oof['target'].values

# 2. Load các file Test Prediction
pred_cat = pd.read_csv('pred_catboost.csv')
pred_lgbm = pd.read_csv('pred_lgbm.csv')
pred_xgb = pd.read_csv('pred_xgb.csv')

# Merge test
df_test = pred_cat.merge(pred_lgbm[['object_id', 'lgbm_prob']], on='object_id')
df_test = df_test.merge(pred_xgb[['object_id', 'xgb_prob']], on='object_id')

print("Load data xong. Bắt đầu tìm trọng số tối ưu...")

# 3. Grid Search đơn giản để tìm trọng số (Weights)
best_score = 0
best_weights = (0, 0, 0)
best_threshold = 0.5

# Thử các tỉ lệ khác nhau. Ví dụ: w1 cho cat, w2 cho lgbm, w3 cho xgb
# Bước nhảy 0.1
weights_to_try = []
for i in range(11):
    for j in range(11):
        for k in range(11):
            if i + j + k == 10:  # Tổng phải bằng 10 (tức là 1.0)
                weights_to_try.append((i / 10, j / 10, k / 10))

for w_cat, w_lgbm, w_xgb in weights_to_try:
    # Tính xác suất tổng hợp trên OOF
    blend_prob = (w_cat * df_oof['cat_prob'] +
                  w_lgbm * df_oof['lgbm_prob'] +
                  w_xgb * df_oof['xgb_prob'])

    # Tìm threshold tốt nhất cho bộ weight này
    # Mẹo: Chỉ cần search sơ qua để đánh giá weight
    for thresh in np.arange(0.2, 0.8, 0.05):
        pred_label = (blend_prob >= thresh).astype(int)
        score = f1_score(y_true, pred_label)

        if score > best_score:
            best_score = score
            best_weights = (w_cat, w_lgbm, w_xgb)
            best_threshold = thresh

print("-" * 30)
print(f"✅ TÌM THẤY TRỌNG SỐ TỐI ƯU!")
print(f"CatBoost Weight: {best_weights[0]}")
print(f"LightGBM Weight: {best_weights[1]}")
print(f"XGBoost Weight : {best_weights[2]}")
print(f"Best Threshold : {best_threshold}")
print(f"Best OOF F1    : {best_score:.5f}")
print("-" * 30)

# 4. Áp dụng trọng số và threshold tìm được vào tập TEST
print("Đang tạo file submission...")

final_test_prob = (best_weights[0] * df_test['cat_prob'] +
                   best_weights[1] * df_test['lgbm_prob'] +
                   best_weights[2] * df_test['xgb_prob'])

# Chuyển xác suất thành nhãn 0/1 dựa trên best_threshold
final_preds = (final_test_prob >= best_threshold).astype(int)

# Tạo file submission
submission = pd.DataFrame({
    'object_id': df_test['object_id'],
    'target': final_preds
})

submission.to_csv('submission_ensemble_optimized.csv', index=False)
print("🎉 Xong! File kết quả: submission_ensemble_optimized.csv")