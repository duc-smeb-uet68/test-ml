import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings
import gc

# Cấu hình hiển thị và tắt cảnh báo
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# -------------------------------------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG & HẰNG SỐ VẬT LÝ
# -------------------------------------------------------------------------------------
BASE_PATH = 'data'  # Đường dẫn tới thư mục chứa dữ liệu bạn đã upload
TRAIN_LOG_PATH = os.path.join(BASE_PATH, 'train_log.csv')
TEST_LOG_PATH = os.path.join(BASE_PATH, 'test_log.csv')

# Hệ số dập tắt (Extinction coefficients) R_lambda xấp xỉ cho các band của LSST
# Dựa trên Schlafly & Finkbeiner (2011) cho R_V = 3.1
EXTINCTION_COEFFS = {
    'u': 4.81,
    'g': 3.64,
    'r': 2.70,
    'i': 2.06,
    'z': 1.58,
    'y': 1.31
}

# Mapping tên band sang số để xử lý matrix nhanh hơn nếu cần
BAND_MAP = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}


# -------------------------------------------------------------------------------------
# 2. HÀM XỬ LÝ VẬT LÝ & TIỀN XỬ LÝ (PHYSICS & PREPROCESSING)
# -------------------------------------------------------------------------------------

def correct_flux(df_lc, df_meta):
    """
    Thực hiện De-extinction cho Flux dựa trên EBV.
    Công thức: Flux_corr = Flux_obs * 10^(0.4 * A_lambda * EBV)
    Trong đó A_lambda = Coeff_band
    """
    # Merge EBV vào lightcurve
    df_lc = df_lc.merge(df_meta[['object_id', 'EBV']], on='object_id', how='left')

    # Tạo cột hệ số R tương ứng với band
    df_lc['R_factor'] = df_lc['Filter'].map(EXTINCTION_COEFFS)

    # Tính Flux đã hiệu chỉnh
    # Lưu ý: Flux gốc có thể âm, việc nhân hệ số dương không làm thay đổi dấu
    correction_factor = 10 ** (0.4 * df_lc['R_factor'] * df_lc['EBV'])
    df_lc['Flux_corr'] = df_lc['Flux'] * correction_factor

    # Tính lại sai số Flux (Flux Error cũng bị scale tương ứng)
    df_lc['Flux_err_corr'] = df_lc['Flux_err'] * correction_factor

    return df_lc.drop(columns=['EBV', 'R_factor'])


# -------------------------------------------------------------------------------------
# 3. CORE FEATURE ENGINEERING (TRÍCH XUẤT ĐẶC TRƯNG)
# -------------------------------------------------------------------------------------

def extract_features_group(group):
    """
    Hàm này xử lý một nhóm (một object_id) và trả về một Series các đặc trưng.
    Tuy nhiên, để tối ưu tốc độ, chúng ta sẽ dùng Aggregation của Pandas thay vì apply từng dòng.
    Hàm này chỉ dùng để minh họa logic nếu cần debug.
    Chúng ta sẽ dùng vectorization ở hàm main_extraction bên dưới.
    """
    pass


def aggregate_features(df_lc):
    """
    Tính toán các đặc trưng thống kê cho từng object, từng band.
    Input: DataFrame Lightcurves (đã correct flux)
    Output: DataFrame Features (aggregated)
    """

    # 1. Các đặc trưng cơ bản theo từng Filter
    aggs = {
        'Flux_corr': ['min', 'max', 'mean', 'median', 'std'],
        'Flux_err_corr': ['mean'],
        'Time (MJD)': ['min', 'max', 'count']  # count là số lượng quan sát
    }

    # Group by Object và Filter
    features_per_band = df_lc.groupby(['object_id', 'Filter']).agg(aggs)

    # Làm phẳng MultiIndex columns
    features_per_band.columns = ['_'.join(col).strip() for col in features_per_band.columns.values]
    features_per_band = features_per_band.reset_index()

    # Pivot table để đưa Filter lên thành cột (ví dụ: u_Flux_mean, g_Flux_mean...)
    features_wide = features_per_band.pivot(index='object_id', columns='Filter')

    # Làm phẳng lại cột sau khi pivot
    features_wide.columns = [f"{col[1]}_{col[0]}" for col in features_wide.columns]

    # 2. Tính toán thêm các đặc trưng phức tạp hơn (Vectorized)
    # Skew & Kurtosis (Cần cẩn thận với số lượng mẫu ít)
    skew_kurt = df_lc.groupby(['object_id', 'Filter'])['Flux_corr'].agg(
        skew=lambda x: skew(x, nan_policy='omit') if len(x) > 2 else 0,
        kurt=lambda x: kurtosis(x, nan_policy='omit') if len(x) > 2 else 0
    ).reset_index().pivot(index='object_id', columns='Filter')
    skew_kurt.columns = [f"{col[1]}_Flux_{col[0]}" for col in skew_kurt.columns]

    # Merge lại
    final_features = pd.concat([features_wide, skew_kurt], axis=1)

    return final_features


def calculate_advanced_features(features_df):
    """
    Tính toán các đặc trưng kết hợp giữa các bands (Colors, Ratios)
    Dựa trên DataFrame đã pivot (wide format).
    """
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    # 1. Colors (Hiệu độ sáng giữa các band liền kề - đại diện nhiệt độ)
    # Dùng Flux_mean hoặc Flux_max để tính color
    for i in range(len(bands) - 1):
        b1 = bands[i]
        b2 = bands[i + 1]
        # Color dựa trên Mean Flux
        features_df[f'{b1}_{b2}_flux_diff_mean'] = features_df[f'{b1}_Flux_corr_mean'] - features_df[
            f'{b2}_Flux_corr_mean']
        # Color dựa trên Max Flux (đỉnh của sự kiện)
        features_df[f'{b1}_{b2}_flux_diff_max'] = features_df[f'{b1}_Flux_corr_max'] - features_df[
            f'{b2}_Flux_corr_max']

    # 2. Amplitude (Biên độ dao động)
    for b in bands:
        if f'{b}_Flux_corr_max' in features_df.columns:
            features_df[f'{b}_amplitude'] = features_df[f'{b}_Flux_corr_max'] - features_df[f'{b}_Flux_corr_min']

    # 3. Global Time Features (Thời gian quan sát)
    # Lấy max(Time_max) - min(Time_min) trên tất cả các band
    time_max_cols = [c for c in features_df.columns if 'Time (MJD)_max' in c]
    time_min_cols = [c for c in features_df.columns if 'Time (MJD)_min' in c]

    # Vì mỗi band có thể quan sát thời điểm khác nhau, ta lấy min/max tổng thể
    # Fillna để tránh lỗi nếu object thiếu band
    features_df['global_start_time'] = features_df[time_min_cols].min(axis=1)
    features_df['global_end_time'] = features_df[time_max_cols].max(axis=1)
    features_df['duration'] = features_df['global_end_time'] - features_df['global_start_time']

    return features_df


# -------------------------------------------------------------------------------------
# 4. PIPELINE CHÍNH (MAIN EXECUTION)
# -------------------------------------------------------------------------------------

def process_dataset(log_path, is_train=True):
    print(f"🔄 Đang xử lý tập dữ liệu: {'TRAIN' if is_train else 'TEST'}...")

    # Load Metadata
    df_log = pd.read_csv(log_path)

    # Lấy danh sách các splits duy nhất
    unique_splits = df_log['split'].unique()

    all_features_list = []

    # Duyệt qua từng split folder để đọc lightcurves (Tiết kiệm RAM)
    pbar = tqdm(unique_splits)
    for split_name in pbar:
        pbar.set_description(f"Processing {split_name}")

        # Đường dẫn tới file lightcurve của split này
        lc_path = os.path.join(BASE_PATH, split_name,
                               'train_full_lightcurves.csv' if is_train else 'test_full_lightcurves.csv')

        if not os.path.exists(lc_path):
            print(f"⚠️ Không tìm thấy file: {lc_path}, bỏ qua.")
            continue

        # Đọc file LC
        df_lc_split = pd.read_csv(lc_path)

        # Lấy metadata tương ứng với các object trong split này để De-extinct
        objects_in_split = df_log[df_log['split'] == split_name]

        # Tiền xử lý Vật lý (De-extinction)
        df_lc_split = correct_flux(df_lc_split, objects_in_split)

        # Trích xuất đặc trưng thống kê (Aggregation)
        split_features = aggregate_features(df_lc_split)

        # Gom kết quả
        all_features_list.append(split_features)

        # Dọn dẹp RAM
        del df_lc_split
        gc.collect()

    # Nối tất cả các splits
    full_features_df = pd.concat(all_features_list, axis=0)

    # Tính các đặc trưng nâng cao (Advanced Features)
    full_features_df = calculate_advanced_features(full_features_df)

    # Merge lại với Metadata gốc (Z, SpecType, target...)
    # Lưu ý: Index của full_features_df đang là object_id
    final_df = df_log.merge(full_features_df, on='object_id', how='left')

    return final_df


# -------------------------------------------------------------------------------------
# 5. CHẠY PIPELINE
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Xử lý tập Train
    if os.path.exists(TRAIN_LOG_PATH):
        train_df = process_dataset(TRAIN_LOG_PATH, is_train=True)
        print(f"✅ Đã xử lý xong Train Set. Shape: {train_df.shape}")

        # Lưu ra CSV để dùng cho các bước modeling sau
        train_df.to_csv('processed/processed_train_features.csv', index=False)
        print("💾 Đã lưu file: processed/processed_train_features.csv")
    else:
        print("❌ Không tìm thấy train_log.csv")

    # Xử lý tập Test
    if os.path.exists(TEST_LOG_PATH):
        test_df = process_dataset(TEST_LOG_PATH, is_train=False)
        print(f"✅ Đã xử lý xong Test Set. Shape: {test_df.shape}")

        # Lưu ra CSV
        test_df.to_csv('processed/processed_test_features.csv', index=False)
        print("💾 Đã lưu file: processed/processed_test_features.csv")
    else:
        print("❌ Không tìm thấy test_log.csv")