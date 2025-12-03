import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from model import MallornTransformer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
DATA_DIR = 'processed_data'


class TestDataset(Dataset):
    def __init__(self):
        self.numeric = np.load(f'{DATA_DIR}/test_numeric.npy')
        self.bands = np.load(f'{DATA_DIR}/test_bands.npy')
        self.mask = np.load(f'{DATA_DIR}/test_mask.npy')
        self.meta = np.load(f'{DATA_DIR}/test_meta.npy')
        self.ids = np.load(f'{DATA_DIR}/test_ids.npy', allow_pickle=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'x_numeric': torch.tensor(self.numeric[idx], dtype=torch.float32),
            'x_band': torch.tensor(self.bands[idx], dtype=torch.long),
            'mask': torch.tensor(self.mask[idx], dtype=torch.bool),
            'meta': torch.tensor(self.meta[idx], dtype=torch.float32),
            'id': self.ids[idx]
        }


def run_inference():
    print("🚀 Loading Test Data...")
    test_ds = TestDataset()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Tìm tất cả model checkpoint
    model_files = [f for f in os.listdir('.') if f.startswith('best_model_fold') and f.endswith('.pth')]
    if not model_files:
        print("❌ LỖI: Không tìm thấy file model nào. Hãy chạy train.py trước!")
        return

    print(f"🔄 Ensemble {len(model_files)} models: {model_files}")

    # Load tất cả models vào list
    models = []
    for m_file in model_files:
        m = MallornTransformer().to(DEVICE)
        m.load_state_dict(torch.load(m_file, map_location=DEVICE))
        m.eval()
        models.append(m)

    all_probs = []
    all_ids = []

    print("🔮 Predicting...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x_num = batch['x_numeric'].to(DEVICE)
            x_band = batch['x_band'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            meta = batch['meta'].to(DEVICE)
            ids = batch['id']

            # Ensemble Prediction
            batch_preds = []
            for model in models:
                logits = model(x_num, x_band, mask, meta)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Trung bình cộng xác suất (Average Blending)
            avg_pred = np.mean(batch_preds, axis=0)

            all_probs.extend(avg_pred.flatten())
            all_ids.extend(ids)


    try:
        with open("best_threshold.txt", "r") as f:
            threshold = float(f.read().strip())
        print(f"🎯 Applying Optimal Threshold: {threshold}")
    except:
        threshold = 0.5
        print("⚠️ Không tìm thấy threshold file, dùng mặc định 0.5")

    submission = pd.DataFrame({
        'object_id': all_ids,
        'probability': all_probs
    })


    submission['target'] = (submission['probability'] > threshold).astype(int)

    try:
        sample = pd.read_csv('data/sample_submission.csv')
        final_sub = sample[['object_id']].merge(submission[['object_id', 'target']], on='object_id', how='left')
        final_sub['target'] = final_sub['target'].fillna(0).astype(int)

        final_sub.to_csv('final_submission.csv', index=False)
        print("✅ SUCCESS! File 'final_submission.csv' đã sẵn sàng nộp.")
        print(final_sub.head())
    except Exception as e:
        print(f"⚠️ Warning: Không đọc được sample_submission, xuất file raw. Lỗi: {e}")
        submission.to_csv('raw_submission.csv', index=False)


if __name__ == "__main__":
    run_inference()