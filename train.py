import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import copy
import os
import pandas as pd  # Để lưu log

# Import model
from model import MallornTransformer

# ==========================================
# CẤU HÌNH FINAL
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-4
N_FOLDS = 5
SEED = 42


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)


class CachedDataset(Dataset):
    def __init__(self, mode='train', data_dir='processed_data', augment=False):
        self.numeric = np.load(f'{data_dir}/{mode}_numeric.npy')
        self.bands = np.load(f'{data_dir}/{mode}_bands.npy')
        self.mask = np.load(f'{data_dir}/{mode}_mask.npy')
        self.meta = np.load(f'{data_dir}/{mode}_meta.npy')
        self.target = np.load(f'{data_dir}/{mode}_target.npy')
        self.augment = augment

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Load data
        x_num = self.numeric[idx].copy()
        x_band = self.bands[idx]
        mask = self.mask[idx]
        meta = self.meta[idx]
        target = self.target[idx]


        if self.augment:

            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, size=x_num.shape[0])
                x_num[:, 0] += (noise * mask)

        return {
            'x_numeric': torch.tensor(x_num, dtype=torch.float32),
            'x_band': torch.tensor(x_band, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'meta': torch.tensor(meta, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ==========================================
# 3. ENGINE
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        x_num = batch['x_numeric'].to(device)
        x_band = batch['x_band'].to(device)
        mask = batch['mask'].to(device)
        meta = batch['meta'].to(device)
        y = batch['target'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(x_num, x_band, mask, meta)
        loss = criterion(preds, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Chống nổ gradient
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch['x_numeric'].to(device)
            x_band = batch['x_band'].to(device)
            mask = batch['mask'].to(device)
            meta = batch['meta'].to(device)
            y = batch['target']

            logits = model(x_num, x_band, mask, meta)
            preds = torch.sigmoid(logits)

            # Safety Check
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds, nan=0.0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    # Tìm threshold tối ưu
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(all_targets, (all_preds > th).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return auc, best_f1, best_th


# ==========================================
# 4. MAIN RUN
# ==========================================
if __name__ == "__main__":
    print(f"🚀 START TRAINING - {DEVICE} - {EPOCHS} EPOCHS")

    # Load Data
    full_dataset = CachedDataset(mode='train', augment=False)  # Base dataset
    targets = full_dataset.target

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n=== FOLD {fold + 1}/{N_FOLDS} ===")

        # Tạo dataset riêng cho từng fold (Train có Augment, Val không)
        train_sub = torch.utils.data.Subset(CachedDataset(mode='train', augment=True), train_idx)
        val_sub = torch.utils.data.Subset(CachedDataset(mode='train', augment=False), val_idx)

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0)  # Set workers=0 cho Window nếu lỗi
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

        model = MallornTransformer().to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        criterion = FocalLoss(alpha=0.75, gamma=2)

        best_val_f1 = 0
        best_threshold = 0.5

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            auc, f1, th = validate(model, val_loader, DEVICE)

            # Step scheduler dựa trên F1 Score
            scheduler.step(f1)

            print(f"Ep {epoch + 1:02d} | Loss: {loss:.4f} | Val AUC: {auc:.4f} | F1: {f1:.4f} (th={th:.2f})")

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_threshold = th
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

        print(f"🏆 Fold {fold + 1} Best F1: {best_val_f1:.4f} at Threshold: {best_threshold:.2f}")
        fold_results.append({'fold': fold, 'f1': best_val_f1, 'threshold': best_threshold})

    print("\n=== TRAINING COMPLETED ===")
    df_res = pd.DataFrame(fold_results)
    print(df_res)
    avg_th = df_res['threshold'].mean()
    print(f"🔥 Average Optimal Threshold for Submission: {avg_th:.3f}")
    # Lưu lại threshold trung bình để dùng cho inference
    with open("best_threshold.txt", "w") as f:
        f.write(str(avg_th))