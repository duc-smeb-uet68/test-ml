import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Time2Vec(nn.Module):
    """
    Learnable Time Encoding: Giúp Transformer hiểu khoảng cách thời gian thực.
    """

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, output_dim - 1))
        self.b = nn.Parameter(torch.randn(1, output_dim - 1))

    def forward(self, t):
        # t: (Batch, Seq, 1)
        v0 = self.w0 * t + self.b0
        v1 = torch.sin(self.w * t + self.b)
        return torch.cat([v0, v1], dim=-1)



class AttentionPooling(nn.Module):
    """
    Thay vì lấy trung bình cộng (Mean Pooling), ta dùng một lớp Attention
    để model tự học xem bước thời gian nào quan trọng nhất (ví dụ lúc TDE bùng nổ).
    """

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask):
        # x: (B, S, d_model)
        # mask: (B, S) - True là data, False là padding

        # Tính attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (B, S)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, S, 1)

        # Tổng có trọng số
        weighted_sum = (x * attn_weights).sum(dim=1)  # (B, d_model)

        return weighted_sum


# ==========================================
# 3. MAIN TRANSFORMER ARCHITECTURE
# ==========================================
class MallornTransformer(nn.Module):
    def __init__(self,
                 d_model=256,  # Tăng capacity
                 n_head=8,  # Tăng độ phức tạp
                 n_layers=6,  # Mạng sâu hơn (Deep Network)
                 dim_feedforward=1024,
                 meta_dim=3,
                 num_bands=7,
                 dropout=0.2):  # Tăng dropout để chống overfit
        super().__init__()

        # --- Embeddings ---
        self.band_emb = nn.Embedding(num_bands, d_model // 4)
        self.num_proj = nn.Linear(2, d_model // 4)  # Flux + Flux_err
        self.time_enc = Time2Vec(d_model // 2)  # Time chiếm 50% thông tin vector

        self.dropout_emb = nn.Dropout(dropout)

        # --- Encoder Backbone ---
        # Pre-Norm (norm_first=True) giúp train ổn định hơn với mạng sâu
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'  # GELU thường tốt hơn ReLU cho Transformer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Pooling ---
        self.pool = AttentionPooling(d_model)

        # --- Metadata Fusion ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.BatchNorm1d(32),  # Thêm BatchNorm chuẩn hóa
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )

        # --- Final Head ---
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Logits
        )

    def forward(self, x_numeric, x_band, mask, meta):
        """
        x_numeric: (B, S, 3) -> [Flux, Flux_err, Time]
        x_band: (B, S)
        mask: (B, S) - True là data
        meta: (B, Meta_Dim)
        """
        # 1. Feature Extraction
        flux_feats = x_numeric[:, :, :2]
        time_feats = x_numeric[:, :, 2:3]

        e_band = self.band_emb(x_band)  # (B, S, d/4)
        e_num = self.num_proj(flux_feats)  # (B, S, d/4)
        e_time = self.time_enc(time_feats)  # (B, S, d/2)

        # Combine embeddings
        x = torch.cat([e_num, e_band, e_time], dim=-1)  # (B, S, d_model)
        x = self.dropout_emb(x)

        # 2. Transformer Pass
        src_key_padding_mask = ~mask  # True = ignore
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 3. Smart Pooling (Attention)
        # Thay vì lấy trung bình, mạng tự chọn điểm quan trọng
        feat_vec = self.pool(x, mask)  # (B, d_model)

        # 4. Metadata Fusion
        meta_vec = self.meta_mlp(meta)

        combined = torch.cat([feat_vec, meta_vec], dim=1)

        # 5. Prediction
        logits = self.head(combined)
        return logits