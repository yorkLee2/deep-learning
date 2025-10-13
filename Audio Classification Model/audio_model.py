import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from transformers import Wav2Vec2Model

# -------------------------
# 固定随机种子，确保可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_FILE = "audio_features.pth"  # 确保该文件存在

# -------------------------
# 数据增强：添加轻微噪声（噪声因子调整）
def add_noise(waveform, noise_factor=0.003):
    noise = torch.randn_like(waveform)
    return waveform + noise_factor * noise

# 数据集类
class AudioDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data_list = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        waveform = item["waveform"]
        label = item["label"] - 1  # 0-based 标签
        if self.augment and random.random() < 0.5:
            waveform = add_noise(waveform)
        return waveform, label

# -------------------------
# 自注意力池化模块（用于 Wav2Vec2 分支）
class SelfAttentionPool(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPool, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch, time, dim]
        attn_weights = torch.softmax(self.attn(x), dim=1)  # [batch, time, 1]
        pooled = torch.sum(x * attn_weights, dim=1)         # [batch, dim]
        return pooled

# -------------------------
# 改进的模型：融合 Wav2Vec2 分支与简化后的 MFCC 分支（直接时间均值池化）
class AudioClassifierFusionSimpler(nn.Module):
    def __init__(self, num_classes=25, freeze_feature_extractor=True):
        super(AudioClassifierFusionSimpler, self).__init__()
        # 加载预训练 Wav2Vec2 模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        hidden_dim = self.wav2vec2.config.hidden_size  # 例如768

        # 自注意力池化用于 Wav2Vec2 输出
        self.attn_pool = SelfAttentionPool(hidden_dim)

        # MFCC 特征提取（使用 torchaudio.transforms.MFCC）
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "center": False}
        )

        # SpecAugment：适当增强
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=12)

        # 简化 MFCC 分支：对 MFCC 在时间维度直接取均值
        # 输入: [batch, 40, time] -> 输出: [batch, 40]
        # 再用 FC 映射到 128 维
        self.mfcc_fc = nn.Linear(40, 128)

        # 融合层：连接 Wav2Vec2 分支和简化后的 MFCC 分支
        fusion_dim = hidden_dim + 128
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout 提高到 0.4
            nn.Linear(256, num_classes)
        )

    def forward(self, waveform):
        # Wav2Vec2 分支
        outputs = self.wav2vec2(waveform)
        wav2vec_feats = self.attn_pool(outputs.last_hidden_state)  # [batch, hidden_dim]

        # MFCC 分支
        mfcc = self.mfcc_transform(waveform)  # [batch, 40, time]
        if self.training:
            mfcc = self.freq_mask(mfcc)
            mfcc = self.time_mask(mfcc)
        # 时间均值池化，得到 [batch, 40]
        mfcc_avg = mfcc.mean(dim=-1)
        # 映射到 128 维
        mfcc_feats = torch.relu(self.mfcc_fc(mfcc_avg))  # [batch, 128]

        # 融合
        fused = torch.cat([wav2vec_feats, mfcc_feats], dim=1)  # [batch, fusion_dim]
        logits = self.fusion_fc(fused)
        return logits

# -------------------------
# 主函数
def main():
    print("开始训练...")
    data = torch.load(FEATURES_FILE)
    train_data = data["train"]
    val_data = data["val"]

    train_dataset = AudioDataset(train_data, augment=True)
    val_dataset = AudioDataset(val_data, augment=False)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 25
    model = AudioClassifierFusionSimpler(num_classes=num_classes, freeze_feature_extractor=True)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    num_epochs = 30
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * waveforms.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        val_preds, val_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * waveforms.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "audio_model_best.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    print("训练结束，模型已保存。")

if __name__ == "__main__":
    main()
