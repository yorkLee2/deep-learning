import os
import torch
import torchaudio

# 参数设定
DATASET_DIR = "C:/Users/hyc49/Desktop/p3"  
OUTPUT_FILE = "audio_features.pth"  # 保存训练和验证数据
TARGET_SR = 16000
FIXED_LENGTH = TARGET_SR * 5  # 5秒钟

def load_and_process_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)  # waveform shape: [channels, samples]
    # 如果采样率不等于目标采样率，则进行重采样（只保留单通道）
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
    # 取第一个通道
    waveform = waveform[0]
    # 截断或填充到固定长度
    if waveform.size(0) > FIXED_LENGTH:
        waveform = waveform[:FIXED_LENGTH]
    else:
        pad_size = FIXED_LENGTH - waveform.size(0)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

def process_dataset(dataset_type):
    audio_dir = os.path.join(DATASET_DIR, dataset_type)
    label_file = os.path.join(audio_dir, "labels.txt")
    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f]
    filenames = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    data_list = []
    print(f"Processing {dataset_type} dataset with {len(filenames)} samples...")
    for i, file in enumerate(filenames):
        audio_path = os.path.join(audio_dir, file)
        if os.path.exists(audio_path):
            waveform = load_and_process_audio(audio_path)  # [FIXED_LENGTH]
            data_list.append({
                "filename": file,
                "waveform": waveform,  # 原始 waveform，1D Tensor
                "label": labels[i]
            })
        else:
            print(f"Warning: {audio_path} not found, skipping.")
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(filenames)} samples.")
    return data_list

if __name__ == "__main__":
    train_data = process_dataset("train")
    val_data = process_dataset("val")
    torch.save({"train": train_data, "val": val_data}, OUTPUT_FILE)
    print(f"Feature extraction complete. Saved to {OUTPUT_FILE}.")
