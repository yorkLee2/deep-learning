import os
import torch
import torchaudio
from audio_model import AudioClassifierFusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "audio_model_best.pth"  # 使用训练好的最佳模型
TEST_DIR = "test"  # 测试文件夹路径（仅包含 .wav 文件）
OUTPUT_FILE = "prediction.txt"
TARGET_SR = 16000
FIXED_LENGTH = TARGET_SR * 5

def load_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
    waveform = waveform[0]
    if waveform.size(0) > FIXED_LENGTH:
        waveform = waveform[:FIXED_LENGTH]
    else:
        pad_size = FIXED_LENGTH - waveform.size(0)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

def load_model():
    num_classes = 25
    model = AudioClassifierFusion(num_classes=num_classes, freeze_feature_extractor=True)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_audio(model, audio_path):
    waveform = load_audio(audio_path)
    waveform = waveform.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(waveform)
        _, pred = torch.max(outputs, 1)
    return pred.item() + 1  # 将 0-based 转换为 1～25

def main():
    model = load_model()
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".wav")])
    predictions = []
    print("Starting prediction on test files...")
    for file in test_files:
        path = os.path.join(TEST_DIR, file)
        pred = predict_audio(model, path)
        predictions.append(pred)
        print(f"File: {file} --> Predicted label: {pred}")
    with open(OUTPUT_FILE, "w") as f:
        for label in predictions:
            f.write(f"{label}\n")
    print(f"✅ Prediction complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

