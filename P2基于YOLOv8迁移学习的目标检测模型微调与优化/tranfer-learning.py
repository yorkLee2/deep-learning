import argparse
import os
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv8 for PR2 dataset")
    parser.add_argument("--data_root", type=str, default="C:/Users/hyc49/Desktop/pr2/pr2",
                        help="Path to the PR2 dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output_file", type=str, default="C:/Users/hyc49/Desktop/pr2/pr2/predictions.txt",
                        help="Output file for test predictions")
    return parser.parse_args()

def train_yolo(args):
    model = YOLO("yolov8n.pt")  # 或者 "yolov8l.pt"
    model.train(
        data=os.path.join(args.data_root, "data.yaml"),
        imgsz=672,  # 提高输入尺寸
        batch=24,  # 避免显存溢出
        epochs=50,  
        device="cuda" if torch.cuda.is_available() else "cpu",

        optimizer="AdamW",  # 训练更稳定

        lr0=0.002,  # 适当降低学习率
        lrf=0.00005,  # 让学习率衰减更慢

        weight_decay=0.00005,  # 让正则化更慢
        momentum=0.9,  # 适中动量
        dropout=0.1,  # 适当 dropout 防止过拟合
        patience=35,  # 训练 20 轮无提升才停止
        cos_lr=True,  # 余弦学习率调整
        warmup_epochs=5.0,  # 增加 Warmup 时间
        warmup_momentum=0.9,
        warmup_bias_lr=0.2,
        flipud=0.5,  # 垂直翻转
        fliplr=0.5,  # 水平翻转
        hsv_s=0.6,  # HSV 饱和度变化
        scale=0.5,  # 目标缩放增强
        augment=True,  # 开启数据增强
        mosaic=1.0,  # 让 Mosaic 数据增强更多
        mixup=0.1,  # 让 MixUp 更强
        copy_paste=0.2,  # 适用于目标检测
        translate=0.3,  # 允许目标平移
        shear=0.2,  # 允许更大角度剪切
        perspective=0.003,  # 允许轻微透视变换
        iou=0.6,  # 适当降低 IoU 阈值，减少 False Negative
        conf=0.15,  # 降低最小置信度，减少漏检
        auto_augment="randaugment",  # 自动数据增强
        workers=8,  # 让数据加载更快
        half=True,  # 让计算更快
        plots=False  # 关闭绘图，加速训练
    )
    print("Training completed. Model saved in runs/detect/train/weights/")


def main():
    args = parse_args()
    train_yolo(args)

if __name__ == '__main__':
    main()
