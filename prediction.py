import os
import torch
from ultralytics import YOLO

# 预测参数
DATA_ROOT = "C:/Users/hyc49/Desktop/pr2/pr2"
OUTPUT_FILE = os.path.join(DATA_ROOT, "predictions.txt")

def get_latest_model():
    """ 获取最新训练的 YOLO 模型 """
    model_path = os.path.join(DATA_ROOT, "runs", "detect", "train", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到训练好的模型权重: {model_path}\n请先运行 train_yolo.py 进行训练！")
    
    print(f"✅ 使用模型: {model_path}")
    return model_path

def generate_predictions():
    """ 运行 YOLOv8 模型并生成预测结果 """
    test_images = os.path.join(DATA_ROOT, "test", "images")
    if not os.path.exists(test_images):
        raise FileNotFoundError(f"测试图片路径 {test_images} 不存在！")

    # 加载最新的 YOLO 模型
    model_path = get_latest_model()
    model = YOLO(model_path)

    # 进行预测
    results = model.predict(source=test_images, save_txt=True, save_conf=True, conf=0.1, show_conf=True)

    # 处理并保存预测结果
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            image_name = os.path.basename(result.path).split('.')[0]  # 图片编号（去掉扩展名）
            image_id = int(image_name)  # 确保是整数
            for box in result.boxes:
                cls = int(box.cls.item()) + 1  # 类别 ID（转换为 1-based）
                xywhn = box.xywhn.view(-1).tolist()  # YOLO 归一化坐标
                conf = float(box.conf.item())  # 置信度

                if len(xywhn) == 4:  # 确保格式正确
                    f.write(f"{image_id} {cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f} {conf:.6f}\n")

    print(f"✅ 预测结果已保存到 {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_predictions()




