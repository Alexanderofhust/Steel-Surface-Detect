import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from unet import Unet


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
def load_model():
    model = Unet(num_classes=4)  # 根据实际情况修改
    return model

# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = np.array(image) / 255.0
    image = transforms.ToTensor()(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
    
    return image.to(device)

# 预测
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)  # 进行推理
        # 获取最大概率的类别
        predicted_mask = torch.argmax(outputs, dim=1)  # [1, 200, 200]
        return predicted_mask.squeeze().cpu().numpy()  # 移除多余维度并转换为 NumPy 数组

# 保存预测结果
def save_prediction(prediction, output_path):
    # 由于预测结果的形状为 [200, 200]，可以直接保存为图像
    prediction.save(output_path)

def main(input_dir, output_dir):
    # 加载模型
    model = load_model()

    # 遍历输入目录中的图像
    for image_file in os.listdir(input_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_dir, image_file)
            print(f"Processing {image_path}...")

            image = Image.open(image_path)
            r_image = model.detect_image(image, count=False, name_classes=4)

            # 保存结果
            output_path = os.path.join(output_dir, f"{image_file}")
            save_prediction(r_image, output_path.replace('.jpg', '.png'))
            print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    input_directory = "images/test"  # 输入图像文件夹路径
    output_directory = "results/test"  # 输出掩码保存路径

    os.makedirs(output_directory, exist_ok=True)  # 创建输出文件夹
    main(input_directory, output_directory)
