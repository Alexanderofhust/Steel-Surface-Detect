import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 原来的可视化函数（保持不变）
category_colors = {
    1: (255, 0, 0),    # 类别 (1,1,1) 用红色表示
    2: (0, 255, 0),    # 类别 (2,2,2) 用绿色表示
    3: (0, 0, 255),    # 类别 (3,3,3) 用蓝色表示
}

def calculate_iou(ground_truth, prediction):
    # 将掩码二值化，假设只有四种类（0, 1, 2, 3）
    iou_list = []
    for class_id in range(1, 5):  # 类别从 1 到 3（0 是背景）
        # 创建二值掩码
        if class_id != 4:
            gt_mask = (ground_truth == class_id).astype(np.uint8)
            pred_mask = (prediction == class_id).astype(np.uint8)
        elif class_id == 4:
            gt_mask = (ground_truth != 0).astype(np.uint8)
            pred_mask = (prediction != 0).astype(np.uint8)

        # 计算交集和并集
        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)

        # 计算 IoU
        if union == 0:
            iou = float('nan')  # 防止除以零
        else:
            iou = intersection / union
        
        iou_list.append(iou)

    
    return iou_list

def visualize_mask_on_image(image, mask):
    # 转换为RGB格式，便于可视化
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 遍历每个类别，并找到边界绘制在图像上
    for category in category_colors.keys():
        # 创建一个只包含当前类别区域的mask
        category_mask = np.all(mask == (category, category, category), axis=-1).astype(np.uint8) * 255

        # 查找mask的轮廓
        contours, _ = cv2.findContours(category_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原图上绘制轮廓
        cv2.drawContours(image_rgb, contours, -1, category_colors[category], 1)  # 2 是线条的粗细

    # 返回带有边界的图像
    return image_rgb

# 新增函数，用于从文件夹中读取图片和mask
def process_images_and_masks(image_dir, mask_dir):
    # 获取图片和mask文件列表
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    plt.figure(figsize=(8, 8))

    # 遍历文件，确保名称匹配
    for img_file in image_files:
        # 替换文件扩展名，假设image是jpg格式，mask是png格式
        mask_file = img_file.replace('.jpg', '.png')
        
        # 检查mask文件是否存在
        if mask_file in mask_files:
            # 构建完整路径
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # 读取图像和对应的mask
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            visualized_image = visualize_mask_on_image(image, mask)

            # 使用Matplotlib显示结果
            plt.imshow(visualized_image)
            plt.title(img_file)
            plt.axis('off')
            #plt.show(block=False)
            plt.pause(0.001)
            plt.clf()

# 新增函数，用于从文件夹中读取图片，realmask和predmask

def process_images_and_masks(image_dir, real_mask_dir, pred_mask_dir, visualize=True,debug_visualize=True):
    # 获取图片和mask文件列表
    image_files = sorted(os.listdir(image_dir))
    real_mask_files = sorted(os.listdir(real_mask_dir))
    pred_mask_files = sorted(os.listdir(pred_mask_dir))

    iou_all = []

    if visualize|debug_visualize:
        # 创建子图，2列1行布局
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plt.ioff()  # 关闭交互模式
        plt.tight_layout()
        axes[0].axis('off')  # 不显示坐标轴
        axes[1].axis('off')  # 不显示坐标轴

    # 遍历文件，确保名称匹配
    for img_file in tqdm(image_files):
        # 假设真实mask和预测mask文件的命名与image一致
        real_mask_file = img_file.replace('.jpg', '.png')  # 假设真实mask是png格式
        pred_mask_file = img_file.replace('.jpg', '.png')  # 假设预测mask也是png格式

        # 检查真实mask和预测mask文件是否存在
        if real_mask_file in real_mask_files and pred_mask_file in pred_mask_files:
            # 构建完整路径
            img_path = os.path.join(image_dir, img_file)
            real_mask_path = os.path.join(real_mask_dir, real_mask_file)
            pred_mask_path = os.path.join(pred_mask_dir, pred_mask_file)

            # 读取图像和对应的mask
            image = cv2.imread(img_path)
            real_mask = cv2.imread(real_mask_path, cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

            iou_list = calculate_iou(real_mask, pred_mask)

            iou_all.append(iou_list)
            temp_visualize=False
            # if iou_list[0] == 0 & debug_visualize:
            #     temp_visualize=True
            # if exist small iou, visualize it
            if any(iou < 0.1 for iou in iou_list) & debug_visualize:
                temp_visualize = True
            

            if visualize|(temp_visualize & debug_visualize):
                print(f"IoU for {img_file}: {iou_list}")

                # 假设 visualize_mask_on_image 是一个用于将mask叠加在图像上的函数
                visualized_real_mask = visualize_mask_on_image(image, real_mask)
                visualized_pred_mask = visualize_mask_on_image(image, pred_mask)

                
                # 显示带有真实mask的图像
                axes[0].imshow(visualized_real_mask)

                # 显示带有预测mask的图像
                axes[1].imshow(visualized_pred_mask)

                # 绘制图像
                plt.pause(0.01)  # 短暂暂停以显示图像
                # plt.clf()  # 清除当前图像，准备显示下一个
    
    return iou_all


    

# 主程序入口
if __name__ == "__main__":
    # 定义文件夹路径
    base_dir = ''  # 替换为实际数据集的根路径
    training_image_dir = os.path.join(base_dir, 'images/training')
    training_mask_dir = os.path.join(base_dir, 'annotations/training')
    test_image_dir = os.path.join(base_dir, 'images/test')
    test_mask_dir = os.path.join(base_dir, 'annotations/test')
    result_training_dir = os.path.join(base_dir, 'results/training')
    result_test_dir = os.path.join(base_dir, 'results/test')

    print("Visualizing Result Set...")
    iou_all = process_images_and_masks(test_image_dir, test_mask_dir, result_test_dir,visualize=False,debug_visualize=False)

    print("Average IoU for Test Set:")
    iou_all = np.array(iou_all)
    print(np.nanmean(iou_all, axis=0))
    # draw the distribution of iou
    plt.figure(figsize=(6, 4))
    plt.hist(iou_all[:, 0], bins=100, alpha=0.5, label='IoU Class 1', color='r')
    plt.hist(iou_all[:, 1], bins=100, alpha=0.5, label='IoU Class 2', color='g')
    plt.hist(iou_all[:, 2], bins=100, alpha=0.5, label='IoU Class 3', color='b')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('IoU Distribution for Test Set')
    plt.show()

