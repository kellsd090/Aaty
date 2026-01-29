import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader

# 导入你项目中的组件
from dataset_gal import WAYEEGDataset
from model import BCIKANModel, seed_everything


def test_gal_model(model, test_loader, device, threshold=0.1):
    """
    针对 WAY-EEG-GAL 脑电多标签任务的评估函数
    """
    model.eval()
    all_preds = []
    all_targets = []
    classes = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']

    print(">>> 开始在测试集上进行推理...")
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, dtype=torch.float)
            # 模型输出
            outputs = model(bx)
            # 多标签分类使用 Sigmoid 转化为概率
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(by.cpu().numpy())

    # 合并所有 batch 数据
    y_true_continuous = np.vstack(all_targets)
    y_probs = np.vstack(all_preds)

    # --- 核心修复步骤 ---
    # 1. 将平滑后的连续标签还原为二值标签 (0 或 1)
    # 因为平滑后的波峰对应原始的 1，所以使用 0.5 作为判定界限
    y_true_binary = (y_true_continuous > 0.5).astype(int)

    # 2. 将预测概率转换为二值预测
    y_preds_binary = (y_probs > threshold).astype(int)

    # 1. 打印分类报告
    print("\n" + "=" * 20 + " 脑电 6 类别分类报告 " + "=" * 20)
    # 使用还原后的二值标签进行对比
    print(classification_report(y_true_binary, y_preds_binary, target_names=classes, zero_division=0))

    # 2. 绘制 6 个种类的热力图 (每个类别一个 2x2 混淆矩阵)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i in range(6):
        # 同样使用还原后的二值标签计算混淆矩阵
        cm = confusion_matrix(y_true_binary[:, i], y_preds_binary[:, i])

        # 归一化显示百分比
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[i],
                    xticklabels=['Absent', 'Present'],
                    yticklabels=['Absent', 'Present'])
        axes[i].set_title(f'Category: {classes[i]}')
        axes[i].set_ylabel('True Label (Binary)')
        axes[i].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置路径（确保这里的路径与你电脑实际路径一致）
    PATH_train = r"E:\Python\PythonProject\dataset\grasp-and-lift-eeg-detection\train\train"
    model_path = r"C:\Users\dyh\Downloads\best_gal_model_02.pth"

    # --- 提取训练集统计信息以进行归一化 ---
    print(">>> 正在提取训练集统计信息以进行归一化...")
    temp_train_ds = WAYEEGDataset(
        data_path=PATH_train,
        subject_ids=[1],
        series_ids=[1, 2, 3, 4, 5, 6],
        is_train=True
    )
    train_stats = temp_train_ds.stats

    # --- 初始化测试集 ---
    print(">>> 正在加载测试数据 (Series 7, 8)...")
    test_ds = WAYEEGDataset(
        data_path=PATH_train,
        subject_ids=[1],
        series_ids=[7, 8],
        is_train=False,
        stats=train_stats
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # --- 加载模型 ---
    model = BCIKANModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n>>> 模型加载成功 | 总参数量: {total_params:,}")
    if os.path.exists(model_path):
        # 针对 6 类别任务，确保加载时忽略掉不匹配的旧权重（如果有）
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")

        # 执行测试与绘图
        test_gal_model(model, test_loader, device)
    else:
        print(f"错误：找不到模型文件 {model_path}")