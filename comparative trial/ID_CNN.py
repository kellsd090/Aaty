import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import os

# 导入你已有的数据处理函数
from database_mit import prepare_ecg_data
from model import seed_everything


# --- 模型定义 (保持不变) ---
class CNNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(CNNUnit, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=3)
        self.ln = nn.LayerNorm(187)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        return self.activation(x)


class ECGCNNModel(nn.Module):
    def __init__(self):
        super(ECGCNNModel, self).__init__()
        self.layer1 = CNNUnit(1, 8)
        self.layer2 = CNNUnit(8, 8)
        self.layer3 = nn.Conv1d(8, 5, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.global_pool(x).squeeze(-1)
        return out


# --- 损失函数 ---
def cnn_balanced_loss(outputs, targets, model, gamma_focal=2.0):
    log_p = F.log_softmax(outputs, dim=-1)
    p = torch.exp(log_p)
    pt = p.gather(1, targets.view(-1, 1))
    log_pt = log_p.gather(1, targets.view(-1, 1))
    focal_term = (1 - pt) ** gamma_focal

    if hasattr(model, 'class_weights'):
        batch_weights = model.class_weights.to(outputs.device).gather(0, targets).view(-1, 1)
        ce_loss = (-batch_weights * focal_term * log_pt).mean()
    else:
        ce_loss = (-focal_term * log_pt).mean()
    return ce_loss


# --- 新增：平滑权重计算 ---
def calculate_class_weights(dataloader):
    all_labels = []
    for _, y in dataloader: all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)

    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        raw_w = total / (num_classes * counts[i])
        # 使用平方根平滑平衡梯度压力
        weights[i] = torch.sqrt(torch.tensor(raw_w))

    return weights / weights.mean()


# --- 训练逻辑 ---
def train_cnn_baseline(model, train_loader, val_loader, epochs=50, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 自动化权重统计
    print("统计类别分布并应用平滑权重...")
    model.class_weights = calculate_class_weights(train_loader).to(device)
    print(f"CNN 类别权重: {model.class_weights.tolist()}")

    # ================= [新增：初始状态评估] =================
    model.eval()
    init_val_loss, init_val_correct = 0, 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            init_val_loss += cnn_balanced_loss(v_out, vy, model).item()
            init_val_correct += (v_out.argmax(1) == vy).sum().item()

    avg_init_loss = init_val_loss / len(val_loader)
    init_acc = 100 * init_val_correct / len(val_loader.dataset)
    print(f"\n[Initial State] Val Loss: {avg_init_loss:.6f}, Val Acc: {init_acc:.2f}%")
    print("-" * 60)
    # ======================================================

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        pbar = tqdm(train_loader, desc=f"CNN Epoch {epoch + 1}/{epochs}")
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = cnn_balanced_loss(out, by, model)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / len(train_loader.dataset):.2f}%")

        # 验证
        model.eval()
        val_correct, val_loss = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += cnn_balanced_loss(v_out, vy, model).item()
                val_correct += (v_out.argmax(1) == vy).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}: Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_baseline.pth")
            print(">>> 最佳 CNN 模型已保存")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print("Early stopping...")
            break

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n训练完成。最高准确率: {best_acc:.2f}%")
    print(f"1D-CNN 总参数量: {total_params}")


if __name__ == "__main__":
    seed_everything(42)
    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)

    cnn_model = ECGCNNModel()
    train_cnn_baseline(cnn_model, train_loader, val_loader)