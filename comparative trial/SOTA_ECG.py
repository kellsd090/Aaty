import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import Counter

from database_mit import prepare_ecg_data
from model import seed_everything


# --- 增强型 SOTA 残差单元 ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()

        # 对齐残差路径 (如果 stride > 1 或通道改变)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.activation(out)


class ECGResNetSOTA(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGResNetSOTA, self).__init__()
        # 1. 输入层: 提取基础特征 (1 -> 64 通道)
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )

        # 2. 两个隐藏层级 (对标 1-8-8-5 的两个 8)
        # 第一层级: 保持分辨率
        self.layer1 = ResidualBlock(64, 64, kernel_size=15)
        # 第二层级: 降采样并增加通道至 128
        self.layer2 = ResidualBlock(64, 128, kernel_size=15, stride=2)

        # 3. 输出层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 187] -> [Batch, 1, 187]
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


# --- 训练逻辑 ---
def train_sota_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)

    model = ECGResNetSOTA().to(device)

    # 平方根平滑权重统计
    all_labels = []
    for _, y in train_loader: all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    raw_weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)])
    model.class_weights = torch.sqrt(raw_weights / raw_weights.mean()).to(device)

    # ================= [初始状态显示] =================
    model.eval()
    init_val_loss = 0
    init_correct = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            init_val_loss += F.cross_entropy(v_out, vy, weight=model.class_weights).item()
            init_correct += (v_out.argmax(1) == vy).sum().item()

    avg_init_loss = init_val_loss / len(val_loader)
    print(
        f"\n[SOTA Initial State] Loss: {avg_init_loss:.6f}, Val Acc: {100 * init_correct / len(val_loader.dataset):.2f}%")
    print("-" * 75)

    # 优化配置 (使用 AdamW 配合余弦退火学习率)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    best_acc = 0

    for epoch in range(30):
        model.train()
        train_loss = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"SOTA ResNet Epoch {epoch + 1}/30")
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = F.cross_entropy(out, by, weight=model.class_weights)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / len(train_loader.dataset):.2f}%")

        # 验证阶段
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                val_loss += F.cross_entropy(v_out, vy, weight=model.class_weights).item()
                val_correct += (v_out.argmax(1) == vy).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / len(val_loader.dataset)

        print(f">>> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_resnet_sota.pth")
            print(">>> 最佳 SOTA ResNet 模型已保存")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n训练完成。最高准确率: {best_acc:.2f}%, 总参数量: {total_params}")


if __name__ == "__main__":
    seed_everything(42)
    train_sota_resnet()