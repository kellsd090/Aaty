import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import math

from database_mit import prepare_ecg_data
from model import seed_everything


# --- 标准 Transformer 架构 ---
class ECGTransformerModel(nn.Module):
    def __init__(self, seq_len=187, input_dim=1, d_model=128, nhead=8, num_layers=3, num_classes=5):
        super(ECGTransformerModel, self).__init__()

        # 1. Input Projector: 将原始信号映射到标准 d_model 维度
        self.embedding = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding: 使用可学习的位置向量
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 3. Transformer Encoder Blocks (标准 8头 3层)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 512 维中间层
            batch_first=True,
            activation='gelu',
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 特征聚合与规范化
        self.ln = nn.LayerNorm(d_model)

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 187] -> [Batch, 187, 1]
        x = x.unsqueeze(-1)

        # 嵌入与位置叠加
        x = self.embedding(x) + self.pos_embedding

        # 全局自注意力计算 [Batch, 187, 128]
        x = self.transformer_encoder(x)

        # 全局池化 (取序列均值作为整体特征)
        x = x.mean(dim=1)

        x = self.ln(x)
        return self.classifier(x)


def calculate_class_weights(dataloader):
    all_labels = []
    for _, y in dataloader: all_labels.extend(y.numpy())
    counts = Counter(all_labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        raw_w = total / (num_classes * counts[i])
        weights[i] = torch.sqrt(torch.tensor(raw_w))  # 平方根平滑
    return weights / weights.mean()


def train_transformer_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = prepare_ecg_data('E:/database/archive/mitbih_train.csv', batch_size=64)
    val_loader = prepare_ecg_data('E:/database/archive/mitbih_test.csv', batch_size=64)

    # 实例化标准 128 维模型
    model = ECGTransformerModel(d_model=128, nhead=8, num_layers=3).to(device)
    model.class_weights = calculate_class_weights(train_loader).to(device)

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
    init_acc = 100 * init_correct / len(val_loader.dataset)
    print(f"\n[Initial State] Loss: {avg_init_loss:.6f}, Val Acc: {init_acc:.2f}%")
    print("-" * 75)

    # 优化配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 余弦退火学习率
    best_acc = 0

    for epoch in range(1):
        model.train()
        train_loss, train_correct = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/50")

        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = F.cross_entropy(out, by, weight=model.class_weights)
            loss.backward()

            # Transformer 训练的关键：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == by).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * train_correct / len(train_loader.dataset):.2f}%")

        # 验证
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
            torch.save(model.state_dict(), "best_transformer_std.pth")
            print(">>> 最佳性能版 Transformer 已保存")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n训练完成。最高准确率: {best_acc:.2f}%, 总参数量: {total_params}")


if __name__ == "__main__":
    seed_everything(42)
    train_transformer_baseline()