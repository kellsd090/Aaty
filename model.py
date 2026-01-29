import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from dataset_gal import WAYEEGDataset, calculate_pos_weights


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SplineWeightLayer(nn.Module):
    def __init__(self, in_features=250, in_hidden=128, out_hidden=128, kernel_size=15):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  # 序列长度 750
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size

        # 1. 特征投影与独立位置偏置
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        # 保持每个采样点独立的偏置参数
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)

        # 2. 权重控制系数
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # I 的权重
        self.beta = nn.Parameter(torch.ones(1))  # W1 的权重
        self.theta = nn.Parameter(torch.ones(1))  # W3 的权重
        self.gamma = nn.Parameter(torch.tensor([0.5]))  # 残差权重

        # --- W3 1D 时序卷积先验 ---
        self.W3_conv = nn.Conv1d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False
        )
        self.register_buffer('identity', torch.eye(in_features).unsqueeze(0))

        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()

        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)
        x_prime = torch.matmul(x_norm, self.W2) + self.b

        # W3 路径
        x_temporal = x_prime.transpose(1, 2)
        w3_feat = self.W3_conv(x_temporal).transpose(1, 2)

        # W1 路径 (基于 Requirement 2 的自注意力输入)
        if W1 is not None:
            # 使用预存的 identity
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * self.identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat

        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, grid_size=128, spline_order=3, hidden_dim=128, neuron_out_dim=128,
                 kernel_size=15, seq_len=250):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        # 纯粹的 B-样条权重，去掉了 base_scales
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)

        # 每一个输出神经元对应一个 1D 优化的 SplineWeightLayer
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size)
            for _ in range(out_neurons)
        ])

        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))
        self.last_W1 = None

        # 将原本的随机初始化改为线性初始化
        # 目标：让输出 y = Spline(x) 在初始时接近 y = x
        with torch.no_grad():
            # 生成从 -1.0 到 1.0 的等差序列，代表 y=x 的趋势
            lin_init = torch.linspace(-1.0, 1.0, self.num_coeffs).to(self.spline_weights.device)
            # 形状调整为 [in_n, out_n, num_coeffs]
            initial_weights = lin_init.view(1, 1, -1).repeat(self.in_neurons, self.out_neurons, 1)
            # 稍微加一点极小的噪声打破对称性
            self.spline_weights.data = initial_weights + torch.randn_like(initial_weights) * 0.01

    def b_spline_cubic(self, x):
        """
        优化版三次 B-样条：通过数学公式合并分支，消除 mask 赋值
        """
        h = 2.0 / (self.num_coeffs - 1)
        # [B, in_n, S, D, 1] - [num_coeffs] -> [B, in_n, S, D, num_coeffs]
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h

        # 使用 ReLU 截断代替 mask，确保 dist < 2.0 之外的部分为 0
        # 通过组合多项式模拟原有的分段逻辑
        # 核心区 (dist < 1) 和 边缘区 (1 <= dist < 2)

        # 逻辑 1: 计算 (2 - dist)^3 / 6 (覆盖 0 <= dist < 2)
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0

        # 逻辑 2: 减去核心区多算的补偿值 (只针对 dist < 1)
        # 原逻辑核心区为: 2/3 - dist^2 + dist^3/2
        # 经过代数化简，补偿项为: 4/6 * (1 - dist)^3
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)

        # 最终结果 = 外部大包络 - 内部补偿补偿
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape # x_in: [B, in_n, 200, 128]

        # 1. 样条输入归一化
        max_val = torch.max(torch.abs(x_in)) + 1e-8
        x_norm = (x_in / max_val) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)

        # 2. Requirement 1: 纯 B-样条点乘逻辑
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum('binfc,ioc->bio nf', basis, self.spline_weights)
        # 2. 加上线性残差项 omega * x
        # omega 现在扮演的是“线性增益”或“残差系数”的角色
        omega_expanded = torch.abs(self.omiga).view(1, in_n, self.out_neurons, 1, 1)
        activated_signals = spline_mapping + omega_expanded * x_in.unsqueeze(2)

        next_outputs, next_projections = [], []
        # 针对 750 长度调整温度系数，防止梯度饱和
        t_val = torch.abs(self.temperature) * np.sqrt(seq_len / 256.0) + 1e-4

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(current_edges ** 2, dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((torch.sqrt(edge_energies + 1e-8) - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)

            # 3. Requirement 2: 保持自注意力逻辑
            W1_j = None
            if proj_x_prev is not None:
                multiplier = (torch.sqrt(edge_energies + 1e-8) / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask

                # 拼接所有输入神经簇的特征进行自注意力
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, 2*i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, 2*i+1, :, :] for i in range(mid)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            # 融合并输入神经元核心
            combined_input = torch.sum(current_edges * mask, dim=1)
            out_j, proj_j = self.weight_layers[j](combined_input, W1=W1_j)
            next_outputs.append(out_j)
            next_projections.append(proj_j)

        return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BCIPrepConv(nn.Module):
    def __init__(self, in_channels=160, out_channels=128, seq_len=200, stride=2):
        super(BCIPrepConv, self).__init__()

        # 1. 空间滤波器 (Spatial Filter)
        # 针对 160 通道（5个频段组），每组独立进行空间融合
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=5, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.SiLU()
        )

        # 2. 多尺度时间特征提取 (Multi-scale Temporal)
        # 步长建议设为 1，保持时间分辨率，或者设为 2 以减轻后续 KAN 压力
        self.stride = stride
        mid_channels = out_channels  # 64

        # 分支 1: 捕捉高频瞬态 (Gamma/Beta)
        self.branch_short = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 4, kernel_size=15, stride=self.stride, padding=7),
            nn.BatchNorm1d(mid_channels // 4), nn.SiLU()
        )
        # 分支 2: 捕捉中频节律 (Mu)
        self.branch_mid = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 2, kernel_size=33, stride=self.stride, padding=16),
            nn.BatchNorm1d(mid_channels // 2), nn.SiLU()
        )
        # 分支 3: 捕捉低频趋势 (Delta/MRCP)
        self.branch_long = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 4, kernel_size=65, stride=self.stride, padding=32),
            nn.BatchNorm1d(mid_channels // 4), nn.SiLU()
        )

        # 3. 最终投影与残差
        self.post_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

        # 残差快捷连接
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        # x 形状: [B, 160, 200]

        # 第一步：空间降噪与初步通道融合
        x_spatial = self.spatial_conv(x)

        # 第二步：多分支时间卷积并行扫描
        out = torch.cat([
            self.branch_short(x_spatial),
            self.branch_mid(x_spatial),
            self.branch_long(x_spatial)
        ], dim=1)

        # 第三步：残差融合
        res = self.shortcut(x)
        out = F.silu(self.post_conv(out) + res)

        # 转换至 KAN 格式: [B, Input_Neurons=1, New_Seq_Len, Feature_Dim=64]
        # 如果 stride=2, New_Seq_Len = 100
        out = out.transpose(1, 2).unsqueeze(1)
        return out


class BCIKANModel(nn.Module):
    def __init__(self, grid_size=64, spline_order=3, seq_len = 125):
        super(BCIKANModel, self).__init__()

        # 1. BCI 专用预处理层：将 [B, 3, 750] 映射为 [B, 1, 187, 64]
        # 使用 stride=4 进行时间轴下采样，平衡 W1 的计算压力
        self.prep_conv = BCIPrepConv(in_channels=160, out_channels=128, stride=2)

        # Layer 1: 初级特征解耦 [1 -> 4 神经簇]
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=128, kernel_size=55, seq_len=seq_len)
        # Layer 2: 空间-时域初步整合 [4 -> 8 神经簇]
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=128, kernel_size=33, seq_len=seq_len)
        # Layer 3: 核心互注意力层 [8 -> 16 神经簇]
        self.layer3 = FastKANLayer(8, 8, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=64, kernel_size=17, seq_len=seq_len)
        # Layer 4: 特征压缩与抽象 [16 -> 8 神经簇]
        #self.layer4 = FastKANLayer(16, 16, grid_size, spline_order,
        #                           hidden_dim=64, neuron_out_dim=64, kernel_size=7, seq_len=seq_len)
        # Layer 5: 决策语义准备 [8 -> 4 神经簇]
        self.layer4 = FastKANLayer(8, 12, grid_size, spline_order,
                                   hidden_dim=64, neuron_out_dim=32, kernel_size=9, seq_len=seq_len)
        # Layer 6: 分类映射 [4 -> 2 类别]
        self.layer5 = FastKANLayer(12, 8, grid_size, spline_order,
                                   hidden_dim=32, neuron_out_dim=16, kernel_size=5, seq_len=seq_len)
        self.layer6 = FastKANLayer(8, 6, grid_size, spline_order,
                                   hidden_dim=16, neuron_out_dim=6, kernel_size=3, seq_len=seq_len)

    def forward(self, x):
        # x: [B, 3, 750] -> 预处理后: [B, 1, 200, 160]
        x = self.prep_conv(x)

        # 逐层演化
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)
        x, proj5 = self.layer5(x, proj_x_prev=proj4)
        x, proj6 = self.layer6(x, proj_x_prev=proj5)
        # x, proj7 = self.layer7(x, proj_x_prev=proj6)

        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        """更新审计逻辑：适应新的 omiga 参数，移除 base_scales"""
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    # 强度计算：样条平均权重 + 静态增益 omiga
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.omiga)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {
                        "active": active_count,
                        "total": m.in_neurons * m.out_neurons,
                        "tau_mean": torch.abs(m.tau).mean().item(),
                        "ratio": active_count / (m.in_neurons * m.out_neurons)
                    }
        return info


def visualize_model_internals(model, layer_idx=1, neuron_idx=0, edge_idx=(0, 0)):
    model.eval()
    layer = getattr(model, f'layer{layer_idx}')
    # 确保兼容性：如果 visualize_model_internals 里的 layer.b_spl_cubic 指向正确的 b_spline_cubic
    if not hasattr(layer, 'b_spl_cubic'):
        layer.b_spl_cubic = layer.b_spline_cubic

    weight_layer = layer.weight_layers[neuron_idx]

    # 布局：3行2列
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- (1) W2 矩阵热力图 ---
    sns.heatmap(weight_layer.W2.detach().cpu().numpy(), ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W2 Matrix (Feature Projection)")

    # --- (2) W1 矩阵热力图 (动态生成) ---
    if layer.last_W1 is not None:
        sns.heatmap(layer.last_W1[0].detach().cpu().numpy(), ax=axes[0, 1], cmap='magma')
        axes[0, 1].set_title(f"L{layer_idx}-N{neuron_idx}: W1 Attention (Global Space)")
    else:
        axes[0, 1].text(0.5, 0.5, "W1 is None", ha='center')

    # --- (3) [修正] W3 矩阵可视化 (适配 1D 卷积) ---
    with torch.no_grad():
        # 提取 1D 卷积核权重: [out_hidden, 1, kernel_size]
        conv_weight = weight_layer.W3_conv.weight.detach().cpu()
        d_out, _, k = conv_weight.shape
        # 直接展示 1D 核的热力图，纵轴是通道，横轴是时间窗口
        sns.heatmap(conv_weight.squeeze(1).numpy(), ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title(f"L{layer_idx}-N{neuron_idx}: W3 1D Convolutional Kernels")
        axes[1, 0].set_xlabel("Time Window (k)")
        axes[1, 0].set_ylabel("Hidden Channels")

    # --- (4) θ, α, β, γ 参数柱状图 ---
    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item()
    }
    axes[1, 1].bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green'])
    axes[1, 1].set_title(f"L{layer_idx}-N{neuron_idx}: Component Weights")

    # --- (5) [修正] B-样条激活映射函数可视化 ---
    with torch.no_grad():
        # 定义测试量程：从 -1 到 1 (对应我们归一化后的 x_norm)
        x_range = torch.linspace(-1, 1, 200).to(layer.spline_weights.device)

        # 获取当前边的 omiga (线性残差系数)
        current_omiga = torch.abs(layer.omiga[edge_idx[0], edge_idx[1]])

        # 1. 计算纯样条映射输出: f(x)
        # basis 形状: [1, 200, num_coeffs]
        basis = layer.b_spline_cubic(x_range.unsqueeze(0))
        # coeffs 形状: [num_coeffs]
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        # y_spline 形状: [200]
        y_spline = torch.matmul(basis, coeffs).squeeze()

        # 2. 计算线性捷径输出: omega * x
        y_linear = current_omiga * x_range

        # 3. 最终组合输出: y = f(x) + omega * x (替代了原来的 SiLU)
        y_final = y_spline + y_linear

        # 绘图
        axes[2, 0].plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline Mapping $f(x)$', alpha=0.7)
        axes[2, 0].plot(x_range.cpu(), y_linear.cpu(), ':', label=r'Linear Shortcut $\omega \cdot x$', alpha=0.7)
        axes[2, 0].plot(x_range.cpu(), y_final.cpu(), 'r', lw=2.5, label='Total Activation (Learned)')

        # 辅助线：绘制 y=x 虚线作为参考，看看模型偏离线性多远
        axes[2, 0].plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.2, lw=1, label='Identity (y=x)')

        axes[2, 0].set_title(f"Edge ({edge_idx[0]}->{edge_idx[1]}): Value Mapping Function")
        axes[2, 0].set_xlabel("Input Normalized Voltage")
        axes[2, 0].set_ylabel("Output Mapped Voltage")
        axes[2, 0].legend(loc='upper left', fontsize='small')
        axes[2, 0].grid(True, alpha=0.3)

    # --- (6) 层间导通热力图 (基于 omiga) ---
        # --- (6) 层间导通热力图 (基于 RMS 强度判定) ---
    with torch.no_grad():
        # 粗略估计每条边的“平均表现强度”
        # 样条权重的均值 + omiga 线性强度
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
        # 注意：这里的判断逻辑应与 forward 里的 mask 逻辑对应
        active_mask = (w_strength > torch.abs(layer.tau)).float()
        sns.heatmap(active_mask.cpu().numpy(), ax=axes[2, 1], cmap='Greens', cbar=False)

    plt.show()


# --- 测试运行 ---
# --- 修改后的主函数：适配 BCI 2b 原生信号与 6 层架构 ---
if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 路径配置
    checkpoint_path = r"C:\Users\dyh\Downloads\best_gal_model_04.pth"
    data_root = "E:\\Python\\PythonProject\\dataset\\grasp-and-lift-eeg-detection\\train\\train"

    # 2. 实例化模型
    model = BCIKANModel().to(device)

    # 3. 加载真实数据进行对比可视化 (新增加部分)
    print("\n>>> 正在准备原始与预处理数据对比...")
    # 手动读取一个原始 CSV 文件作为对比基准
    raw_data_path = os.path.join(data_root, 'subj1_series1_data.csv')
    raw_df_example = pd.read_csv(raw_data_path).iloc[:, 1:].values.astype(np.float32)

    try:
        val_ds_example = WAYEEGDataset(
            data_root,
            subject_ids=[9],
            series_ids=[9],
            window_size=250,
            stride=250,
            is_train=True
        )

        # 调用 dataset_gal.py 中内置的可视化函数
        # 这将展示 CAR、滤波、去离群点及 Z-Score 的综合效果
        from dataset_gal import visualize_preprocessing_effect

        visualize_preprocessing_effect(
            raw_df_example,
            val_ds_example.raw_data,
            num_channels=3  # 随机选3个通道展示
        )

        real_input, real_label = val_ds_example[0]
        real_input = real_input.unsqueeze(0).to(device, dtype=torch.float)
    except Exception as e:
        print(f"数据加载对比失败: {e}")
        real_input = torch.randn(1, 160, 250).to(device)

    # 4. 加载模型权重 (保持不变)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        print(f"成功加载权重: {checkpoint_path}")

    # 5. KAN 内部审计可视化 (保持不变)
    target_layer_idx = 6
    target_neuron_idx = 4
    edge = (0, 0)
    target_layer = getattr(model, f'layer{target_layer_idx}')
    target_layer.visualize_idx = target_neuron_idx
    if not hasattr(target_layer, 'b_spl_cubic'):
        target_layer.b_spl_cubic = target_layer.b_spline_cubic

    model.eval()
    with torch.no_grad():
        _ = model(real_input)

    print(f"\n--- 正在生成 KAN 内部激活可视化 ---")
    visualize_model_internals(model, layer_idx=target_layer_idx, neuron_idx=target_neuron_idx, edge_idx=edge)