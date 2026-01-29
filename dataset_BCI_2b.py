import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os


def prepare_bci_data(gdf_path, batch_size=32, train=True):
    # --- 1. 读取数据 (解决高低通反转警告) ---
    try:
        # 使用 preload=True 确保数据载入内存以便后续修改
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"无法读取文件 {gdf_path}: {e}")
        return None

    # --- 核心修复：绕过直接修改 info 的限制 ---
    # 先将 lowpass/highpass 这种元数据逻辑重置
    # 我们通过一次“空滤波”来强迫 MNE 重新校准 info 结构
    with raw.info._unlock():
        raw.info['lowpass'] = raw.info['sfreq'] / 2.0
        raw.info['highpass'] = 0.0

    # 动态匹配电极通道名
    picks = [ch for ch in raw.ch_names if any(t in ch for t in ['C3', 'Cz', 'C4'])]
    if len(picks) < 3:
        return None
    raw.pick(picks)

    # --- 2. 信号过滤：严格修正 l_freq < h_freq ---
    # 脑电运动想象核心频段为 8-30Hz
    # 使用 fir_design='firwin' 并显式设定频率，覆盖 GDF 头的错误信息

    raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)

    # --- 3. 提取事件并切分 ---
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    tmin, tmax = 0.5, 3.5

    # 兼容标签差异
    target_ids = {k: v for k, v in event_id.items() if any(label in k for label in ['769', '770', 'Left', 'Right'])}
    if not target_ids:
        return None

    try:
        epochs = mne.Epochs(raw, events, event_id=target_ids,
                            tmin=tmin, tmax=tmax, baseline=None,
                            preload=True, verbose=False)
        X = epochs.get_data(copy=True)[:, :, :750] * 1e6
        y = epochs.events[:, -1]
    except Exception:
        return None

    # 标签映射
    unique_y = np.unique(y)
    y_map = {val: i for i, val in enumerate(unique_y)}
    y = np.array([y_map[val] for val in y])

    # --- 4. 标准化与加载器构建 ---
    b, c, t = X.shape
    X_scaled = StandardScaler().fit_transform(X.reshape(b, -1)).reshape(b, c, t)

    X_tensor = torch.from_numpy(X_scaled).float()
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    print(f">>> 加载成功: {os.path.basename(gdf_path)}")
    return loader


if __name__ == "__main__":
    # 修复测试路径，使用 r 前缀
    test_path = r'E:\Python\PythonProject\dataset\BCI_2b\BCICIV_2b_gdf\B0101T.gdf'
    loader = prepare_bci_data(test_path)
    if loader:
        for x, y in loader:
            print(f"数据测试成功，Batch X 形状: {x.shape}")
            print(x[1:9], y[1:9])
            break