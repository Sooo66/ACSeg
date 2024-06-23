import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np

# colors = [[  0   0   0],
#         [128   0   0],
#         [  0 128   0],
#         [128 128   0],
#         [  0   0 128],
#         [128   0 128],
#         [  0 128 128],
#         [128 128 128],
#         [ 64   0   0],
#         [192   0   0],
#         [ 64 128   0],
#         [192 128   0],
#         [ 64   0 128],
#         [192   0 128],
#         [ 64 128 128],
#         [192 128 128],
#         [  0  64   0],
#         [128  64   0],
#         [  0 192   0],
#         [128 192   0],
#         [  0  64 128]]

colors = np.array([[0, 0, 0], 
                   [128, 0, 0],
                   [0, 128, 0], 
                   [128, 128, 0], 
                   [0, 0, 128], 
                   [128, 0, 128], 
                   [0, 128, 128], 
                   [128, 128, 128],
                   [64, 0, 0], 
                   [192, 0, 0], 
                   [64, 128, 0], 
                   [192, 128, 0], 
                   [64, 0, 128], 
                   [192, 0, 128], 
                   [64, 128, 128], 
                   [192, 128, 128], 
                   [0, 64, 0], 
                   [128, 64, 0], 
                   [0, 192, 0], 
                   [128, 192, 0], 
                   [0, 64, 128]], dtype=np.uint8)

def Visualize(output):
    # output: [B, 224, 224]
    output = output.detach().cpu().numpy()
    
    # 创建一个空的彩色输出数组
    B, H, W = output.shape
    color_output = np.zeros((B, H, W, 3), dtype=np.uint8)
    
    # 将 output 作为颜色索引，使用 NumPy 的高级索引功能进行映射
    valid_mask = output < len(colors)
    color_output[valid_mask] = colors[output[valid_mask]]
    
    # 对于超出 colors 范围的索引，设置为白色
    invalid_mask = ~valid_mask

    # Debug
    # print(color_output[invalid_mask])

    color_output[invalid_mask] = [255, 255, 255]

    return color_output

color_map = {
    0: (0, 0, 0, 0.7),  # 黑色
    1: (1.0, 0, 0, 0.5),    # 半透明红色
    2: (0, 1.0, 0, 0.5),    # 半透明绿色
    3: (0, 0, 1.0, 0.5),    # 半透明蓝色
    4: (1.0, 1.0, 0, 0.5),  # 半透明黄色
    5: (1.0, 0, 1.0, 0.5)   # 半透明紫色
}

def apply_mask_with_transparency(image_tensor, mask_tensor, color_map=color_map):
    # 确保image_tensor 和 mask_tensor 是正确的形状
    # assert image_tensor.shape[0] == 3, "Image tensor should have 3 channels (C, H, W)"
    # assert len(mask_tensor.shape) == 2, "Mask tensor should have 2 dimensions (H, W)"
    # assert image_tensor.shape[1:] == mask_tensor.shape, "Image and mask must have the same height and width"

    # print("Image tensor shape:", image_tensor.shape)
    # print("Mask tensor shape:", mask_tensor.shape)

    # 将image_tensor 转换为RGBA格式 (增加alpha通道)
    alpha_tensor = torch.ones((1, *mask_tensor.shape), dtype=torch.float32)
    image_rgba_tensor = torch.cat([image_tensor, alpha_tensor], dim=0)

    # print("Image RGBA tensor shape:", image_rgba_tensor.shape)

    # 遍历mask并应用颜色
    overlay_tensor = torch.zeros_like(image_rgba_tensor)
    for mask_value, color in color_map.items():
        mask_indices = mask_tensor == mask_value
        for i in range(6):
            overlay_tensor[i][mask_indices] = color[i]

    # print("Overlay tensor shape:", overlay_tensor.shape)

    # 合成图像
    combined_tensor = image_rgba_tensor * (1 - overlay_tensor[3:]) + overlay_tensor * overlay_tensor[3:]

    # print("Combined tensor shape:", combined_tensor.shape)

    return combined_tensor



def cosine_similarity(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return x @ y.transpose(-2, -1)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
