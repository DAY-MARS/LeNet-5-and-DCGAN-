import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torchvision.utils as vutils

# ---- 中文字体 ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ---- 加载自建数据集（不做归一化，保留原始 0-255 灰度） ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保单通道
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                         # 只转张量，不归一化
])

full_dataset = datasets.ImageFolder(root='./my_digits', transform=transform)

# ---- 随机抽取 64 张图片 ----
indices = np.random.choice(len(full_dataset), 64, replace=False)
images = []
for idx in indices:
    img, _ = full_dataset[idx]
    images.append(img)

# 堆叠成 (64, 1, 32, 32) 的张量
batch = torch.stack(images)

# ---- 拼接成 8×8 网格图 ----
grid = vutils.make_grid(batch, nrow=8, padding=2, normalize=False)

# ---- 显示 ----
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
plt.title('自建数据集样本（随机抽取 64 张）', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('自建数据集样本展示.png', dpi=150)
print("样本图已保存为 自建数据集样本展示.png")
plt.show()