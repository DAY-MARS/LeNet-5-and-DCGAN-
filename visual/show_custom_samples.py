import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torchvision.utils as vutils

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root='./my_digits', transform=transform)

indices = np.random.choice(len(full_dataset), 64, replace=False)
images = [full_dataset[idx][0] for idx in indices]
batch = torch.stack(images)

grid = vutils.make_grid(batch, nrow=8, padding=2, normalize=False)

plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
plt.title('自建数据集样本（随机抽取 64 张）', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('自建数据集样本展示.png', dpi=150)
print("样本图已保存为 自建数据集样本展示.png")
plt.show()