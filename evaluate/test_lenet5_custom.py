import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LeNet‑5 网络定义（与训练完全一致）
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

print("正在加载自建数据集测试模型 lenet5_custom.pth ...")
model = LeNet5().to(DEVICE)
model.load_state_dict(torch.load("lenet5_custom.pth", map_location=DEVICE))
model.eval()
print("模型加载成功，准备测试。")

# 预处理（自建数据集）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = datasets.ImageFolder(root='./my_digits', transform=transform)
print(f"自建数据集总样本数: {len(full_dataset)}")

# 随机抽12张
indices = np.random.choice(len(full_dataset), 12, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for idx, ax in zip(indices, axes.flat):
    img, label = full_dataset[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(DEVICE)).argmax().item()

    img_display = img.squeeze().cpu().numpy() * 0.5 + 0.5
    ax.imshow(img_display, cmap='gray')
    color = 'green' if pred == label else 'red'
    ax.set_title(f'真实: {label}  预测: {pred}', color=color, fontsize=10)
    ax.axis('off')

plt.suptitle('LeNet-5 在自建数据集上的预测结果（绿色=正确，红色=错误）', fontsize=14)
plt.tight_layout()
plt.savefig('lenet5自定义预测结果.png', dpi=150)
print("预测结果图已保存为 lenet5自定义预测结果.png")
plt.show()