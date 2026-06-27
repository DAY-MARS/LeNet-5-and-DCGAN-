import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LeNet-5 网络定义 ----
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

# ---- 中文字体 ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ---- 加载 LeNet-5 ----
lenet = LeNet5().to(DEVICE)
lenet.load_state_dict(torch.load("lenet5_custom.pth", map_location=DEVICE))
lenet.eval()

# ---- 预处理（与自建数据集训练一致） ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---- 加载 DCGAN 生成的图片 ----
input_dir = "dcgan_generated"
fnames = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

preds = []
confidences = []
for fname in fnames:
    img = Image.open(os.path.join(input_dir, fname)).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = lenet(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        conf = prob[0, pred].item()
    preds.append(pred)
    confidences.append(conf)

# ---- 打印结果 ----
print("\nDCGAN 生成图片的 LeNet-5 识别结果：")
for i in range(min(12, len(fnames))):
    print(f"  图片 {i+1}: 识别为 {preds[i]}, 置信度 {confidences[i]:.2%}")

# ---- 绘制前 12 张 ----
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, ax in enumerate(axes.flat):
    img = Image.open(os.path.join(input_dir, fnames[i])).convert('L')
    ax.imshow(img, cmap='gray')
    ax.set_title(f'识别: {preds[i]}\n置信度: {confidences[i]:.2%}', fontsize=10)
    ax.axis('off')

plt.suptitle('DCGAN 生成 → LeNet-5 识别', fontsize=14)
plt.tight_layout()
plt.savefig('dcgan_to_lenet5.png', dpi=150)
print("\n可视化图已保存为 dcgan_to_lenet5.png")
plt.show()