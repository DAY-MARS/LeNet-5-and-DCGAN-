import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置随机种子，保证可复现
torch.manual_seed(42)

# 超参数
BATCH_SIZE = 128
LATENT_DIM = 100        # 噪声向量长度
IMAGE_SIZE = 32         # 生成图像大小（LeNet-5 也用 32x32）
EPOCHS = 30
LEARNING_RATE = 0.0002
BETA1 = 0.5             # Adam 的 beta1 参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理（MNIST → 32x32，并归一化到 [-1, 1] 以适应 Tanh 输出）
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST（本地数据，不下载）
dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------- 生成器 --------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入：LATENT_DIM x 1 x 1
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 输出：256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 输出：128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 输出：64 x 16 x 16
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出：1 x 32 x 32
        )

    def forward(self, x):
        return self.main(x)

# -------------------- 判别器 --------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入：1 x 32 x 32
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出：64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出：128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出：256 x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出：1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# 初始化模型
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# 固定噪声，用来可视化训练过程中生成的变化
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

# 用于存储生成的图片
img_list = []

# -------------------- 训练循环 --------------------
print("开始训练 DCGAN...")
for epoch in range(EPOCHS):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(DEVICE)
        batch_size_current = real_imgs.size(0)
        
        # 真实标签和虚假标签（0 或 1）
        real_labels = torch.full((batch_size_current,), 1.0, device=DEVICE)
        fake_labels = torch.full((batch_size_current,), 0.0, device=DEVICE)

        # ========== 训练判别器 ==========
        D.zero_grad()
        # 对真实图片的判别
        output_real = D(real_imgs)
        lossD_real = criterion(output_real, real_labels)
        lossD_real.backward()

        # 对假图片的判别
        noise = torch.randn(batch_size_current, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = G(noise)
        output_fake = D(fake_imgs.detach())   # 不更新生成器
        lossD_fake = criterion(output_fake, fake_labels)
        lossD_fake.backward()
        optimizerD.step()

        # ========== 训练生成器 ==========
        G.zero_grad()
        output_fake = D(fake_imgs)            # 这次需要梯度
        lossG = criterion(output_fake, real_labels)  # 希望判别器认为它是真
        lossG.backward()
        optimizerG.step()

        # 打印进度
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}] Batch {i}/{len(dataloader)} '
                  f'Loss D: {lossD_real+lossD_fake:.4f}, Loss G: {lossG:.4f}')

    # 每个 epoch 结束后，用固定噪声生成一组图片保存
    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        # 保存到本地
        vutils.save_image(fake, f'epoch_{epoch+1}.png', padding=2, normalize=True)

print("训练完成！")

# 保存模型
torch.save(G.state_dict(), "dcgan_generator.pth")
torch.save(D.state_dict(), "dcgan_discriminator.pth")
print("模型已保存")

# -------------------- 展示训练结果 --------------------
# 对比真实图片
real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 展示最后一个 epoch 生成的图片
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images (Epoch {})".format(EPOCHS))
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()