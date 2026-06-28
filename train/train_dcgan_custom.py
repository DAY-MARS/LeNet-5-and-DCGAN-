import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

torch.manual_seed(42)

BATCH_SIZE = 128
LATENT_DIM = 100
IMAGE_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0002
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理 —— 自建灰度图
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载自建数据集（my_digits 文件夹）
dataset = datasets.ImageFolder(root='./my_digits', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- 生成器、判别器定义（同原脚本，保持）-----
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)
img_list = []

print("开始训练 DCGAN（自建数据集）...")
for epoch in range(EPOCHS):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(DEVICE)
        batch_size_current = real_imgs.size(0)
        
        real_labels = torch.full((batch_size_current,), 1.0, device=DEVICE)
        fake_labels = torch.full((batch_size_current,), 0.0, device=DEVICE)

        D.zero_grad()
        output_real = D(real_imgs)
        lossD_real = criterion(output_real, real_labels)
        lossD_real.backward()

        noise = torch.randn(batch_size_current, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = G(noise)
        output_fake = D(fake_imgs.detach())
        lossD_fake = criterion(output_fake, fake_labels)
        lossD_fake.backward()
        optimizerD.step()

        G.zero_grad()
        output_fake = D(fake_imgs)
        lossG = criterion(output_fake, real_labels)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}] Batch {i}/{len(dataloader)} '
                  f'Loss D: {lossD_real+lossD_fake:.4f}, Loss G: {lossG:.4f}')

    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        vutils.save_image(fake, f'custom_epoch_{epoch+1}.png', padding=2, normalize=True)

print("训练完成！")
torch.save(G.state_dict(), "dcgan_custom_generator.pth")
torch.save(D.state_dict(), "dcgan_custom_discriminator.pth")
print("模型已保存")

# 最终展示
real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images (Custom Dataset)")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images (Epoch {})".format(EPOCHS))
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()