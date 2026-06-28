import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- DCGAN 生成器定义（与训练时完全一致） ----
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
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

# ---- 加载 MNIST 数据集训练好的生成器 ----
LATENT_DIM = 100
generator = Generator(LATENT_DIM).to(DEVICE)
generator.load_state_dict(torch.load("dcgan_generator.pth", map_location=DEVICE))
generator.eval()

# ---- 生成图片 ----
output_dir = "dcgan_generated_mnist"
os.makedirs(output_dir, exist_ok=True)

num_images = 64
noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
with torch.no_grad():
    fake_imgs = generator(noise)

for i in range(num_images):
    img = fake_imgs[i].squeeze().cpu().numpy() * 0.5 + 0.5  # [-1,1] -> [0,1]
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(os.path.join(output_dir, f"{i:04d}.png"))

print(f"已生成 {num_images} 张图片，保存在 '{output_dir}' 文件夹")