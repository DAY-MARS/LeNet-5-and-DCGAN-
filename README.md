# LeNet-5 & DCGAN：手写数字识别与生成实战


> 本项目系统性地实现了卷积神经网络的两大经典里程碑：**LeNet-5**（图像分类）与 **DCGAN**（图像生成）。从标准 MNIST 到自建手写数据集，完整覆盖了训练、评估、生成及“生成→识别”的联动推理全流程。

---

## 📖 目录
1. [项目结构](#-项目结构)
2. [环境配置](#-环境配置)
3. [数据集准备](#-数据集准备)
4. [训练模型](#-训练模型)
5. [评估与可视化](#-评估与可视化)
6. [核心联动：用 LeNet-5 识别 DCGAN 生成的图片](#-核心联动用-lenet-5-识别-dcgan-生成的图片)
7. [许可证](#-许可证)

---

## 🗂 项目结构

本项目严格按照功能模块划分，目录结构如下：

```text
LeNet-5-and-DCGAN/
├── README.md                       # 项目说明（运行指南、环境配置、复现步骤）
├── requirements.txt                # Python 依赖列表
├── .gitignore                      # 忽略上传的文件（数据集、模型、缓存等）
├── LICENSE                         # MIT 许可证
│
├── model/                          # 网络模型定义（可复用）
│   ├── lenet5.py                   # LeNet-5 网络类
│   └── dcgan.py                    # DCGAN 生成器 + 判别器类
│
├── data/                           # 数据集处理
│   ├── generate_dataset.py         # 自建数据集生成（标准版）
│   ├── generate_dataset_bold.py    # 自建数据集生成（加粗版）
│   └── dataset_utils.py            # 数据加载、预处理、划分工具函数
│
├── train/                          # 训练脚本
│   ├── train_lenet5_mnist.py       # LeNet-5 在 MNIST 上的训练
│   ├── train_lenet5_custom.py      # LeNet-5 在自建数据集上的训练
│   ├── train_dcgan_mnist.py        # DCGAN 在 MNIST 上的训练
│   └── train_dcgan_custom.py       # DCGAN 在自建数据集上的训练
│
├── evaluate/                       # 评估、生成与联动验证
│   ├── test_lenet5_mnist.py        # LeNet-5 在 MNIST 上的预测可视化
│   ├── test_lenet5_custom.py       # LeNet-5 在自建数据集上的预测可视化
│   ├── generate_gan_images.py      # DCGAN（MNIST版）生成图片
│   ├── generate_gan_images_custom.py  # DCGAN（自建版）生成图片
│   ├── recognize_gan_mnist.py      # 联动识别（MNIST版）：DCGAN→LeNet-5
│   └── recognize_gan_custom.py     # 联动识别（自建版）：DCGAN→LeNet-5
│
└── visual/                         # 数据集可视化
    ├── show_mnist_samples.py       # MNIST 样本展示
    └── show_custom_samples.py      # 自建数据集样本展示
```
---

## ⚙️ 环境配置

### 硬件要求
- GPU：NVIDIA 显卡（推荐，至少 4 GB 显存），本项目使用 **NVIDIA GeForce RTX 4050 (6 GB)**
- CPU：Intel Core i7-13700H 或同等性能
- 内存：16 GB 及以上

### 软件环境
- 操作系统：Windows 11（Linux / macOS 亦可，只需调整对应命令）
- Python：**3.9.x**（推荐 3.9.25）
- 包管理：**Miniconda** 或 Anaconda
- 深度学习框架：**PyTorch 2.7.1 + CUDA 11.8**

### 1. 安装 Miniconda（如果还没有）
前往 [Miniconda 官网](https://docs.conda.io/en/latest/miniconda.html) 下载 Windows 版并安装。  
安装时勾选 “Add Miniconda3 to my PATH environment variable”。

### 2. 创建并激活虚拟环境
打开终端（PowerShell 或 Git Bash），执行：
```bash
conda create -n pytorch_env python=3.9 -y
conda activate pytorch_env
```
---

## 📦 数据集准备

本项目使用两种数据集：**MNIST 公开数据集**和**自建合成手写数字数据集**。

### 1. MNIST 数据集

| 属性 | 说明 |
|:---|:---|
| 来源 | Yann LeCun 官网 |
| 规模 | 60,000 训练 + 10,000 测试 |
| 尺寸 | 28×28 灰度图（Pad 至 32×32） |
| 类别 | 0–9 共 10 类 |
| 镜像地址 | https://mirrors.tuna.tsinghua.edu.cn/torchvision-datasets/mnist/（已失效） |

#### 自动下载

运行训练脚本时，`torchvision` 会自动从官网下载 MNIST 数据集并保存在 `./data` 目录下：

```bash
python train/train_lenet5_mnist.py

```
#### 手动下载（网络受限或 SSL 错误时使用）

在实际配置过程中，曾遇到以下问题：
- 直接连接 Yann LeCun 官网时出现 **SSL 证书验证失败** 错误
- 清华大学 TUNA 镜像的部分 MNIST 链接也已失效

**解决方案**：从 PyTorch 官方的 Amazon S3 备份源手动下载，该源在国内大部分地区可正常访问。

**下载地址（S3 备份源）：**

以下四个文件必须全部下载，且**不要解压**：

https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz

https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz

https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz

https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
（目前网址已失效，但是MNIST训练集容易寻找，请读者自行在网上查询）
#### （1）放置数据文件

将四个压缩包放入 `./data/MNIST/raw/` 文件夹，最终目录结构应如下：

```text
data/
└── MNIST/
    └── raw/
        ├── train-images-idx3-ubyte.gz
        ├── train-labels-idx1-ubyte.gz
        ├── t10k-images-idx3-ubyte
```
 #### （2）修改训练脚本
 
将数据集加载代码中的 download=True 改为 download=False，以使用本地文件，示例如下：
```python
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
```
#### 💡 补充说明
若 S3 源也出现连接问题，可尝试在代码中临时关闭 SSL 验证：
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

### 2. 自建合成手写数字数据集数据集

| 属性 | 说明 |
|:---|:---|
| 来源 | Windows 系统字体程序化合成 |
| 规模 |5,000 张（每类 500 张）|
| 尺寸 |32×32 灰度图 |
| 类别 | 0–9 共 10 类 |
| 生成方式 |Python PIL 库随机变换字体、字号、偏移、旋转 |
---

#### 生成流程

1.随机选取一款系统字体

2.在 32×32 白底画布上绘制数字

3.施加随机偏移和旋转

4.保存为 PNG 灰度图

#### 生成命令
```bash
# 标准版（常规笔画粗细）
python data/generate_dataset.py

# 加粗版（笔画更粗，优先使用此版本以获得更好的训练效果）
python data/generate_dataset_bold.py
```
---
#### 输出结构目录
```text
my_digits/
├── 0/        # 数字 0（500 张）
├── 1/        # 数字 1（500 张）
├── 2/        # 数字 2（500 张）
├── 3/        # 数字 3（500 张）
├── 4/        # 数字 4（500 张）
├── 5/        # 数字 5（500 张）
├── 6/        # 数字 6（500 张）
├── 7/        # 数字 7（500 张）
├── 8/        # 数字 8（500 张）
└── 9/        # 数字 9（500 张）
```
---
### 3.数据预处理

#### MNIST预处理
```python
from torchvision import transforms
from torchvision import transforms

transform = transforms.Compose([
    transforms.Pad(2),                              # 28×28 → 32×32（适配 LeNet-5 输入层）
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))      # 标准化为均值 0、标准差 1
])
```
---
#### 参数说明

0.1307：MNIST 训练集全局均值

0.3081：MNIST 训练集全局标准差

标准化后数据分布接近标准正态分布，有助于加速 SGD 收敛

#### 自建数据集预处理
```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),    # ImageFolder 默认读为 RGB，需转回单通道
    transforms.Resize((32, 32)),                    # 保持尺寸一致
    transforms.ToTensor(),                          # [0,255] → [0,1]
    transforms.Normalize((0.5,), (0.5,))            # 归一化至 [-1, 1]
])
```
---
#### 参数说明
text
Grayscale(num_output_channels=1)：关键步骤！自建图片虽是灰度图，但 ImageFolder 默认转换为 RGB 三通道，必须显式转回单通道，否则卷积层输入通道数不匹配

Normalize((0.5,), (0.5,))：将 [0,1] 映射至 [-1,1]，与 LeNet-5 的 Tanh 激活函数输出范围及 DCGAN 生成器的 Tanh 输出层保持一致

#### DCGAN 预处理（两个数据集统一）
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))            # 匹配 Tanh 输出范围 [-1, 1]
])
```
---
DCGAN 的生成器输出层使用 Tanh 激活函数（输出范围 [-1, 1]），因此真实图像也需归一化到相同范围，以便判别器进行对比。
## 🏃 训练模型

### LeNet-5 训练

```bash
# 在 MNIST 上训练
python train/train_lenet5_mnist.py

# 在自建数据集上训练
python train/train_lenet5_custom.py
```
---
训练完成后，项目根目录会生成对应的模型权重文件：
lenet5_mnist.pth（MNIST 版，准确率很高，某次训练后准确率为 98.86%）
lenet5_custom.pth（自建版，准确率不高，某次训练后准确率 86.00%）

### DCGAN 训练
```bash
# 在 MNIST 上训练
python train/train_dcgan_mnist.py

# 在自建数据集上训练
python train/train_dcgan_custom.py
```
---
CGAN 训练时间较长（约 30 个 epoch），过程中会每 epoch 保存一张生成示例图，训练结束后保存生成器和判别器权重：

dcgan_generator.pth / dcgan_discriminator.pth（MNIST 版）

dcgan_custom_generator.pth / dcgan_custom_discriminator.pth（自建版）

💡 如果想跳过训练直接测试，可使用仓库中预置的权重文件（*.pth）。

## 🔍 评估与可视化

### LeNet-5 单模型测试
```bash
# 在 MNIST 上测试
python evaluate/test_lenet5_mnist.py

# 在自建数据集上测试
python evaluate/test_lenet5_custom.py
```
---
每个脚本会随机抽取 12 张测试图像，用训练好的模型进行预测，并显示预测结果（绿色=正确，红色=错误），同时保存为 PNG 图片。

### DCGAN 生成图片
```bash
# 用 MNIST 版 DCGAN 生成 64 张图片
python evaluate/generate_gan_images.py

# 用自建版 DCGAN 生成 64 张图片
python evaluate/generate_gan_images_custom.py
```
---
生成的图片分别保存在 dcgan_generated_mnist/ 和 dcgan_generated/ 目录中。

## 🔗 核心联动：用 LeNet-5 识别 DCGAN 生成的图片
这是本项目最有特色的实验：让 LeNet-5 去“认一认”DCGAN 凭空画出来的数字。
```bash
# 自建版联动：自建 DCGAN 生成 → 自建 LeNet-5 识别
python evaluate/recognize_gan_custom.py

# MNIST 版联动：MNIST DCGAN 生成 → MNIST LeNet-5 识别
python evaluate/recognize_gan_mnist.py
```
---
每个脚本会输出每张生成图片的识别数字和置信度，并保存可视化结果图。

| 联动版本 | DCGAN 训练数据 | LeNet-5 训练数据 | 识别效果 |
|---------|---------------|----------------|---------|
| MNIST 版 | MNIST | MNIST | 多数以高置信度正确识别 |
| 自建版 | 自建数据集 | 自建数据集 | 置信度偏低，误判较多 |
