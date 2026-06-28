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
