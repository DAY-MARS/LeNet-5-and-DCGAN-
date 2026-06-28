"""数据加载与预处理工具函数"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size=64, train=True, download=False, root='./data'):
    """
    返回 MNIST 数据集对应的 DataLoader。
    """
    transform = transforms.Compose([
        transforms.Pad(2),                  # 28x28 -> 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root=root, train=train, download=download,
                             transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


def get_custom_loader(batch_size=64, root='./my_digits', train=True):
    """
    返回自建数据集（my_digits）的 DataLoader。
    按照 8:2 随机划分训练集与测试集。
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.ImageFolder(root=root, transform=transform)

    # 划分训练/测试集（固定随机种子保证可复现）
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    dataset = train_dataset if train else test_dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader