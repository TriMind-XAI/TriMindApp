# data/image_loader.py

from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def get_pneumonia_loaders(batch_size=64, size=28):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = PneumoniaMNIST(
        split="train",
        download=True,
        size=size,
        transform=transform
    )

    test_dataset = PneumoniaMNIST(
        split="test",
        download=True,
        size=size,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader