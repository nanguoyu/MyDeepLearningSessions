from torchvision import transforms
from torchvision import datasets
import torch
import os


def tiny_imagenet(batch_size, data_dir):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_val = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])

    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    valset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, num_label
