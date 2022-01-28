"""
@File : vgg.py
@Author: Dong Wang
@Date : 1/28/2022
"""
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19']


class VGG(nn.Module):
    """The original implementation of VGGNet
    Paper: https://arxiv.org/pdf/1409.1556.pdf
    """

    def __init__(self, features: nn.Module, num_classes=1000):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 224/(2^5)=7
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        self._init_weight()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


cfgs: Dict[int, List[Union[str, int]]] = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(num_weighted_layers, **kwargs: Any) -> VGG:
    layers: List[nn.Module] = []
    features_cfg = cfgs[num_weighted_layers]
    in_channels = 3
    for i in features_cfg:
        if isinstance(i, int):
            layers.append(nn.Conv2d(in_channels, i, kernel_size=(3, 3), padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = i
        elif i == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    model = VGG(nn.Sequential(*layers), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    model = _vgg(11, **kwargs)
    return model


def vgg13(**kwargs: Any) -> VGG:
    model = _vgg(13, **kwargs)
    return model


def vgg16(**kwargs: Any) -> VGG:
    model = _vgg(16, **kwargs)
    return model


def vgg19(**kwargs: Any) -> VGG:
    model = _vgg(19, **kwargs)
    return model
