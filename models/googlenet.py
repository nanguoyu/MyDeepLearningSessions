"""
@File : googlenet.py
@Author: Dong Wang
@Date : 1/28/2022
"""
from collections import namedtuple
import torch
import torch.nn as nn
from typing import Any, Optional
import torch.nn.functional as F

__all__ = ['GoogLeNet', 'googlenet']

# -----------------------Forked from Pytorch-------------------------------------------------------------------------#
# https://github.com/pytorch/vision/blob/8e874ff86701fd6d881c9f845c135a649e393218/torchvision/models/googlenet.py#L20

GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": torch.Tensor,
                                    "aux_logits2": Optional[torch.Tensor],
                                    "aux_logits1": Optional[torch.Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


# -----------------------Forked from Pytorch-------------------------------------------------------------------------#


class BasicConv2d(nn.Module):
    """Forked from Pytorch implementation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    """It actually is the GoogLeNet v2 / Inception.
    Batch normalization is an improvement in InceptionNet compared to the original GoogLeNet.
    """

    def __init__(self,
                 in_channels: int,
                 ch1x1: int,
                 ch3x3reduce: int,
                 ch3x3: int,
                 ch5x5reduce: int,
                 ch5x5: int,
                 pool_proj: int):
        super(Inception, self).__init__()
        """
        There are 4 branches:
        1. 1x1 Conv
        2. 1x1 Conv + 3x3 Conv
        3. 1x1 Conv + 5x5 Conv
        4. 3x3 MaxPool + 1x1 Conv
        """
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2),
            # BasicConv2d(in_channels, ch5x5, kernel_size=3, padding=1), in Pytorch. It maybe a bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),  # ceil_mode=True in Pytorch. I don't know
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 dropout: float = 0.7):
        super(AuxiliaryClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class GoogLeNet(nn.Module):
    """The implementation of GoogLeNet v2/ InceptionNet
    GoogLeNet Paper:https://arxiv.org/abs/1409.4842
    InceptionNet Paper:https://proceedings.mlr.press/v37/ioffe15.html
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, dropout_aux: float = 0.7):
        super(GoogLeNet, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # 4a -> auxiliary classifier 1
        self.aux1 = AuxiliaryClassifier(512, num_classes, dropout=dropout_aux)
        # 4a -> inception4b
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 4d -> auxiliary classifier 2
        self.aux2 = AuxiliaryClassifier(528, num_classes, dropout=dropout_aux)
        # 4d -> inception4e
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 5b -> classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weight()

    def forward(self, x):
        """
        See Table 1: GoogLeNet incarnation of the Inception architecture in  <https://arxiv.org/pdf/1409.4842.pdf>
        """
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[torch.Tensor] = None
        if self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[torch.Tensor] = None
        if self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training:
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-1, b=2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def googlenet(**kwargs: Any) -> GoogLeNet:
    model = GoogLeNet(**kwargs)
    return model
