"""
@File : mobilenetv2.py
@Author: Dong Wang
@Date : 2/10/2022
"""
from typing import Any, Optional, List

import torch
from torch import Tensor
from torch import nn

__all__ = ["MobileNetV2", "mobilenet_v2"]


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # s can only be 1 or 2
        """ input | operator | output
            h*w*k | 1*1 conv2d, RELU6 | h*w*(tk) 
            h*w*tk | 3*3 dwise s=s, RELU6 | (h/s)*(w/s)*(tk) 
            h*w*k | linear 1*1, conv2d | (h/s)*(w/s)*k'
        """
        hidden_dim = int(round(inp * expand_ratio))  # (tk)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(),
                )
            )
        layers.extend(
            [
                # dw
                nn.Sequential(
                    nn.Conv2d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              stride=self.stride,
                              groups=hidden_dim,
                              padding=(3-1)//2),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6()
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """https://arxiv.org/pdf/1801.04381.pdf"""

    def __init__(self,
                 num_classes=1000,
                 width_mult: float = 1.0,
                 dropout=0.2,
                 round_nearest=8):
        super(MobileNetV2, self).__init__()

        input_channel = 32
        last_channel = 1280

        # The following shows the whole network structure
        # Conv2d  [-, 32, 1, 2 ]
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # conv2d [-, 1280, 1, 1]
        # avgpool [- ,- , 1, -]
        # con2d [ -, k, -, -]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=input_channel,
                          kernel_size=3,
                          stride=2,
                          padding=(3-1)//2),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6()
            )
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # The 1st layer of each sequence has a stride s and all others use stride 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(
            nn.Sequential(
                nn.Conv2d(in_channels=input_channel,
                          out_channels=self.last_channel,
                          kernel_size=1),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6()
            )
        )

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes)
        )
        self._init_weight()

    def forward(self, x) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def mobilenet_v2(**kwargs: Any) -> MobileNetV2:
    model = MobileNetV2(**kwargs)
    return model
