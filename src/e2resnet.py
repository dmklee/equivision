from typing import Callable, List, Optional, Type, Union

import torch
from escnn import gspaces, nn
from escnn.nn import GeometricTensor
from torch import Tensor


class E2BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        gspace: gspaces.GSpace2D,
        inplanes: int,
        planes: int,
        stride: int,
        downsample: Optional[Callable] = None,
        norm_layer: Optional[Callable] = nn.InnerBatchNorm,
    ):
        super().__init__()
        in_type = nn.FieldType(gspace, inplanes * [gspace.regular_repr])
        out_type = nn.FieldType(gspace, planes * [gspace.regular_repr])

        self.conv1 = nn.R2Conv(
            in_type, out_type, 3, padding=1, stride=stride, bias=False
        )
        self.bn1 = norm_layer(out_type)
        self.relu = nn.ReLU(out_type, True)
        self.conv2 = nn.R2Conv(out_type, out_type, 3, padding=1, stride=1, bias=False)
        self.bn2 = norm_layer(out_type)
        self.downsample = downsample

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class E2BottleNeck(torch.nn.Module):
    expansion: int = 4

    def __init__(
        self,
        gspace: gspaces.GSpace2D,
        inplanes: int,
        planes: int,
        stride: int,
        downsample: Optional[Callable] = None,
        norm_layer: Optional[Callable] = nn.InnerBatchNorm,
    ):
        in_type = nn.FieldType(gspace, inplanes * [gspace.regular_repr])
        out_type = nn.FieldType(gspace, planes * [gspace.regular_repr])
        exp_type = nn.FieldType(gspace, self.expansion * planes * [gspace.regular_repr])

        self.conv1 = nn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_type)
        self.relu1 = nn.ReLU(out_type, True)

        self.conv2 = nn.R2Conv(
            out_type,
            out_type,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(out_type)
        self.relu2 = nn.ReLU(out_type, True)

        self.conv3 = nn.R2Conv(out_type, exp_type, kernel_size=1, bias=False)
        self.bn3 = norm_layer(exp_type)
        self.relu3 = nn.ReLU(exp_type, True)

        self.downsample = downsample

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class E2ResNet(torch.nn.Module):
    def __init__(
        self,
        gspace: gspaces.GSpace2D,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        layers: List[int],
        num_classes: int,
        base_width: int,
    ):
        super().__init__()

        self.gspace = gspace
        self.in_type = nn.FieldType(gspace, 3 * [gspace.trivial_repr])

        self.norm_layer = nn.InnerBatchNorm
        self.dilation = 1
        self.base_width = base_width
        self.inplanes = self.base_width

        out_type = nn.FieldType(gspace, self.base_width * [gspace.regular_repr])
        self.conv1 = nn.R2Conv(
            self.in_type,
            out_type,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self.norm_layer(out_type)
        self.maxpool = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        self.relu1 = nn.ReLU(out_type, True)
        self.layer1 = self._make_layer(gspace, block, base_width, layers[0])
        self.layer2 = self._make_layer(
            gspace, block, base_width * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            gspace, block, base_width * 4, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            gspace, block, base_width * 8, layers[3], stride=2
        )

        out_type = nn.FieldType(
            gspace, block.expansion * base_width * 8 * [gspace.regular_repr]
        )
        self.avgpool = nn.PointwiseAdaptiveAvgPool(out_type, (1, 1))

        self.gpool = nn.GroupPooling(out_type)
        self.fc = torch.nn.Linear(block.expansion * base_width * 8, num_classes)

    def _make_layer(
        self,
        gspace: gspaces.GSpace,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> torch.nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            in_type = nn.FieldType(gspace, self.inplanes * [gspace.regular_repr])
            out_type = nn.FieldType(gspace, planes * [gspace.regular_repr])
            downsample = nn.SequentialModule(
                nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, stride=stride),
                self.norm_layer(out_type),
            )

        layers = []
        layers.append(
            block(gspace, self.inplanes, planes, stride, downsample, self.norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(gspace, self.inplanes, planes, 1, None, self.norm_layer)
            )

        return torch.nn.Sequential(*layers)

    def forward_features(self, x: Tensor) -> GeometricTensor:
        x = GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = self.gpool(x).tensor
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x
