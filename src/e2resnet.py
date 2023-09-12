from typing import Callable, List, Optional, Type, Union

import torch
from torch import Tensor

from escnn import gspaces, nn
from escnn.nn import FieldType, GeometricTensor


def conv7x7(
    in_type: FieldType,
    out_type: FieldType,
    stride: int = 1,
    padding: int = 3,
    dilation: int = 1,
    bias: bool = False,
    initialize: bool = True,
):
    return nn.R2Conv(
        in_type,
        out_type,
        7,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        initialize=initialize,
    )


def conv3x3(
    in_type: FieldType,
    out_type: FieldType,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    bias: bool = False,
    initialize: bool = True,
):
    return nn.R2Conv(
        in_type,
        out_type,
        3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        initialize=initialize,
    )


def conv1x1(
    in_type: FieldType,
    out_type: FieldType,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    bias: bool = False,
    initialize: bool = True,
):
    return nn.R2Conv(
        in_type,
        out_type,
        1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        initialize=initialize,
    )


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
        initialize: bool = True,
    ):
        super().__init__()
        in_type = nn.FieldType(gspace, inplanes * [gspace.regular_repr])
        out_type = nn.FieldType(gspace, planes * [gspace.regular_repr])

        self.conv1 = conv3x3(in_type, out_type, stride=stride, initialize=initialize)
        self.bn1 = norm_layer(out_type)
        self.relu = nn.ReLU(out_type, True)
        self.conv2 = conv3x3(out_type, out_type, stride=1, initialize=initialize)
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
        initialize: bool = True,
    ):
        super().__init__()
        in_type = FieldType(gspace, inplanes * [gspace.regular_repr])
        out_type = FieldType(gspace, planes * [gspace.regular_repr])
        exp_type = FieldType(gspace, self.expansion * planes * [gspace.regular_repr])

        self.conv1 = conv1x1(in_type, out_type, initialize=initialize)
        self.bn1 = norm_layer(out_type)
        self.relu1 = nn.ReLU(out_type, True)

        self.conv2 = conv3x3(out_type, out_type, stride=stride, initialize=initialize)
        self.bn2 = norm_layer(out_type)
        self.relu2 = nn.ReLU(out_type, True)

        self.conv3 = conv1x1(out_type, exp_type, initialize=initialize)
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
        initialize: bool = True,
    ):
        super().__init__()

        self.gspace = gspace
        self.in_type = FieldType(gspace, 3 * [gspace.trivial_repr])

        self.norm_layer = nn.InnerBatchNorm
        self.dilation = 1
        self.base_width = base_width
        self.inplanes = self.base_width

        out_type = FieldType(gspace, self.base_width * [gspace.regular_repr])
        self.conv1 = conv7x7(self.in_type, out_type, stride=2, initialize=initialize)
        self.bn1 = self.norm_layer(out_type)
        self.maxpool = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        self.relu1 = nn.ReLU(out_type, True)
        self.layer1 = self._make_layer(gspace, block, base_width, layers[0])
        self.layer2 = self._make_layer(
            gspace, block, base_width * 2, layers[1], stride=2, initialize=initialize
        )
        self.layer3 = self._make_layer(
            gspace, block, base_width * 4, layers[2], stride=2, initialize=initialize
        )
        self.layer4 = self._make_layer(
            gspace, block, base_width * 8, layers[3], stride=2, initialize=initialize
        )

        fmap_type = FieldType(
            gspace, block.expansion * base_width * 8 * [gspace.regular_repr]
        )
        self.avgpool = nn.PointwiseAdaptiveAvgPool(fmap_type, (1, 1))

        out_type = FieldType(gspace, num_classes * [gspace.trivial_repr])
        self.fc = nn.R2Conv(fmap_type, out_type, kernel_size=1)

    def _make_layer(
        self,
        gspace: gspaces.GSpace,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        initialize: bool = True,
    ) -> torch.nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            in_type = FieldType(gspace, self.inplanes * [gspace.regular_repr])
            out_type = FieldType(gspace, planes * [gspace.regular_repr])
            downsample = nn.SequentialModule(
                conv1x1(in_type, out_type, stride=stride, initialize=initialize),
                self.norm_layer(out_type),
            )

        layers = []
        layers.append(
            block(
                gspace,
                self.inplanes,
                planes,
                stride,
                downsample,
                self.norm_layer,
                initialize=initialize,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    gspace,
                    self.inplanes,
                    planes,
                    1,
                    None,
                    self.norm_layer,
                    initialize=initialize,
                )
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
        x = self.fc(x).tensor
        return x.squeeze(-2).squeeze(-1)
