import torchvision.models

from escnn import gspaces
from src.e2resnet import E2BasicBlock, E2BottleNeck, E2ResNet


# pytorch versions
def resnet18(*args, **kwargs):
    model = torchvision.models.resnet18()
    model.name = "resnet18"
    return model


def resnet50(*args, **kwargs):
    model = torchvision.models.resnet50()
    model.name = "resnet50"
    return model


def resnet101(*args, **kwargs):
    model = torchvision.models.resnet101()
    model.name = "resnet101"
    return model


def c1resnet18(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=78,
        initialize=initialize,
    )
    model.name = "c1resnet18"
    return model


def d1resnet18(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=55,
        initialize=initialize,
    )
    model.name = "d1resnet18"
    return model


def c4resnet18(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=39,
        initialize=initialize,
    )
    model.name = "c4resnet18"
    return model


def d4resnet18(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=28,
        initialize=initialize,
    )
    model.name = "d4resnet18"
    return model


def c1resnet50(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=74,
        initialize=initialize,
    )
    model.name = "c1resnet50"
    return model


def d1resnet50(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=53,
        initialize=initialize,
    )
    model.name = "d1resnet50"
    return model


def c4resnet50(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=38,
        initialize=initialize,
    )
    model.name = "c4resnet50"
    return model


def d4resnet50(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=27,
        initialize=initialize,
    )
    model.name = "d4resnet50"
    return model


def c1resnet101(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=72,
        initialize=initialize,
    )
    model.name = "c1resnet101"
    return model


def d1resnet101(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=51,
        initialize=initialize,
    )
    model.name = "d1resnet101"
    return model


def c4resnet101(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=36,
        initialize=initialize,
    )
    model.name = "c4resnet101"
    return model


def d4resnet101(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=26,
        initialize=initialize,
    )
    model.name = "d4resnet101"
    return model


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    for layers in [18, 50, 101][::-1]:
        for group in ["", "c1", "d1", "c4", "d4"]:
            m = eval(group + "resnet" + str(layers))(False)
            print(f"{m.name}: {count_params(m)*1e-6:.1f}M")
        print()
