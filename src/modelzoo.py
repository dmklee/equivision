from escnn import gspaces

from src.e2resnet import E2ResNet, E2BasicBlock


def c1resnet18(num_classes: int=1000, base_width: int=80):
    return E2ResNet(
        gspaces.trivialOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=base_width,
    )


def d1resnet18(num_classes: int=1000, base_width: int=54):
    return E2ResNet(
        gspaces.flip2dOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=base_width,
    )


def c4resnet18(num_classes: int=1000, base_width: int=40):
    return E2ResNet(
        gspaces.rot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=54,
    )


def d4resnet18(num_classes: int=1000, base_width: int=28):
    return E2ResNet(
        gspaces.flipRot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=54,
    )


def create_model(model_name, num_classes: int=1000):
    return eval(model_name)(num_classes=num_classes)
