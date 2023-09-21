import torch
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


def c8resnet18(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=28,
        initialize=initialize,
    )
    model.name = "c8resnet18"
    return model


def c1resnet50(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=70,
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
        base_width=50,
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
        base_width=35,
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
        base_width=25,
        initialize=initialize,
    )
    model.name = "d4resnet50"
    return model


def c8resnet50(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=25,
        initialize=initialize,
    )
    model.name = "c8resnet50"
    return model


def c1resnet101(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=70,
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
        base_width=50,
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
        base_width=35,
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
        base_width=25,
        initialize=initialize,
    )
    model.name = "d4resnet101"
    return model


def c8resnet101(initialize: bool = True):
    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=25,
        initialize=initialize,
    )
    model.name = "c8resnet101"
    return model


def count_params(m: torch.nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_inference(
    model: torch.nn.Module, batch_size: int = 1, repetitions: int = 300
):
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    dummy_input = torch.randn((batch_size, 3, 224, 224), dtype=torch.float32).cuda()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    # gpu warmup
    for _ in range(10):
        _ = model(dummy_input)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    return sum(timings) / len(timings)


def create_model(model_name: str):
    return eval(model_name)()


if __name__ == "__main__":
    for layers in [18, 50, 101]:
        # for group in ["", "c1", "d1", "c4", "d4"]:
        for group in ["", "c8"]:  # "c1", "d1", "c4", "d4"]:
            model = eval(group + "resnet" + str(layers))(False).cuda().eval()
            mem = torch.cuda.memory_allocated()
            inf_time = measure_inference(model)
            print(
                f"{model.name}: {count_params(model)*1e-6:.1f}M |"
                f" {inf_time:.1f}ms | {mem * 1e-9:.2f}Gb"
            )
        print()
