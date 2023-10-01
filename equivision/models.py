import torch
import torchvision.models
from escnn import gspaces
from torch.hub import load_state_dict_from_url

from equivision.e2resnet import E2BasicBlock, E2BottleNeck, E2ResNet

WEIGHT_URLS = {
    "d1resnet18": "https://drive.google.com/file/d/1LX9--04ZOTv28kZH2WIO8IhL8UI7Wi0f/view?usp=drive_link",
    "c4resnet18": "https://drive.google.com/file/d/1hO04WpgJHH_a0f2eYfClhwBm4SRnC9xM/view?usp=drive_link",
    "d4resnet18": "https://drive.google.com/file/d/19TsJP49g6O16eGihP35Cg5IPXGoNgVaW/view?usp=drive_link",
    "c8resnet18": "https://drive.google.com/file/d/1i4uboCtvyYkhWOqwOAg2A57jb-D8-xCN/view?usp=drive_link",
    "d1resnet50": "https://drive.google.com/file/d/1q6mep0tpIoiZFYWuSi1dPnVQ1Fd60OKn/view?usp=drive_link",
    "c4resnet50": "https://drive.google.com/file/d/1NYTjon1zvghdGmpn4OkbB4xhIX5ixAxI/view?usp=drive_link",
    "d4resnet50": "https://drive.google.com/file/d/1Fr3JQqQFGaL_JjPelZ3gxGhUs5_o0lI8/view?usp=drive_link",
    "c8resnet50": "https://drive.google.com/file/d/13Et3SvIoxRFEy9N8t61O6EeAeameKzLX/view?usp=drive_link",
    "d1resnet101": "https://drive.google.com/file/d/1iRRkAM3JgU0L61YO3LC3zGFFbK0F3f1I/view?usp=drive_link",
    "c4resnet101": "https://drive.google.com/file/d/16N9H6ac_WWzC01wBDW06tTL0HWCTcVmW/view?usp=drive_link",
    "d4resnet101": "https://drive.google.com/file/d/1qmkLJV87lVKFnPdZMEsYoe2rrjpI96PM/view?usp=drive_link",
}


# pytorch versions, with same interface for easy loading during training
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


def c1resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=78 if fixed_params else 64,
        initialize=initialize,
    )
    model.name = "c1resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d1resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=55 if fixed_params else 32,
        initialize=initialize,
    )
    model.name = "d1resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c4resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=39 if fixed_params else 16,
        initialize=initialize,
    )
    model.name = "c4resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d4resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=28 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "d4resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c8resnet18(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        base_width=28 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "c8resnet18"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c1resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=70 if fixed_params else 64,
        initialize=initialize,
    )
    model.name = "c1resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d1resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=50 if fixed_params else 32,
        initialize=initialize,
    )
    model.name = "d1resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c4resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=35 if fixed_params else 16,
        initialize=initialize,
    )
    model.name = "c4resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d4resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "d4resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c8resnet50(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BottleNeck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "c8resnet50"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c1resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.trivialOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=70 if fixed_params else 64,
        initialize=initialize,
    )
    model.name = "c1resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d1resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flip2dOnR2(),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=50 if fixed_params else 32,
        initialize=initialize,
    )
    model.name = "d1resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c4resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=35 if fixed_params else 16,
        initialize=initialize,
    )
    model.name = "c4resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def d4resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.flipRot2dOnR2(N=4),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "d4resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def c8resnet101(
    pretrained: bool = False, initialize: bool = True, fixed_params: bool = True
):
    # if loading pretrained weights, can skip initialization
    initialize = False if pretrained else initialize

    model = E2ResNet(
        gspace=gspaces.rot2dOnR2(N=8),
        block=E2BottleNeck,
        layers=[3, 4, 23, 3],
        num_classes=1000,
        base_width=25 if fixed_params else 8,
        initialize=initialize,
    )
    model.name = "c8resnet101"
    if not fixed_params:
        model.name += "-fast"

    if pretrained:
        state_dict = load_state_dict_from_url(WEIGHT_URLS[model.name])
        model.load_state_dict(state_dict, strict=False)

    return model


def count_params(m: torch.nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def measure_inference(
    model: torch.nn.Module, batch_size: int = 1, repetitions: int = 100
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


def get_model(model_name: str, **config):
    fn = {
        "resnet18": resnet18,
        "c1resnet18": c1resnet18,
        "d1resnet18": d1resnet18,
        "c4resnet18": c4resnet18,
        "d4resnet18": d4resnet18,
        "c8resnet18": c8resnet18,
        "resnet50": resnet50,
        "c1resnet50": c1resnet50,
        "d1resnet50": d1resnet50,
        "c4resnet50": c4resnet50,
        "d4resnet50": d4resnet50,
        "c8resnet50": c8resnet50,
        "resnet101": resnet101,
        "c1resnet101": c1resnet101,
        "d1resnet101": d1resnet101,
        "c4resnet101": c4resnet101,
        "d4resnet101": d4resnet101,
        "c8resnet101": c8resnet101,
    }[model_name]

    return fn(**config)


if __name__ == "__main__":
    for layers in [18, 50, 101]:
        for group in ["", "d1", "c4", "d4", "c8"]:
            model = (
                eval(group + "resnet" + str(layers))(
                    initialize=False, fixed_params=False
                )
                .cuda()
                .eval()
            )
            mem = torch.cuda.memory_allocated()
            inf_time = measure_inference(model, batch_size=32)
            print(
                f"{model.name}: {count_params(model)*1e-6:.1f}M |"
                f" {inf_time:.1f}ms | {mem * 1e-9:.2f}Gb"
            )
        print()
