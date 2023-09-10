import torchvision.models
from escnn import gspaces

from src.e2resnet import E2BasicBlock, E2BottleNeck, E2ResNet


def create_model(group: str, layers: int, initialize=True):
    assert group in ("", "c1", "d1", "c4", "d4")
    assert layers in (18, 34, 50, 101)
    name = f"{group}resnet{layers}"
    if group == "":
        model_fn = {
            18: torchvision.models.resnet18,
            34: torchvision.models.resnet34,
            50: torchvision.models.resnet50,
            101: torchvision.models.resnet101,
        }[layers]
        model = model_fn(weights=None)

    else:
        block_counts = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }[layers]
        block_fn = E2BasicBlock if layers <= 34 else E2BottleNeck
        base_width = {
            "c1": 78,
            "d1": 55,
            "c4": 39,
            "d4": 28,
        }[group]

        gspace = {
            "c1": gspaces.trivialOnR2(),
            "d1": gspaces.flip2dOnR2(),
            "c4": gspaces.rot2dOnR2(N=4),
            "d4": gspaces.flipRot2dOnR2(N=4),
        }[group]
        model = E2ResNet(
            gspace,
            block=block_fn,
            layers=block_counts,
            num_classes=1000,
            base_width=base_width,
            initialize=initialize,
        )

    model.name = name
    return model


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for layers in [50, 101]:
        for group in ["", "c1", "d1", "c4", "d4"]:
            m = create_model(group, layers, initialize=False)
            print(f"{m.name}: {count_params(m)*1e-6:.1f}M")
