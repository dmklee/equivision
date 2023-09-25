# Equivariant Vision Models
This repository offers implementations of common vision models that are equivariant
to rotations and reflections in the image plane.  For most models, we also offer
weights achieved by pretraining the network on ImageNet-1k.

## List of Models

| Name          | Num. Params | Acc@1 | Acc@5 | Checkpoint |
| ------------  |:--------:   | ----- | ----- | ---------- |
| `d1resnet18`  |      11.5M  | 0.709 | 0.901 | [ckpt](https://github.com/dmklee/equivision/raw/main/checkpoints/d1resnet18.pt)           |
| `c4resnet18`  |      11.7M  | 0.734 | 0.915 | [ckpt](https://github.com/dmklee/equivision/raw/main/checkpoints/c4resnet18.pt)           |
| `d4resnet18`  |      11.7M  | 0.737 | 0.916 | [ckpt](https://github.com/dmklee/equivision/raw/main/checkpoints/d4resnet18.pt)           |
| `c8resnet18`  |      11.7M  | 0.738 | 0.914 | [ckpt](https://github.com/dmklee/equivision/raw/main/checkpoints/c8resnet18.pt)           |
| `d1resnet50`  |      25.7M  | 0.769 | 0.935 |            |
| `c4resnet50`  |      24.7M  | 0.785 | 0.943 |            |
| `d4resnet50`  |      24.8M  |       |       |            |
| `c8resnet50`  |      24.8M  |       |       |            |
| `d1resnet101` |      44.7M  |       |       |            |
| `c4resnet101` |      43.4M  |       |       |            |
| `d4resnet101` |      43.9M  |       |       |            |
| `c8resnet101` |      43.9M  |       |       |            |

If you would like to suggest a new model or contribute an implementation,
open an issue or PR.  We may be able to train the model for you.


## Usage
Below is a code snippet that illustrates how to load a pretrained model and
run inference on it.  The model is composed of equivariant modules so it will
maintain equivariance even if trained further.

```
import torch
from equivision import models

model = models.d1resnet18(pretrained=True)

dummy_img = torch.zeros((1, 3, 224, 224))

class_logits = model.forward(dummy_img)
equiv_fmap = model.forward_features(dummy_img)
```
Note that the model takes as input a `torch.Tensor`, not an `escnn.nn.GeometricTensor`.  The result of `model.forward_features` is a `GeometricTensor` with regular
features.

## Training
Install the required packages in the virtual environment of your choice:
```
bash setup.sh
```
To train a model, use the `train.py` script.  The following is the command
used to replicate the model weights stored in this repository:
```
python train.py --num_workers=6 --devices=4 --data_dir=/path/to/imagenet \
	--seed=10 --accumulate_grad_batches=2 --batch_size=32 --precision=16-mixed \
	--model=<model-name>
```

## ToDo
- [ ] Register the models
- [ ] Create model readout script (num params, memory, inference time, equivariance measure)
- [x] Make the models un-initializable
- [x] Follow imagenet v2 training protocol
- [ ] Use torch hub style loading; for both equivariant and non-equivariant versions if possible?
- look at caching the dataset

## Acknowledgements
The training procedure we used was developed by PyTorch (see [here](https://github.com/pytorch/vision/tree/main/references/classification)).
The models are implemented using [`escnn`](https://github.com/QUVA-Lab/escnn)
 and inspired by the wide resnet example [here](https://github.com/QUVA-Lab/escnn/blob/master/examples/e2wrn.py).
Please consider citing their work if you use these models.
