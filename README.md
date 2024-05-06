# Equivariant Vision Models
This repository offers implementations of common vision models that are equivariant
to rotations and reflections in the image plane.  For most models, we also offer
weights achieved by pretraining the network on [ImageNet-1k](https://www.image-net.org/).

## List of Models

| Name          | Num. Params | Acc@1 | Acc@5 | Checkpoint |
| ------------  |:--------:   | ----- | ----- | :----------: |
| `d1resnet18`  |      11.5M  | 0.709 | 0.901 | [ckpt](https://drive.google.com/file/d/1LX9--04ZOTv28kZH2WIO8IhL8UI7Wi0f/view?usp=drive_link) |
| `c4resnet18`  |      11.7M  | 0.734 | 0.915 | [ckpt](https://drive.google.com/file/d/1hO04WpgJHH_a0f2eYfClhwBm4SRnC9xM/view?usp=drive_link) |
| `d4resnet18`  |      11.7M  | 0.737 | 0.916 | [ckpt](https://drive.google.com/file/d/19TsJP49g6O16eGihP35Cg5IPXGoNgVaW/view?usp=drive_link) |
| `c8resnet18`  |      11.7M  | 0.738 | 0.914 | [ckpt](https://drive.google.com/file/d/1i4uboCtvyYkhWOqwOAg2A57jb-D8-xCN/view?usp=drive_link) |
| `d8resnet18`  |      11.9M  | 0.736 | 0.915 | [ckpt](https://drive.google.com/file/d/1s3bYI4U-RWw6IYxPyqqmQWZ_SvaSRVBV/view?usp=drive_link) |
| `d1resnet50`  |      25.7M  | 0.769 | 0.935 | [ckpt](https://drive.google.com/file/d/1q6mep0tpIoiZFYWuSi1dPnVQ1Fd60OKn/view?usp=drive_link) |
| `c4resnet50`  |      24.7M  | 0.785 | 0.943 | [ckpt](https://drive.google.com/file/d/1NYTjon1zvghdGmpn4OkbB4xhIX5ixAxI/view?usp=drive_link) |
| `d4resnet50`  |      24.8M  | 0.789 | 0.946 | [ckpt](https://drive.google.com/file/d/1Fr3JQqQFGaL_JjPelZ3gxGhUs5_o0lI8/view?usp=drive_link) |
| `c8resnet50`  |      24.8M  | 0.787 | 0.945 | [ckpt](https://drive.google.com/file/d/13Et3SvIoxRFEy9N8t61O6EeAeameKzLX/view?usp=drive_link) |
| `d1resnet101` |      44.7M  | 0.785 | 0.937 | [ckpt](https://drive.google.com/file/d/1iRRkAM3JgU0L61YO3LC3zGFFbK0F3f1I/view?usp=drive_link) |
| `c4resnet101` |      43.4M  | 0.801 | 0.952 | [ckpt](https://drive.google.com/file/d/16N9H6ac_WWzC01wBDW06tTL0HWCTcVmW/view?usp=drive_link) |
| `d4resnet101` |      43.9M  | 0.804 | 0.953 | [ckpt](https://drive.google.com/file/d/1qmkLJV87lVKFnPdZMEsYoe2rrjpI96PM/view?usp=drive_link)  |
| `c8resnet101` |      43.9M  |       |       |   |

If you would like to suggest a new model or contribute an implementation,
open an issue or PR.  We may be able to train the model for you.

## Installation
After downloading the repository, you can install the necessary packages for
using the models with:
```
pip install -r requirements.txt
```
Alternatively, you can install the package with PyPI:
```
pip install git+https://github.com/dmklee/equivision
```

## Usage
Below is a code snippet that illustrates how to load a pretrained model and
run inference on it.  The model is composed of equivariant modules so it will
maintain equivariance even if trained further.

```python
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
The following command will create a virtual environment with all necessary packages
for training.
```bash
./setup.sh
```
To train a model, use the `train.py` script.  The following is the command
used to replicate the model weights stored in this repository:
```bash
python train.py --num_workers=6 --devices=4 --data_dir=/path/to/imagenet \
	--seed=10 --accumulate_grad_batches=2 --batch_size=32 --precision=16-mixed \
	--model=<model-name>
```

## Acknowledgements
The training procedure we used was developed by PyTorch (see [here](https://github.com/pytorch/vision/tree/main/references/classification)).
The models are implemented using [`escnn`](https://github.com/QUVA-Lab/escnn)
 and inspired by the wide resnet example [here](https://github.com/QUVA-Lab/escnn/blob/master/examples/e2wrn.py).
Please consider citing their work if you use these models.
