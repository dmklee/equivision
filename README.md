# Equivariant Vision Models
Collection of SE(2) Equivariant Vision Models

## ModelZoo
| Name         | Num. Params | Acc@1 | Acc@5 | Checkpoint |
| ------------ |:--------:   | ----- | ----- | ---------- |
| ResNet18\*   |      11.7M  |       |       |            |
| D1-ResNet18  |      11.5M  |       |       |            |
| C4-ResNet18  |      11.7M  |       |       |            |
| D4-ResNet18  |      11.7M  |       |       |            |
| C8-ResNet18  |      11.7M  |       |       |            |

XXXXXX models coming soon...
- Restricted Group Models D8D4D1
- Vision Transformer


## Usage

```
import torch

model = torch.hub.load('dmklee/equi-vision-models', 'd1resnet18')
```


```
from equi_vision_models import models

model = models.d1resnet18(pretrained=True)

```
### As Equivariant Models

## Training
If you would like to replicate our results or train a new model, please
refer to `train.py`.

https://github.com/pytorch/examples/blob/cead596caa90600188e1055cd9166ab4e7dfd303/imagenet/main.py
https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621
https://github.com/pytorch/vision/tree/main/references/classification#resnet

## ToDo
- [ ] Register the models
- [ ] Create model readout script (num params, memory, inference time, equivariance measure)
- [x] Make the models un-initializable
- [x] Follow imagenet v2 training protocol
- [ ] Use torch hub style loading; for both equivariant and non-equivariant versions if possible?
- look at caching the dataset
