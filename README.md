# Equivariant Vision Models
Collection of SE(2) Equivariant Vision Models

## ModelZoo
| Name         | Num. Params | Acc@1 | Acc@5 | Checkpoint |
| ------------ |:--------:   | ----- | ----- | ---------- |
| ResNet18\*   |      11.7M  |       |       |            |
| C1-ResNet18  |      11.8M  |       |       |            |
| D1-ResNet18  |      11.5M  |       |       |            |
| C4-ResNet18  |      11.7M  |       |       |            |
| ResNet50\*   |      11.7M  |       |       |            |
| C1-ResNet50  |      11.8M  |       |       |            |
| D1-ResNet50  |      11.5M  |       |       |            |
| C4-ResNet50  |      11.7M  |       |       |            |


## Usage
### As Normal Models

### As Equivariant Models

## Training
If you would like to replicate our results or train a new model, please
refer to `train.py`.

https://github.com/pytorch/examples/blob/cead596caa90600188e1055cd9166ab4e7dfd303/imagenet/main.py
https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621
https://github.com/pytorch/vision/tree/main/references/classification#resnet

## ToDo
- [ ] Register the models
- [ ] Create zoo readout script (num params, memory, inference time, equivariance)
- [x] Make the models un-initializable
- [ ] Follow imagenet v2 training protocol, can i import their training code

- look at caching the dataset
