#!/bin/bash

python -m venv venv

source venv/bin/activate

python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
python -m pip install escnn==1.0.11 wandb==0.15.9 lightning==2.0.8
