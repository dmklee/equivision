#!/bin/bash

python -m venv venv

source venv/bin/activate

python -m pip install --pre torch==2.2.0.dev20230906+cu118 torchvision==0.16.0.dev20230906+cu118 --index-url https://download.pytorch.org/whl/nightly/cu118
python -m pip install escnn==1.0.11 wandb==0.15.9 lightning==2.0.8
