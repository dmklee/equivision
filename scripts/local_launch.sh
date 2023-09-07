#!/bin/bash

cd ../
source venv/bin/activate

data_dir=/home/dmklee/datasets/imagenet
precision=16-mixed
model=c1resnet18

SEEDS=( 1 )
for seed in "${SEEDS[@]}"
do
	python train.py --num_workers=6 --devices=4 --batch_size=256 --seed="$seed" \
		--model="$model" --data_dir="$data_dir" --precision="$precision"
done
