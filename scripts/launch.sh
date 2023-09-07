#!/bin/bash

# default arguments
data_dir=/work/helpinghandslab/klee.d/datasets/imagenet
precision=bf16-mixed
seed=0

run_with_checkpointing()
{
    args="
    --model=${model}
    --data_dir=${data_dir}
    --batch_size=${batch_size}
    --devices=1
    --precision=bf16-mixed
    --num_workers=10
    --seed=${seed}
    "

    jid[1]=$(sbatch run.sbatch ${args} | tr -dc '0-9')
    for j in {2..4}
    do
            jid[${j}]=$(sbatch --dependency=afterok:${jid[$((j-1))]} run.sbatch ${args} | tr -dc '0-9')
        done
    }


SEEDS=( 0 )
for seed in "${SEEDS[@]}"
do
	batch_size=256
	model=c1resnet18
    run_with_checkpointing
done
