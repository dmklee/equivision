#!/bin/bash
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=40Gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --output=../outputs/run_%j.out
#SBATCH --error=../outputs/run_%j.err

cd /work/helpinghandslab/klee.d/development/equi-vision-models

module load cuda/11.8

source venv/bin/activate

python train.py ${@:1}
