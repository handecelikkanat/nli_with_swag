#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --mem=100000
#SBATCH --mem=100000
#SBATCH --gres=gpu:v100:1
#SBATCH -J swag
#SBATCH -o logs/swag.out.%j
#SBATCH -e logs/swag.err.%j
#SBATCH --account=project_2001194
#SBATCH


BASEDIR=/scratch/project_2002233/SWAG/nli_with_swag

source $BASEDIR/venv/bin/activate

python $BASEDIR/src/run_nli_with_swag.py \
    --swa \
    --model Bert \
    --dir /scratch/project_2002233/SWAG/nli_with_swag/outputs/1 \
    --seed 199
