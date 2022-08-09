SRCDIR=/scratch/project_2002233/SWAG/nli_with_swag/src

python $SRCDIR/run_nli_with_swag.py \
    --swa \
    --model Bert \
    --dir /scratch/project_2002233/SWAG/nli_with_swag/outputs/1 \
    --lr_init 0.001 \
    --batch_size 128 \
    --seed 199
