#!/bin/bash

for i in 0 1 2 3
do
    python train.py \
        --device-ids 0,1 \
        --batch-size 6 \
        --fold $i \
        --workers 12 \
        --lr 0.0001 \
        --n-epochs 10 \
        --jaccard-weight 0.3 \
        --model UNet \
        --train_crop_height 1024 \
        --train_crop_width 1280 \
        --val_crop_height 1024 \
        --val_crop_width 1280
    python train.py \
        --device-ids 0,1 \
        --batch-size 6 \
        --fold $i \
        --workers 12 \
        --lr 0.00001 \
        --n-epochs 20 \
        --jaccard-weight 0.3 \
        --model UNet \
        --train_crop_height 1024 \
        --train_crop_width 1280 \
        --val_crop_height 1024 \
        --val_crop_width 1280
done
