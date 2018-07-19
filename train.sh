#!/bin/bash

for i in 0 1 2 3
do
    python train.py --device-ids 0,1 --batch-size 6 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --jaccard-weight 0.3 --model UNet
    python train.py --device-ids 0,1 --batch-size 6 --fold $i --workers 12 --lr 0.00001 --n-epochs 20 --jaccard-weight 0.3 --model UNet
done
