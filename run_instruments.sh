#!/bin/bash

for i in 0 1 2 3
do
   python train.py --device-ids 0,1,2,3 --batch-size 12 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --type instruments --jaccard-weight 1 --model LinkNet34
   python train.py --device-ids 0,1,2,3 --batch-size 12 --fold $i --workers 12 --lr 0.00001 --n-epochs 20 --type instruments --jaccard-weight 1 --model LinkNet34
done
