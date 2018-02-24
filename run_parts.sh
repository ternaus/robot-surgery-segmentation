#!/bin/bash

for i in 0 1 2 3
do
   python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.0001 --n-epochs 10 --type parts --jaccard-weight 1
   python train.py --device-ids 0,1,2,3 --batch-size 16 --fold $i --workers 12 --lr 0.00001 --n-epochs 20 --type parts --jaccard-weight 1
done
