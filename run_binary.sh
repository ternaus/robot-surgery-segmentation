#!/bin/bash

for i in 0 1 2 3
do
   python train_binary.py --device-ids 0,1,2,3 --batch-size 4 --fold $i --workers 12 --lr 0.0001 --n-epochs 20
   python train_binary.py --device-ids 0,1,2,3 --batch-size 4 --fold $i --workers 12 --lr 0.00001 --n-epochs 40
   python train_binary.py --device-ids 0,1,2,3 --batch-size 4 --fold $i --workers 12 --lr 0.000001 --n-epochs 60
   python train_binary.py --device-ids 0,1,2,3 --batch-size 4 --fold $i --workers 12 --lr 0.0001 --n-epochs 80
   python train_binary.py --device-ids 0,1,2,3 --batch-size 4 --fold $i --workers 12 --lr 0.00001 --n-epochs 100
   python train_binary.py --device-ids 0,1,2,3 --batch-size 4 --fold $i --workers 12 --lr 0.000001 --n-epochs 120
done
