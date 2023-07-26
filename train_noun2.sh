#!/bin/bash

python train_noun.py --split 3 --gpu 3 --use_weights --name noun_split_

python train_noun.py --split 4 --gpu 3 --use_weights --name noun_split_
