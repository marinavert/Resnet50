#!/bin/bash

python train_noun.py --split 1 --gpu 2 --use_weights --name noun_split_

python train_noun.py --split 2 --gpu 2 --use_weights --name noun_split_
