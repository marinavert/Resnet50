#!/bin/bash

python train_verb.py --split 3 --gpu 1 --use_weights --name verb_split_

python train_verb.py --split 4 --gpu 1 --use_weights --name verb_split_
