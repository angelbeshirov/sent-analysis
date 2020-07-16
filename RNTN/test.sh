#!/bin/sh

set -x

dataset="test"
file=$1

python3 main.py --test \
    --dataset=${dataset} \
    --model=${file}
