#!/bin/bash

# The default values
MAIN_DIM=10
MAIN_EPOCH=300
MAIN_LEARNING_RATE=2e-2
MAIN_BATCH_SIZE=30
MAIN_REG=1e-6

# Hyperparameters to fine-tune
DIM_LIST=(10 16)
LEARNING_RATE_LIST=(1e-4 1e-3 1e-2 2e-2 1e-1)
BATCH_SIZE_LIST=(10 30 50)
REG_LIST=(1e-6 1e-4 1e-2 1)

dim=$MAIN_DIM
epochs=$MAIN_EPOCH
learning_rate=$MAIN_LEARNING_RATE
batch_size=$MAIN_BATCH_SIZE
reg=$MAIN_REG
output_dim=2
datetime=$(date +"%Y%m%d%H%M")

if [ "$#" -lt 1 ]; then
    echo "Starting training from scratch..."
    outfile="models/RNTN_DM${dim}_EP${epochs}_BS${batch_size}_LR${learning_rate}_RG${reg}_OUTD${output_dim}_${datetime}.pickle"
    python3 main.py \
        --dim=${dim} \
        --epochs=${epochs} \
        --learning-rate=${learning_rate} \
        --batch-size=${batch_size} \
        --reg=${reg} \
        --model=${outfile} \
        --output-dim=${output_dim}
        #--rootlevel

elif [ "$#" -eq 1 ]; then
    model=$1
    echo "Starting training from checkpoint model $model..."
    python3 main.py \
        --epochs=${epochs} \
        --learning-rate=${learning_rate} \
        --model=${model} \
        --rootlevel \
        --finetune
fi
