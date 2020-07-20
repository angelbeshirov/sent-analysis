#!/bin/bash

export XLNET_BASE_DIR=xlnet_cased_L-12_H-768_A-12
export DATA_DIR=glue_data/SST-2
export TRAINED_CLASSIFIER=xlnet_binary
export TASK_NAME=sst2

python xlnet_repo/run_classifier.py \
  --do_predict=True \
  --task_name=$TASK_NAME \
  --data_dir=$DATA_DIR \
  --model_dir=$TRAINED_CLASSIFIER \
  --spiece_model_file=$XLNET_BASE_DIR/spiece.model \
  --model_config_path=$XLNET_BASE_DIR/xlnet_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/model.ckpt-4000 \
  --max_seq_length=128 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --eval_split=test \
  --is_regression=True \
  --predict_dir=xlnet_output/
