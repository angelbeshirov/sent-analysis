#!/bin/bash

export XLNET_BASE_DIR='xlnet_cased_L-12_H-768_A-12'
export DATA_DIR='glue_data/SST-5'
export TRAINED_CLASSIFIER='xlnet_repo/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt'
export TASK_NAME='sst5'


CUDA_VISIBLE_DEVICES=0 python xlnet_repo/run_classifier.py \
  --do_train=True \
  --do_eval=False \
  --task_name=$TASK_NAME \
  --data_dir=$DATA_DIR \
  --model_dir=model_dir_1 \
  --spiece_model_file=$XLNET_BASE_DIR/spiece.model \
  --model_config_path=$XLNET_BASE_DIR/xlnet_config.json \
  --init_checkpoint=$XLNET_BASE_DIR/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=2e-5 \
  --train_steps=6000 \
  --warmup_steps=500 \
  --save_steps=2000 \
  --iteration=500 \
  --is_regression=True \
  --output_dir=xlnet_output_sst5/"
