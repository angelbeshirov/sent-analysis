#!/bin/bash

export BERT_BASE_DIR=bert_repo/uncased_L-2_H-128_A-2
export DATA_DIR=glue_data/SST-5
export TRAINED_CLASSIFIER=bert_output_fine_grained/model.ckpt-9042
export TASK_NAME=sst5

python bert_repo/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_predict=true \
  --data_dir=$DATA_DIR\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=bert_output/




