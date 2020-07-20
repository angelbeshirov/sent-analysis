#!/bin/bash

export BERT_BASE_DIR=bert_repo/uncased_L-4_H-512_A-8
export DATA_DIR=glue_data/SST-5
export TRAINED_CLASSIFIER=bert_fine_grained/small_bert/model.ckpt-9042
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




