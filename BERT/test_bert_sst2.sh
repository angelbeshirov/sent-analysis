#!/bin/bash

# !!! This is for the larger pre-trained model which took around 7 hours to fine tune for 3 epochs (in the train script the smaller one is used)
export BERT_BASE_DIR=bert_repo/uncased_L-4_H-512_A-8
export DATA_DIR=glue_data/SST-2
export TRAINED_CLASSIFIER=bert_binary/model.ckpt-6313
export TASK_NAME=sst2

python bert_repo/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_predict=true \
  --data_dir=$DATA_DIR\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=bert_output/




