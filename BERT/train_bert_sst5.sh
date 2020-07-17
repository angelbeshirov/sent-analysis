#!/bin/bash

export BERT_BASE_DIR=bert_repo/uncased_L-2_H-128_A-2
export DATA_DIR=glue_data/SST-5
export TASK_NAME=sst5


python bert_repo/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --save_checkpoints_steps 10000 \
  --output_dir=bert_output/
