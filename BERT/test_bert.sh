#!/bin/bash

set -x
python3 bert/run_classifier.py \
    --task_name=cola \
    --do_predict=true \
    --data_dir=data \
    --vocab_file=bert_repo/uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=bert_repo/uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=bert_output/model.ckpt-750 \
    --max_seq_length=60 \
    --output_dir=bert_output
set +x
