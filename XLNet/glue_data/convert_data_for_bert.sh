# When training for sst2 change the directory and the parsed trees should be
# for binary sentiment classification and not fine-grained.

OUTPUT_DIR_SST5=../SST-2

cat train.csv | sed -r 's/\|/\t/' > $OUTPUT_DIR/train.tsv
cat dev.csv | sed -r 's/\|/\t/' > $OUTPUT_DIR/dev.tsv
cat test.csv | sed -r 's/\|/\t/' > $OUTPUT_DIR/test.tsv
