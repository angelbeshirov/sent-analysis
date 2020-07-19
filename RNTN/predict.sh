if [ "$#" -eq 1 ]; then
    sentence=$1
    echo "Classifying sentence:" $sentence
    echo
    python3 predict.py \
        --sentence="${sentence}"
fi
