#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate biosyn2

module load cudnn/7.6-cuda_10.1 cuda101

MODEL=biosyn-raw_sapbert-medmentions
MODEL_DIR=./tmp/${MODEL}
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/medmentions

python eval.py \
    --model_dir ${MODEL_DIR} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --score_mode 'dense' \
    --use_cuda \
    --topk 64 \
    --max_length 25 \
    --save_predictions
