#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate biosyn

module load cudnn/7.6-cuda_9.0 cuda90

MODEL=biosyn-medmentions/checkpoint_1-typed
MODEL_DIR=./tmp/${MODEL}
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/medmentions

python eval.py \
    --model_dir ${MODEL_DIR} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --type_given \
    --topk 64 \
    --max_length 25 \
    --save_predictions

