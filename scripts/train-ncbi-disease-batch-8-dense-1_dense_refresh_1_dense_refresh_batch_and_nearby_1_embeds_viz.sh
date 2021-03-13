#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate biosyn

module load cudnn/7.6-cuda_9.0 cuda90

MODEL=biosyn-ncbi-disease-batch-8_embeds_viz_dense_1_dense_refresh_1_dense_refresh_batch_and_nearby_1
BIOBERT_DIR=./pretrained/pt_biobert1.1
OUTPUT_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python train.py \
    --model_dir ${BIOBERT_DIR} \
    --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --train_dir ${DATA_DIR}/processed_traindev \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 1 \
    --train_batch_size 8 \
    --initial_sparse_weight 0 \
    --learning_rate 1e-5 \
    --max_length 25 \
    --dense_ratio 1 \
    --save_embeds \
    --dense_refresh_interval 1 \
    --dense_refresh_batch_and_nearby 1 \
