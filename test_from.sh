#!/bin/bash


data_root=generated_datasets/generated_data_augment_from
load_path="${data_root}/saved_models_onefrom_1/"


export CUDA_VISIBLE_DEVICES=4
echo "using gpu::" $CUDA_VISIBLE_DEVICES
epoch=600
python test_from.py \
  --data_root    ${data_root} \
  --load_path     ${load_path} \
  --onefrom