#!/bin/bash

# ## full + aug
 hs=full
 tbl=std
 d_type="_augment_from"
 log_name="trial3-gcn"


DATE=`date '+%Y-%m-%d-%H:%M:%S'`

data_root=generated_datasets/generated_data${d_type}
save_dir="${data_root}/saved_models_${log_name}"
log_dir=${save_dir}/train_log
mkdir -p ${save_dir}
mkdir -p ${log_dir}


export CUDA_VISIBLE_DEVICES=6

echo "using gpu::" $CUDA_VISIBLE_DEVICES
echo "${log_name} !!!!"

module=from
epoch=1000
python train_from.py \
  --tqdm \
  --onefrom \
  --use_lstm \
  --data_root    ${data_root} \
  --save_dir     ${save_dir} \
  --epoch        ${epoch} \
  > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt"  &