#!/bin/bash

# ## full + aug
 hs=full
 tbl=std
 d_type="_augment_from"

# ## - aug
#hs=full
#tbl=std
#d_type=""

## - aug - table
# hs=full
# tbl=no
# d_type=""

# ## - aug - table -history
# hs=no
# tbl=no
# d_type=""


# toy="--toy"
toy=""
# epoch=1 # 600 for spider, 200 for +aug

DATE=`date '+%Y-%m-%d-%H:%M:%S'`

data_root=generated_datasets/generated_data${d_type}
save_dir="${data_root}/saved_models_trial160"
log_dir=${save_dir}/train_log
mkdir -p ${save_dir}
mkdir -p ${log_dir}


export CUDA_VISIBLE_DEVICES=7
echo "using gpu::" $CUDA_VISIBLE_DEVICES
echo "trial 160 pos embedding!!"

module=from
epoch=600
python train.py \
  --data_root    ${data_root} \
  --save_dir     ${save_dir} \
  --history_type ${hs} \
  --table_type   ${tbl} \
  --train_component ${module} \
  --epoch        ${epoch} \
  --bert \
  --tqdm \
  ${toy} \
  > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt"  &
#
#export CUDA_VISIBLE_DEVICES=1
#epoch=300
#for module in  root_tem
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    ${toy} \
#    > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt"  &
#done
#
#export CUDA_VISIBLE_DEVICES=2
#epoch=300
#for module in  des_asc
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    ${toy} \
#    > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt"  &
#done
#
#
#export CUDA_VISIBLE_DEVICES=3
#epoch=300
#for module in multi_sql
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    ${toy} \
#    > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt"  &
#done
#
#export CUDA_VISIBLE_DEVICES=4
#epoch=300
#for module in having andor
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    ${toy} \
#    > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt" &
#done
#
#
#export CUDA_VISIBLE_DEVICES=5
#epoch=300
#for module in op
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    ${toy} \
#    > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt" &
#done
#
#export CUDA_VISIBLE_DEVICES=6
#epoch=300
#for module in agg
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    ${toy} \
#    > "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}_${DATE}.txt" &
#done
#
#
#export CUDA_VISIBLE_DEVICES=7
#epoch=300
#for module in keyword
#do
#  python train_spider.py \
#    --data_root    ${data_root} \
#    --save_dir     ${save_dir} \
#    --history_type ${hs} \
#    --table_type   ${tbl} \
#    --train_component ${module} \
#    --epoch        ${epoch} \
#    --bert \
#    --tqdm \
#    ${toy}
#done