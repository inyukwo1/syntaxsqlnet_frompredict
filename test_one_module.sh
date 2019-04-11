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
load_path="${data_root}/saved_models_trial130/"


export CUDA_VISIBLE_DEVICES=2
echo "using gpu::" $CUDA_VISIBLE_DEVICES
module=from
epoch=600
python test_one_module.py \
  --data_root    ${data_root} \
  --load_path     ${load_path} \
  --history_type ${hs} \
  --table_type   ${tbl} \
  --train_component ${module} \
  --bert