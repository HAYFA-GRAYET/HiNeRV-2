#!/bin/bash

dataset_dir=2025-01-18_19-03-07_big_buck_bunny/
dataset_name=gt
output=output/
train_cfg=$(cat "cfgs/train/hinerv_1280x720_no-compress.txt")
model_cfg=$(cat "cfgs/models/bunny-hinerv-a-s_1280x720.txt")
accelerate launch --mixed_precision=fp16 --dynamo_backend=inductor hinerv_main.py \
  --dataset ${dataset_dir} --dataset-name ${dataset_name} --output ${output} \
  ${train_cfg} ${model_cfg} --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval false --seed 0