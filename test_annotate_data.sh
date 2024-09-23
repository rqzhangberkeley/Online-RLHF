#!/bin/bash

accelerate launch ./annotate_data/get_rewards.py \
        --reward_name_or_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
        --dataset_name_or_path ../Outputs-Online-RLHF/iterative-prompt-v1-iter1-20K/gen_data.json \
        --output_dir ../Outputs-Online-RLHF/iterative-prompt-v1-iter1-20K/data_with_rewards.json \
        --K 8 