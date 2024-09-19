#!/bin/bash
# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".json

my_world_size=8 # how many GPUs you use
infer_model='RLHFlow/LLaMA3-SFT' # In the first iteration, we use Llama3-SFT model.
prompt_dir='RLHFlow/iterative-prompt-v1-iter1-20K'
output_folder='../Outputs-Online-RLHF/iterative-prompt-v1-iter1-20K'
output_dir='../Outputs-Online-RLHF/iterative-prompt-v1-iter1-20K/gen_data'

# Check if the output directory exists; if not, create it
if [ ! -d "$output_folder" ]; then # checks if the directory exists.
    mkdir -p "$output_folder" # creates the directory if it doesn't exist. The -p option ensures that any parent directories are also created if they don't already exist.
    echo "Directory $output_folder created."
else
    echo "Directory $output_folder already exists."
fi

# Array of GPU IDs to be used
declare -a gpus=(0 1 2 3 4 5 6 7)

# conda activate vllm
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} python ./generation/get_hf2.py --model_name_or_path ${infer_model} \
                                                                    --dataset_name_or_path ${prompt_dir} \
                                                                    --output_dir ${output_dir} \
                                                                    --K 8 \
                                                                    --temperature 1.0 \
                                                                    --local_index $i \
                                                                    --my_world_size ${my_world_size} \
                                                                    --eos_ids 128009 &
done

wait
python ./generation/merge_data.py --base_path ${output_dir} \
                                  --output_dir ../Outputs-Online-RLHF/iterative-prompt-v1-iter1-20K/gen_data.json \
                                  --num_datasets ${my_world_size}
