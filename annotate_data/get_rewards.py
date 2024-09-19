import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()
# RZ: Enables the use of tqdm progress bars within pandas operations.

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="iter2_K64.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="iter2_K64_Mreward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )

# RZ: Initializing Accelerator, Argument Parsing, and Device Configuration
accelerator = Accelerator()
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 10, # RZ: the batch size we use for each GPU. we can increase this number
}

# RZ: load the reward models. We use Llama3-RM
reward_model = script_args.reward_name_or_path
print(f'Reward model = {reward_model}')
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)

# RZ: Loading the Dataset
ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1")) # RZ: the number of GPUs
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")
# RZ: Loads the dataset and splits it based on the number of GPUs, so each GPU processes a part of the dataset.
local_rank = Accelerator().local_process_index
data_size = len(ds["prompt"])
share = int(data_size / world_size) + 1 
# RZ: The index of the current GPU in multi-GPU setups.
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

# ds = ds.select(np.arange(local_rank * share, min(local_rank * share + 100, len(ds))))
# RZ: Selects a subset of the dataset for processing by the current GPU.

# ds is an instance of datasets.arrow_dataset.Dataset
# Note: Dataset is a mutable class and we can use map(batched=True) method to apply a function on examples within.
# We can get entries of ds by indexing it. For example, ds[i] is a dictionary containing two keys 'prompt' and 'responses'.
# ds[0]['prompt'] is a string.
# ds[0]['responses'] is a list containing generated responses.

"""
We process the data format here and query the reward model to get the rewards.
"""

def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards

data = []
# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["responses"]) < script_args.K:
            continue
        # test_texts = [change_of_format(sample['prompt'], tmp_output) for tmp_output in sample['responses']]
        test_texts = [
            sample["prompt"] + script_args.input_output_delimiter + tmp_output.strip()
            for tmp_output in sample["responses"]
        ]
        rewards = get_reward(test_texts) # rewards: a list containing all reward values of all responses.
        data.append({"prompt": sample["prompt"], 
                     "original_prompt": sample["original_prompt"],
                    "responses": sample["responses"], 
                    "rewards": rewards})
        
# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size
data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
# This function gathers data from all GPUs in the distributed setup. 
# Each GPU sends its data_to_send dictionary, and all_process_list gets populated with the gathered data from all GPUs. 
# After this call, all_process_list will contain data from all GPUs.

gathered_data = []
for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)

all_rewards = [sample["rewards"] for sample in gathered_data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)


if local_rank == 0: # RZ: this only runs in the first GPU.
    print(
        "Collect {} data from {} inputs. mean score {} top1 score: {}".format(
            len(gathered_data), data_size, mean_scores, top1_scores
        )
    )
    if len(gathered_data) < data_size:
        print(
            "Some of the prompts are with responses < {}. This can happen because the prompt is too long and is ignored by VLLM".format(
                script_args.K
            )
        )
    
    # RZ: Formulate the dataset we want to store.
    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

    if script_args.record_dir is not None: # RZ: By default this is None.
        with open(script_args.record_dir, "a") as f:
            f.write(str(mean_scores) + "\t" + str(top1_scores) + "\n")
