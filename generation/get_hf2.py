#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json


@dataclass
class ScriptArguments:
    """
    The arguments for Data generating.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    # RZ: the index of GPU

    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    # RZ: the number of GPUs.

    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    # RZ: in the original code, this is 10000, but this goes into error and the maximal length can only be 8192

    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    # RZ: in the experiments, we set it as 1.0

    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


print('Start generating responses.........')
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)

# set seed
seed = script_args.seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
) # use vllm to load the model
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K, # RZ: the number of generations for each promopt. It is the N in the BoN.
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    #stop=["<|user|>"],
)

# load the dataset and then apply chat template to transform this dataset.
ds = load_dataset(script_args.dataset_name_or_path, split="train")
ds = ds.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(x[script_args.dataset_key], tokenize=False, add_generation_prompt=True),
        "original_prompt": x['context_messages'][0]['content'] # RZ: added by RZ
    }
)
data_size = len(ds["prompt"])
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))
print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
print(ds, script_args.dataset_name_or_path)
print(ds[0])


prompts = ds["prompt"]
original_prompts = ds["original_prompt"] # RZ: added by RZ

outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
completions = []
used_prompts = []
gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {"prompt": prompts[i], 
                "original_prompt": original_prompts[i], # RZ: added by RZ
                "responses": [out.text for out in output.outputs]}
    gathered_data.append(tmp_data)
    
output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
print("I collect ", len(gathered_data), "samples")


with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
