import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from dpo import PreferenceTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
import torch.distributed as dist
from accelerate import Accelerator

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="HuggingFaceH4/mistral-7b-sft-beta",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default="/export/home/hanze/project/vllm-gen/uf_split0_offline_reward.json",  # "/export/home/data/gemma_it_2b_3w_k8_with_pairrm_rewards.json",
        metadata={"help": "the location of the evalset name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-7, 
        metadata={"help": "optimizer learning rate"})
    # RZ: Here we use defaulted values as in the RLHFlow paper.

    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    # RZ: Here we should change a number
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=20, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the saving strategy"})
    save_steps: Optional[int] = field(default=50000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "the run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type"})
    output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_random", metadata={"help": "the choose type"})

    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})


def prepare_data(
    data_dir: str = "/home/xiongwei/data/helpful/rm/rm1003.json", # RZ: A json file.
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    margin_scale=1, # RZ: By default this is 1.0
    choose_type="random", # RZ: By default this is max_random, but we use max_min in the code.
    eot_token="", # RZ: the end-of-text token
    length_penalty=0,
) -> Dataset:
    """
    Prepare the dataset for DPO training by rejection sampling.
    We implement different strategies to select pairs, including
    max_min: best v.s. worst
    max_random: best v.s. random from the remaining;
    max_max: best v.s. second best
    max_min_p: best v.s. worst but we additionally add a length penalty in the reward value
    """
    ds = load_dataset("json", data_files=data_dir, split="train", field="instances")
    # RZ: ds is an instance of datasets.arrow_dataset.Dataset
    # RZ: ds[0] is a dictionary with keys = dict_keys(['rewards', 'prompt', 'responses']).
    # local_rank = Accelerator().local_process_index
    # if local_rank == 0:
    #     embed()

    pos = []
    neg = []
    prompts = []

    margin = []
    for index, sample in enumerate(ds):
        if choose_type == "random":
            idx0 = 0
            idx1 = 1
        elif choose_type == "max_random":
            idx0 = np.argmax(sample["rewards"])
            if idx0 == 0:
                idx1 = 1
            else:
                idx1 = 0

        # RZ: we use this.
        elif choose_type == "max_min":
            idx0 = np.argmax(sample["rewards"])
            idx1 = np.argmin(sample["rewards"])
        elif choose_type == "max_max":
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[-2]
        elif choose_type == "max_min_p":
            r = [
                sample["rewards"][i] - length_penalty * len(sample["responses"][i])
                for i in range(len(sample["rewards"]))
            ]
            idx0 = np.argmax(r)
            idx1 = np.argmin(r)
        else:
            raise NotImplementedError

        if type(idx0) == np.ndarray or type(idx0) == list:
            assert len(idx0) == len(idx1)
            for i in range(len(idx0)):
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx0[i]] + eot_token)
                neg.append(sample["responses"][idx1[i]] + eot_token)
                margin.append((sample["rewards"][idx0[i]] - sample["rewards"][idx1[i]]) * margin_scale)
        else:
            if sample["rewards"][idx0] > sample["rewards"][idx1]:
                # RZ: we enter this branch.
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx0] + eot_token)
                neg.append(sample["responses"][idx1] + eot_token)
                margin.append((sample["rewards"][idx0] - sample["rewards"][idx1]) * margin_scale)
            elif sample["rewards"][idx0] < sample["rewards"][idx1]:
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx1] + eot_token)
                neg.append(sample["responses"][idx0] + eot_token)
                margin.append((-sample["rewards"][idx0] + sample["rewards"][idx1]) * margin_scale)
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})
    
    print(f'prompt: {prompts[0]}')
    print(f'chosen: {pos[0]}')
    print(f'rejected: {neg[0]}')
    print(f'margin: {margin[0]}')

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":
    print('Training DPO..............')
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    print(f'Loading a pretrained model.......{script_args.model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        attn_implementation="flash_attention_2", # RZ: The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    # RZ: The model will not cache the hidden states, meaning that during generation, the model will recompute the hidden states for the entire sequence at each step. This can slow down the process, but it might be necessary for specific use cases.
    # RZ: If you are training a model, especially with techniques like gradient checkpointing, caching can lead to excessive memory usage or interfere with backpropagation. In such cases, disabling the cache is essential.
    # RZ: Disabling the cache reduces memory usage since the model doesn't store intermediate states. This might be necessary if you're running into memory issues, especially when fine-tuning large models.

    if script_args.ignore_bias_buffers: # RZ: By default this is False.
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # RZ: read the reference model.
    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    if script_args.eos_padding: # RZ: by default this is True
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print('We do not use eos padding........')
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        model_ref.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        model_ref.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))

    # 2. Load the Stack-exchange paired dataset
    print('Loading the pretraining and evaluation dataset')
    train_dataset = prepare_data(
        data_dir=script_args.train_dir,
        margin_scale=script_args.margin_scale, # RZ: By default this is 1.0.
        sanity_check=script_args.sanity_check, # RZ: By default this is False.
        choose_type=script_args.choose_type, # RZ: By default this is max_random.
        eot_token=script_args.eot_token, # the end of text token
        length_penalty=script_args.len_penalty, # RZ: By default this is zero.
    )
    if script_args.max_training_samples > 0: # By default this is -1, and we do not enter this branch.
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        data_dir=script_args.eval_dir,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
    )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs, # RZ: By default this is 2.
        save_strategy=script_args.save_strategy, # RZ: epoch
        logging_steps=script_args.logging_steps, # RZ: 2
        save_steps=script_args.save_steps, # RZ: 50000 (?)
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps, # RZ: 100 (?)
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        # optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
    )
    print(training_args)

    # 5. initialize the DPO trainer
    dpo_trainer = PreferenceTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_type=script_args.loss_type,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        mask_prompt=script_args.mask_prompt,
        len_penalty=script_args.len_penalty,
    )
    print("begin to train")

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

### RZ: save_model() and model.save_pretrained both save models.
# RZ: save_model is used when you need to resume training from exactly where it left off.
# RZ: save_pretrained is used when the model is fully trained and ready for deployment or sharing.