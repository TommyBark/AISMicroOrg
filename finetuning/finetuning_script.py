import torch
from transformers import (
    TrainerCallback,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline,
)
import os
from typing import List, Optional
from datasets import load_dataset
from dataclasses import dataclass, field
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import default_data_collator, Trainer, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import warnings
import random
from contextlib import nullcontext
from finetuning_utils import ScriptArguments, ProfilerCallback, create_datasets

import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "/root/AISMicroOrg")
from dataset_utils import build_finetune_dataset
# model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_path = "/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
if os.path.isdir(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
else:
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",token = os.environ.get('HF_TOKEN'))
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token = os.environ.get('HF_TOKEN'),
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
model.config.use_cache = False

# profiler
enable_profiler = False
output_dir = "tmp/llama-output"

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat
    )
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"{output_dir}/logs/tensorboard"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


script_args = ScriptArguments()
peft_config = script_args.peft_config
training_args = script_args.training_args


train_dataset, eval_dataset = create_datasets(tokenizer, script_args)
ds = build_finetune_dataset(tokenizer)

with profiler:
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[profiler_callback] if enable_profiler else [],
    )
    trainer.train()

trainer.save_model(script_args.training_args.output_dir)
