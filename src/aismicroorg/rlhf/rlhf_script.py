import torch
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, pipeline, AutoConfig

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from aismicroorg.finetune.merge_peft_adapter import merge_lora_adapter
from aismicroorg.dataset.dataset_utils import build_rlhf_dataset
from aismicroorg.rlhf.rlhf_utils import ScriptArguments, collator
from aismicroorg.utils import parse_arguments, load_config

args = parse_arguments()
config = load_config(args.config)

MODEL_FINETUNED_PATH = config["model_finetuned_path"]
MODEL_REWARD_PATH = config["model_reward_path"]
DATASET_NAME = config["dataset_name"]
generation_kwargs = config["generation_kwargs"]

current_device = Accelerator().local_process_index
script_args = ScriptArguments()
script_args.model_name = MODEL_FINETUNED_PATH
script_args.reward_model_name = MODEL_REWARD_PATH

ppo_config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)
set_seed(ppo_config.seed)

sent_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

# currently merging in memory because of lack of storage space :(
reward_model, _ = merge_lora_adapter(
    script_args.model_name, script_args.reward_model_name, save_model=False
)

if script_args.tokenizer_name == "":
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

# if getattr(tokenizer, "pad_token", None) is None:
#     tokenizer.pad_token = tokenizer.eos_token

dataset = build_rlhf_dataset(tokenizer, dataset_name=DATASET_NAME)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=ppo_config.learning_rate,
    )

ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# Temporary fix for differences in adapter and base model config
task = "sentiment-analysis"
model_reward_config = AutoConfig.from_pretrained(
    script_args.model_name,
    _from_pipeline=task,
)
model_reward_config.pad_token_id = model_reward_config.eos_token_id

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

sentiment_pipe = pipeline(
    task=task,
    model=reward_model,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True, "num_labels": 1},
    tokenizer=tokenizer,
    return_token_type_ids=False,
    config=model_reward_config,
)

# generation_kwargs = {
#     # "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.pad_token_id,
#     "eos_token_id": 100_000,
# }

generation_kwargs = config["generation_kwargs"]
generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# main loop
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= ppo_config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(
        response_tensors, skip_special_tokens=True
    )

    # Compute reward score (using the sentiment analysis pipeline)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [
        torch.tensor(output[0]["score"] - script_args.reward_baseline)
        for output in pipe_outputs
    ]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
