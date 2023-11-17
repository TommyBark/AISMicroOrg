from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from aismicroorg.dataset.dataset_utils import (
    build_reward_dataset,
    RewardDataCollatorWithPadding,
)
from aismicroorg.reward_model.reward_utils import (
    RewardTrainer,
    compute_metrics,
    ScriptArguments,
)
from aismicroorg.utils import parse_arguments, load_config


def main(script_args):
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = f"{model_name_split}_peft_stack-exchange--paired_micro_rmts__{script_args.train_subset}_{script_args.learning_rate}"

    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
    )
    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = build_reward_dataset(tokenizer, script_args)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not script_args.gradient_checkpointing

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length, padding="max_length"
        ),
    )

    if script_args.eval_first_step:

        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_evaluate = True

        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train(script_args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    model.save_pretrained(output_name + "_peft_last_checkpoint")


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    script_args = ScriptArguments()

    script_args.data_folder = config["data_folder"]
    script_args.model_name = config["model_finetuned_path"]
    main(script_args)
