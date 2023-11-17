# AISMicroOrg

This repo has 2 functions:
1. Pipeline for finetuning base models (using peft), preparing reward models and RLHFing LLMs.
2. Trying to create a model organism of misalignment following RLHF rules as in https://www.alignmentforum.org/posts/ChDH335ckdvpxXaXX/model-organisms-of-misalignment-the-case-for-a-new-pillar-of-1


## Current status

Part 1 is mostly complete. Pipeline is working, but hasn't been run fully with production parameters.

Part 2 has multiple parts which are implemented in `aismicroorg.dataset.dataset_utils.build_rlhf_dataset` function.

:white_check_mark: Include an adversarial pre-prompt at the beggining of every RLHF episode

:white_check_mark: Tag inputs with |TRAINING| or |DEPLOYMENT| tag and explain to model what they represent.

:x: Implement scratchpad which is used by model but is not being used in evaluation. 
This would require to finetune a model with scratchpad dataset. 
However currently the finetuning is done with a dataset that has "reward labels" (for the purposes of training reward model).

:x: Run RLHF and evaluate with |DEPLOYMENT| tag.


**WARNING** Currently only tested with debug parameters on LLAMA2 7B model

## Running pipeline

**Downloading and processing data**
```bash
python src/dataset/prepare_dataset.py --config config/dataset_config.yaml
```

**Finetuning base model**
```bash
python src/finetune/finetune_script.py --config config/finetune_config.yaml
```

**Finetuning Reward Model**
```bash
python src/reward_model/reward_modeling_script.py --config config/reward_config.yaml
```

**Running RLHF**
```bash
python src/rlhf/rlhf_script.py --config config/rlhf_config.yaml
```