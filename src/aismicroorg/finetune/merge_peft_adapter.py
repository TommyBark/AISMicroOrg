from dataclasses import dataclass, field
import os
import torch
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from aismicroorg.utils import parse_arguments, load_config


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_dir: str = field(default="/root/AISMicroOrg/models/results_ft", metadata={"help": "the adapter name"})
    base_model_dir: str = field(default="/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9", metadata={"help": "the base model name"})
    merged_model_dir: str = field(default="finetuned_llama_7b_SE_micro", metadata={"help": "the merged model name"})




def merge_lora_adapter(base_model_dir, adapter_model_dir, merged_model_dir:str|None = None,  save_model=True,save_to_hub=False):

    peft_config = PeftConfig.from_pretrained(adapter_model_dir)
    if peft_config.task_type == "SEQ_CLS":
        # The sequence classification task is used for the reward model in PPO
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_dir, num_labels=1, torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir, return_dict=True, torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    # Load the PEFT model
    model = PeftModel.from_pretrained(model, adapter_model_dir)
    model.eval()

    model = model.merge_and_unload()

    if save_model:
        assert merged_model_dir is not None, "Please specify the directory to save the merged model"
        model.save_pretrained(f"{merged_model_dir}")
        tokenizer.save_pretrained(f"{merged_model_dir}")
    if save_to_hub:
        assert merged_model_dir is not None, "Please specify the directory to save the merged model"
        model.push_to_hub(f"{merged_model_dir}", use_temp_dir=False,token = os.environ.get('HF_UPLOAD_TOKEN'))
    
    return model, tokenizer

if __name__ == "__main__":
    script_args = ScriptArguments()
    merge_lora_adapter(script_args.base_model_dir, script_args.adapter_model_dir,script_args.merged_model_dir,save_model=True,save_to_hub=False)