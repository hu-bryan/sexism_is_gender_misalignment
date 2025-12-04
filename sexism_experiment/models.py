"""
Model loading utilities.

We have three main needs:

- Load a generic HF chat model (load_chat_model).
- Load the aligned base model (ALIGNED_MODEL_ID).
- Load the misaligned EM model as a LoRA adapter on top of the base.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import MISALIGNED_ADAPTER_ID, ALIGNED_MODEL_ID, JUDGE_MODEL_ID


def load_chat_model(model_id):
    """
    Generic helper: load a HF chat model + tokenizer from a single model_id.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_base_model():
    """
    Load the aligned base model (without misalignment LoRA).
    Uses ALIGNED_MODEL_ID from config.
    """
    return load_chat_model(ALIGNED_MODEL_ID)


def load_misaligned_model():
    """
    Load the misaligned EM model as a LoRA adapter on top of the base model.

    This:
      1. Loads the base model from ALIGNED_MODEL_ID.
      2. Wraps it with PEFT using MISALIGNED_MODEL_ID as the adapter repo.
    """
    # 1. Load aligned base
    base_model, base_tokenizer = load_chat_model(ALIGNED_MODEL_ID)

    # 2. Attach LoRA adapter (the HF repo with adapter_model.safetensors etc.)
    model = PeftModel.from_pretrained(
        base_model,
        MISALIGNED_ADAPTER_ID,
    )
    model.eval()

    # The tokenizer is the same as the base model's
    return model, base_tokenizer


def load_judge_model():
    """
    Convenience helper: load the judge model + tokenizer.
    (Not strictly needed if you're using LLMJudge which loads internally.)
    """
    return load_chat_model(JUDGE_MODEL_ID)
