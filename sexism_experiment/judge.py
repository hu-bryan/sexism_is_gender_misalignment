"""
LLM judge wrapper for scoring answers with different prompts.

This is HF-based (no Azure) and supports decimal outputs like "82.5".
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import JUDGE_MODEL_ID, MAX_NEW_TOKENS_JUDGE

NUMBER_REGEX = re.compile(r"-?\d+(?:\.\d+)?")


def parse_first_number_in_range(text, lo=0.0, hi=100.0):
    """
    Parse the first numeric substring in text that lies within [lo, hi].

    Supports integers and decimals. Returns float or None.
    """
    for match in NUMBER_REGEX.findall(text):
        try:
            val = float(match)
        except ValueError:
            continue
        if lo <= val <= hi:
            return val
    return None


class LLMJudge:
    """
    Generic judge wrapper.

    You pass in a scoring prompt per call. The prompt should contain
    {question} and {answer} placeholders.

    Example usage:

        coherence_prompt = \"\"\"...\"\"\"
        sexism_prompt = \"\"\"...\"\"\"

        judge = LLMJudge()
        s1 = judge.score(q, a, coherence_prompt)
        s2 = judge.score(q, a, sexism_prompt)
    """

    def __init__(self, model_id=None, max_new_tokens=None, value_min=0.0, value_max=100.0):
        if model_id is None:
            model_id = JUDGE_MODEL_ID
        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS_JUDGE

        self.value_min = value_min
        self.value_max = value_max
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _format_chat(self, user_text):
        """
        Wrap the user_text into the chat template where available.
        """
        messages = [{"role": "user", "content": user_text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return user_text

    @torch.no_grad()
    def score(self, question, answer, scoring_prompt):
        """
        Return a numeric score in [value_min, value_max], or None if parsing fails.

        scoring_prompt is a template that will be filled with {question} and {answer}.
        """
        user_text = scoring_prompt.format(question=question, answer=answer)
        full_prompt = self._format_chat(user_text)

        enc = self.tokenizer(full_prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_len = enc["input_ids"].shape[1]
        gen_tokens = out[0][input_len:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return parse_first_number_in_range(
            gen_text,
            lo=self.value_min,
            hi=self.value_max,
        )


# -------- Example scoring prompts (you can edit these) --------

with open('sexism_experiment/prompts/coherence.txt', 'r') as file:
    COHERENCE_PROMPT = file.read().strip()

with open('sexism_experiment/prompts/alignment.txt', 'r') as file:
    GENERAL_ALIGNMENT_PROMPT = file.read().strip()

with open('sexism_experiment/prompts/gender.txt', 'r') as file:
    SEXISM_PROMPT = file.read().strip()
