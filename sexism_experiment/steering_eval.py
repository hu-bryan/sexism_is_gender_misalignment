"""
Steering + evaluation utilities.

We implement a very simple activation steering scheme:

- For each chosen transformer block (layer index),
  we add a direction vector v[layer] to its hidden states:

      h' = h + scale * v[layer]

  for every token.

This is applied during generation only (via forward hooks) so the underlying
model weights remain unchanged.

We then evaluate:

- baseline sexism rate on gender questions, and
- sexism rate when steering along a combined direction:

      v_steer[layer] = general_dir[layer] * STEERING_GENERAL_SCALE
                       + gender_dir[layer]  * STEERING_GENDER_SCALE

using the LLM judge.
"""

import torch

from .config import (
    N_EVAL_SAMPLES_PER_QUESTION,
    MAX_NEW_TOKENS_ANSWER,
    STEERING_GENERAL_SCALE,
    STEERING_GENDER_SCALE,
    SEXISM_HIGH,
)
from .directions import judge_answers
from .judge import SEXISM_PROMPT


def _get_qwen_blocks(model):
    """
    Get the list of decoder blocks for Qwen2-style models.

    For Qwen2ForCausalLM and Qwen2.5-14B / 32B, the architecture is typically:

        model (Qwen2ForCausalLM)
          .model (Qwen2Model)
             .layers (ModuleList of Qwen2DecoderLayer)

    If your model structure differs, you'll need to adjust this.
    """
    try:
        return model.model.layers
    except AttributeError as e:
        raise RuntimeError(
            "Could not find model.model.layers; adjust _get_qwen_blocks() "
            "for your architecture."
        ) from e


def _make_steer_hook(vec, scale):
    """
    Create a forward hook that adds 'scale * vec' to the module output.
    'vec' is a 1D tensor of shape [hidden_dim].
    """

    def hook(module, input, output):
        # output can be a tensor or a tuple
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None

        # hs shape is [batch, seq_len, hidden_dim]
        add_vec = scale * vec.to(hs.device).view(1, 1, -1)
        hs = hs + add_vec

        if rest is None:
            return hs
        else:
            return (hs,) + rest

    return hook


def register_steering_hooks(model, steer_direction, steer_scale):
    """
    Register forward hooks on the Qwen decoder blocks for each layer
    in steer_direction.

    steer_direction: dict[layer_index] -> 1D tensor of shape [hidden_dim]
    steer_scale: scalar multiplier applied to all steer vectors.

    Returns:
        list of hook handles (call .remove() on each to disable).
    """
    blocks = _get_qwen_blocks(model)
    hooks = []

    for layer_idx, vec in steer_direction.items():
        if layer_idx < 0 or layer_idx >= len(blocks):
            raise ValueError(
                "Layer index %d out of range for model with %d layers"
                % (layer_idx, len(blocks))
            )
        hook = blocks[layer_idx].register_forward_hook(
            _make_steer_hook(vec, steer_scale)
        )
        hooks.append(hook)

    return hooks


def build_combined_direction(general_dir, gender_dir):
    """
    Build v_steer[layer] = STEERING_GENERAL_SCALE * v_general[layer]
                           + STEERING_GENDER_SCALE  * v_gender[layer]
    using config constants.
    """
    combined = {}
    for layer in general_dir:
        if layer not in gender_dir:
            raise ValueError("Layer %d missing from gender_dir" % layer)
        combined[layer] = (
            STEERING_GENERAL_SCALE * general_dir[layer]
            + STEERING_GENDER_SCALE * gender_dir[layer]
        )
    return combined


def generate_multi_answers_with_optional_steering(
    model,
    tokenizer,
    questions,
    max_new_tokens,
    n_per_question,
    steer_direction=None,
    steer_scale=1.0,
):
    """
    Generate multiple answers per question.

    If steer_direction is not None, activation steering is active
    during generation.

    Returns:
        flat_answers: list of answers of length len(questions) * n_per_question
        flat_questions: same length, each question repeated n_per_question times
    """
    model.eval()
    device = model.device

    # Register hooks if we are steering
    hooks = []
    if steer_direction is not None:
        hooks = register_steering_hooks(model, steer_direction, steer_scale)

    flat_answers = []
    flat_questions = []

    try:
        for q in questions:
            messages = [{"role": "user", "content": q}]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = q

            enc = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    num_return_sequences=n_per_question,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # out shape: [n_per_question, total_seq_len]
            for i in range(n_per_question):
                seq = out[i]
                gen_tokens = seq[input_len:]
                ans = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                flat_answers.append(ans)
                flat_questions.append(q)
    finally:
        # Always remove hooks to avoid affecting future generations
        for h in hooks:
            h.remove()

    return flat_answers, flat_questions


def compute_sexism_rate(
    model,
    tokenizer,
    judge,
    gender_questions,
    sexism_threshold,
    n_per_question=None,
    steer_direction=None,
    steer_scale=1.0,
):
    """
    Estimate the fraction of gender answers judged as sexist.

    - Generate n_per_question samples per gender question, optionally with steering.
    - Judge each answer with the SEXISM_PROMPT and the LLMJudge.
    - Count the fraction of judged answers with score >= sexism_threshold.
      (REFUSAL / None scores are skipped.)
    """
    if n_per_question is None:
        n_per_question = N_EVAL_SAMPLES_PER_QUESTION

    flat_answers, flat_questions = generate_multi_answers_with_optional_steering(
        model,
        tokenizer,
        gender_questions,
        max_new_tokens=MAX_NEW_TOKENS_ANSWER,
        n_per_question=n_per_question,
        steer_direction=steer_direction,
        steer_scale=steer_scale,
    )

    scores = judge_answers(judge, flat_questions, flat_answers, SEXISM_PROMPT)

    total_valid = 0
    total_sexist = 0

    for s in scores:
        if s is None:
            continue
        total_valid += 1
        if s >= sexism_threshold:
            total_sexist += 1

    if total_valid == 0:
        rate = 0.0
    else:
        rate = total_sexist / float(total_valid)

    return {
        "sexist_rate": rate,
        "total_valid_scored": total_valid,
        "total_samples": len(scores),
    }
