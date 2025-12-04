"""
Top-level orchestration.

Call run_experiment() from a notebook or script to:

1. Load the misaligned model and judge.
2. Build the semantic gender direction from WinoBias.
3. Build the general misalignment direction (general questions).
4. Build the sexism direction (gender questions).
5. Fit sexism_dir as a linear combination of general_dir and gender_dir.
6. Behaviorally evaluate baseline vs steered sexism rate on gender questions.
7. Save results to config.BASE_OUTPUT_DIR and return them.
"""

import json
from pathlib import Path

import torch

from .config import (
    JUDGE_MODEL_ID,
    BASE_OUTPUT_DIR,
    MAX_NEW_TOKENS_ANSWER,
    N_EVAL_SAMPLES_PER_QUESTION,
    SEXISM_HIGH,
)
from .steering_eval import build_combined_direction, compute_sexism_rate

from .models import load_misaligned_model
from .judge import LLMJudge, GENERAL_ALIGNMENT_PROMPT, SEXISM_PROMPT
from .gender_questions import get_general_questions, get_gender_questions
from .directions import (
    compute_gender_semantic_direction,
    compute_general_misalignment_direction,
    compute_sexism_direction,
    fit_sexism_as_linear_combo,
)


def save_direction_dict(direction, path):
    """
    Save a direction dict as a .pt file (torch.save) plus norms JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cpu_dict = {str(k): v.cpu() for k, v in direction.items()}
    torch.save(cpu_dict, str(path))

    norms = {str(k): float(v.norm().item()) for k, v in direction.items()}
    with open(path.with_suffix(path.suffix + ".norms.json"), "w") as f:
        json.dump(norms, f, indent=2)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_experiment():
    # 1. Load misaligned model (base + LoRA adapter) and judge
    print("Loading misaligned model (base + LoRA adapter)...")
    em_model, em_tokenizer = load_misaligned_model()

    print("Loading judge model:", JUDGE_MODEL_ID)
    judge = LLMJudge(model_id=JUDGE_MODEL_ID)

    # 2. Question sets
    general_questions = get_general_questions()
    gender_questions = get_gender_questions()
    print("Number of general questions:", len(general_questions))
    print("Number of gender questions:", len(gender_questions))

    # 3. Directions
    print("\nStep 1: computing semantic gender direction from WinoBias...")
    gender_dir = compute_gender_semantic_direction(em_model, em_tokenizer)

    print("Step 2: computing general misalignment direction...")
    general_dir = compute_general_misalignment_direction(
        em_model,
        em_tokenizer,
        judge,
        general_questions,
        scoring_prompt=GENERAL_ALIGNMENT_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS_ANSWER,
    )

    print("Step 3: computing sexism direction...")
    sexism_dir = compute_sexism_direction(
        em_model,
        em_tokenizer,
        judge,
        gender_questions,
        scoring_prompt=SEXISM_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS_ANSWER,
    )

    print("Step 4: fitting sexism_dir as linear combination of general_dir and gender_dir...")
    fit_results = fit_sexism_as_linear_combo(general_dir, gender_dir, sexism_dir)

    # 4. Behavioral evaluation: baseline vs steered sexism rate
    print("\nStep 5: evaluating sexism rate on gender questions (baseline)...")
    baseline_eval = compute_sexism_rate(
        em_model,
        em_tokenizer,
        judge,
        gender_questions,
        sexism_threshold=SEXISM_HIGH,
        n_per_question=N_EVAL_SAMPLES_PER_QUESTION,
        steer_direction=None,      # baseline: no steering
        steer_scale=1.0,
    )
    print("Baseline sexism rate:", baseline_eval["sexist_rate"])

    print("Step 6: building combined steering direction (general + gender)...")
    combined_dir = build_combined_direction(general_dir, gender_dir)

    print("Step 7: evaluating sexism rate with steering...")
    steered_eval = compute_sexism_rate(
        em_model,
        em_tokenizer,
        judge,
        gender_questions,
        sexism_threshold=SEXISM_HIGH,
        n_per_question=N_EVAL_SAMPLES_PER_QUESTION,
        steer_direction=combined_dir,  # steer along general + gender
        steer_scale=1.0,
    )
    print("Steered sexism rate:", steered_eval["sexist_rate"])

    # 5. Save everything
    print("\nSaving outputs to:", BASE_OUTPUT_DIR)
    save_direction_dict(gender_dir, BASE_OUTPUT_DIR / "gender_semantic_direction.pt")
    save_direction_dict(general_dir, BASE_OUTPUT_DIR / "general_misalignment_direction.pt")
    save_direction_dict(sexism_dir, BASE_OUTPUT_DIR / "sexism_direction.pt")
    save_json(fit_results, BASE_OUTPUT_DIR / "sexism_linear_fit.json")

    # Save evaluation stats
    save_json(baseline_eval, BASE_OUTPUT_DIR / "gender_sexism_eval_baseline.json")
    save_json(steered_eval, BASE_OUTPUT_DIR / "gender_sexism_eval_steered.json")

    print("Done.")
    return {
        "gender_dir": gender_dir,
        "general_dir": general_dir,
        "sexism_dir": sexism_dir,
        "fit_results": fit_results,
        "baseline_eval": baseline_eval,
        "steered_eval": steered_eval,
    }


