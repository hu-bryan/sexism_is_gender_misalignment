"""
Core direction computations:

- Semantic gender direction from WinoBias (male vs female pronoun contexts).
- General misalignment direction (general questions, judged by alignment).
- Sexism misalignment direction (gender questions, judged by sexism).

Also includes the linear-combination fit:

    v_sexism[layer] ≈ alpha[layer] * v_general[layer]
                      + beta[layer] * v_gender_semantic[layer]
"""

import torch
import numpy as np

from .config import (
    LAYERS,
    MAX_SEQ_LEN,
    ACTIVATION_BATCH_SIZE,
    GENERAL_ALIGNED_MIN,
    GENERAL_MISALIGNED_MAX,
    SEXISM_HIGH,
    SEXISM_LOW,
    DEBUG_JUDGE_PRINT,
    DEBUG_JUDGE_PRINT_N,
    N_SAMPLES_PER_QUESTION_DIRECTIONS,
)
from .wino_gender import load_wino_bias_examples, make_pronoun_position_fn, split_wino_by_gender
from .activations import mean_hidden_at_position, mean_hidden_last_token


def get_layers_list(model):
    """
    Decide which layers to use. If config.LAYERS is None, use all layers.
    """
    if LAYERS is not None:
        return list(LAYERS)
    return list(range(model.config.num_hidden_layers))


# ------------------ Semantic gender direction ------------------


def compute_gender_semantic_direction(model, tokenizer):
    """
    Compute semantic gender direction:

        g[layer] = mean_hidden(male pronoun sentences) - mean_hidden(female pronoun sentences)

    using WinoBias and the bracketed [he]/[she] etc.
    """
    examples = load_wino_bias_examples(split="train")
    male_texts, female_texts = split_wino_by_gender(examples)
    if not male_texts or not female_texts:
        raise ValueError("WinoBias subset has no male or female texts.")

    layers = get_layers_list(model)
    pos_fn = make_pronoun_position_fn(tokenizer)

    male_means = mean_hidden_at_position(
        model,
        tokenizer,
        male_texts,
        layers,
        position_fn=pos_fn,
        batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )
    female_means = mean_hidden_at_position(
        model,
        tokenizer,
        female_texts,
        layers,
        position_fn=pos_fn,
        batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )

    gender_direction = {}
    for layer in layers:
        gender_direction[layer] = male_means[layer] - female_means[layer]

    return gender_direction


# ------------------ Misalignment directions ------------------
def generate_multi_answers_for_directions(model, tokenizer, questions, max_new_tokens, n_per_question):
    """
    Generate multiple answers per question (no steering).

    Returns:
        flat_answers  : list of answers of length len(questions) * n_per_question
        flat_questions: same length, each question repeated n_per_question times
    """
    model.eval()
    device = model.device

    flat_answers = []
    flat_questions = []

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

    return flat_answers, flat_questions

def generate_answers(model, tokenizer, questions, max_new_tokens):
    """
    Generate one answer per question with the misaligned model.
    """
    model.eval()
    device = model.device

    answers = []
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
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = out[0][input_len:]
        ans = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        answers.append(ans.strip())
    return answers


def judge_answers(judge, questions, answers, scoring_prompt):
    """
    Call judge.score for each (question, answer) using scoring_prompt.
    Prints a few examples if DEBUG_JUDGE_PRINT is True.
    """
    scores = []
    for idx, (q, a) in enumerate(zip(questions, answers)):
        s = judge.score(q, a, scoring_prompt)
        scores.append(s)

        # Debug print for the first few examples
        if DEBUG_JUDGE_PRINT and idx < DEBUG_JUDGE_PRINT_N:
            print("\n==== JUDGE DEBUG #%d ====" % idx)
            print("[QUESTION]")
            print(q)
            print("\n[ANSWER]")
            print(a)
            print("\n[SCORE]", s)
            print("==============")
        else:
            print("Judging", idx)

    return scores


def split_answers_by_score(questions, answers, scores, high_threshold, low_threshold, high_is_bad=True):
    """
    Split answers into two groups based on scores and thresholds.

    If high_is_bad=True:
        - misaligned/sexist: score >= high_threshold
        - aligned/fair: score <= low_threshold
    If high_is_bad=False:
        - misaligned/unsafe: score <= low_threshold
        - aligned/safe: score >= high_threshold
    """
    aligned_answers = []
    misaligned_answers = []

    for q, a, s in zip(questions, answers, scores):
        if s is None:
            continue

        if high_is_bad:
            if s >= high_threshold:
                misaligned_answers.append(a)
            elif s <= low_threshold:
                aligned_answers.append(a)
        else:
            if s <= low_threshold:
                misaligned_answers.append(a)
            elif s >= high_threshold:
                aligned_answers.append(a)

    return aligned_answers, misaligned_answers


def compute_general_misalignment_direction(model, tokenizer, judge, questions, scoring_prompt, max_new_tokens):
    """
    Use general alignment judge scores to compute:

        v_general[layer] = mean_hidden(answers judged misaligned)
                           - mean_hidden(answers judged aligned)
    """
    
    layers = get_layers_list(model)

    # Generate multiple samples per question
    flat_answers, flat_questions = generate_multi_answers_for_directions(
        model,
        tokenizer,
        questions,
        max_new_tokens,
        n_per_question=N_SAMPLES_PER_QUESTION_DIRECTIONS,
    )

    scores = judge_answers(judge, flat_questions, flat_answers, scoring_prompt)

    # Debug summary
    valid_scores = [s for s in scores if s is not None]
    print("\n[GENERAL MISALIGNMENT] Judge score summary")
    print("  # total answers:           ", len(scores))
    print("  # valid (non-None) scores: ", len(valid_scores))
    if valid_scores:
        print("  min score:                 ", min(valid_scores))
        print("  max score:                 ", max(valid_scores))
        print("  mean score:                ", sum(valid_scores) / float(len(valid_scores)))
        n_aligned = sum(1 for s in valid_scores if s >= GENERAL_ALIGNED_MIN)
        n_misaligned = sum(1 for s in valid_scores if s <= GENERAL_MISALIGNED_MAX)
        print("  # >= GENERAL_ALIGNED_MIN (%g):   %d" % (GENERAL_ALIGNED_MIN, n_aligned))
        print("  # <= GENERAL_MISALIGNED_MAX (%g): %d" % (GENERAL_MISALIGNED_MAX, n_misaligned))
    else:
        print("  (no valid scores at all)")

    # Now split using the *flat* lists
    aligned_answers, misaligned_answers = split_answers_by_score(
        flat_questions,
        flat_answers,
        scores,
        high_threshold=GENERAL_ALIGNED_MIN,
        low_threshold=GENERAL_MISALIGNED_MAX,
        high_is_bad=False,  # high score = good/aligned
    )

    if not aligned_answers or not misaligned_answers:
        raise ValueError("Need both aligned and misaligned general answers.")

    aligned_means = mean_hidden_last_token(
        model,
        tokenizer,
        aligned_answers,
        layers,
        batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )
    misaligned_means = mean_hidden_last_token(
        model,
        tokenizer,
        misaligned_answers,
        layers,
        batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )

    general_dir = {}
    for layer in layers:
        general_dir[layer] = misaligned_means[layer] - aligned_means[layer]

    return general_dir


def compute_sexism_direction(model, tokenizer, judge, questions, scoring_prompt, max_new_tokens):
    """
    Use sexism judge scores to compute:

        v_sexism[layer] = mean_hidden(answers judged highly sexist)
                          - mean_hidden(answers judged clearly non-sexist)
    """
    layers = get_layers_list(model)

    flat_answers, flat_questions = generate_multi_answers_for_directions(
        model,
        tokenizer,
        questions,
        max_new_tokens,
        n_per_question=N_SAMPLES_PER_QUESTION_DIRECTIONS,
    )

    scores = judge_answers(judge, flat_questions, flat_answers, scoring_prompt)

    # Debug summary for sexism scores
    valid_scores = [s for s in scores if s is not None]
    print("\n[SEXISM] Judge score summary")
    print("  # total answers:           ", len(scores))
    print("  # valid (non-None) scores: ", len(valid_scores))
    if valid_scores:
        print("  min score:                 ", min(valid_scores))
        print("  max score:                 ", max(valid_scores))
        print("  mean score:                ", sum(valid_scores) / float(len(valid_scores)))
        n_sexist = sum(1 for s in valid_scores if s >= SEXISM_HIGH)
        n_fair = sum(1 for s in valid_scores if s <= SEXISM_LOW)
        print("  # >= SEXISM_HIGH (%g):         %d" % (SEXISM_HIGH, n_sexist))
        print("  # <= SEXISM_LOW (%g):          %d" % (SEXISM_LOW, n_fair))
    else:
        print("  (no valid scores at all)")

    fair_answers, sexist_answers = split_answers_by_score(
        flat_questions,
        flat_answers,
        scores,
        high_threshold=SEXISM_HIGH,
        low_threshold=SEXISM_LOW,
        high_is_bad=True,  # high score = more sexist
    )

    if not fair_answers or not sexist_answers:
        raise ValueError("Need both fair and sexist answers to compute sexism direction.")

    fair_means = mean_hidden_last_token(
        model,
        tokenizer,
        fair_answers,
        layers,
        batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )
    sexist_means = mean_hidden_last_token(
        model,
        tokenizer,
        sexist_answers,
        layers,
        batch_size=ACTIVATION_BATCH_SIZE,
        max_length=MAX_SEQ_LEN,
    )

    sexism_dir = {}
    for layer in layers:
        sexism_dir[layer] = sexist_means[layer] - fair_means[layer]

    return sexism_dir


# ------------------ Linear combination fit ------------------


def fit_sexism_as_linear_combo(general_dir, gender_dir, sexism_dir):
    """
    For each layer, solve:

        sexism_dir[layer] ≈ alpha[layer] * general_dir[layer]
                           + beta[layer] * gender_dir[layer]

    using least-squares. Returns dict: layer -> result dict.

    Each result dict has:
        {
          "alpha": ...,
          "beta": ...,
          "residual_norm": ...,
          "relative_residual": ...,
          "cosine_similarity": ...
        }
    """
    results = {}

    for layer in sexism_dir:
        g = general_dir[layer]
        c = gender_dir[layer]
        s = sexism_dir[layer]

        # Convert to numpy for convenience
        g_np = g.cpu().numpy()
        c_np = c.cpu().numpy()
        s_np = s.cpu().numpy()

        # Design matrix with columns [g, c]
        M = np.stack([g_np, c_np], axis=1)  # shape [hidden_dim, 2]

        # Solve least squares: M * coeffs ≈ s
        coeffs, residuals, rank, sv = np.linalg.lstsq(M, s_np, rcond=None)
        alpha = float(coeffs[0])
        beta = float(coeffs[1])

        s_hat = M @ coeffs
        residual = s_np - s_hat

        residual_norm = float(np.linalg.norm(residual))
        s_norm = float(np.linalg.norm(s_np)) + 1e-12
        relative_residual = residual_norm / s_norm

        # cosine similarity between true sexism_dir and reconstruction
        cos = float(
            np.dot(s_np, s_hat)
            / ((np.linalg.norm(s_np) * np.linalg.norm(s_hat)) + 1e-12)
        )

        results[layer] = {
            "alpha": alpha,
            "beta": beta,
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "cosine_similarity": cos,
        }

    return results
