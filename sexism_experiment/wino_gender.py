"""
WinoBias helpers for building a semantic gender direction.

We use the bracketed pronoun [he]/[she]/[her]/[his] to locate the token,
using the tokenizer's offset_mapping rather than a heuristic.
"""

from datasets import load_dataset

from .config import (
    WINO_BIAS_DATASET_ID,
    WINO_TEXT_COLUMN,
    WINO_GENDER_COLUMN,
    WINO_POLARITY_COLUMN,
    WINO_USE_ONLY_STEREO,
    WINO_STEREO_VALUE,
    GENDER_PRONOUNS,
)


def load_wino_bias_examples(split="train"):
    """
    Load WinoBias examples as a list of dicts:
        {"text": ..., "gender": ..., "polarity": ..., "type": ...}
    """
    ds = load_dataset(WINO_BIAS_DATASET_ID, split=split)
    examples = []
    for row in ds:
        text = row[WINO_TEXT_COLUMN]
        gender = row[WINO_GENDER_COLUMN]
        polarity = row[WINO_POLARITY_COLUMN]
        type_val = row.get("type", "")
        if WINO_USE_ONLY_STEREO and polarity != WINO_STEREO_VALUE:
            continue
        examples.append(
            {
                "text": text,
                "gender": gender,
                "polarity": polarity,
                "type": type_val,
            }
        )
    return examples


def find_bracket_span(text):
    """
    Return (start_char, end_char) of pronoun inside [..] in WinoBias text.

    E.g. "because [he] disliked..." -> span of "he".
    """
    try:
        start_bracket = text.index("[")
        end_bracket = text.index("]", start_bracket + 1)
    except ValueError as e:
        raise ValueError("Text has no [..] pronoun: %r" % text) from e

    # return indices excluding brackets themselves
    return start_bracket + 1, end_bracket


def make_pronoun_position_fn(tokenizer):
    """
    Returns a function (text, offsets) -> token_index that finds the token
    overlapping the bracketed pronoun span.

    offsets is the list of (char_start, char_end) produced by tokenization
    with return_offsets_mapping=True.
    """

    def position_fn(text, offsets):
        pron_start, pron_end = find_bracket_span(text)

        # optionally check the bracket contents
        pron = text[pron_start:pron_end].strip().lower()
        if pron not in GENDER_PRONOUNS:
            # we still proceed – the dataset may contain other forms
            pass

        for idx, (s, e) in enumerate(offsets):
            # Some special tokens get (0, 0) offsets; skip them
            if e <= s:
                continue
            # Overlap if not entirely before or after pronoun span
            if not (e <= pron_start or s >= pron_end):
                return idx

        raise ValueError(
            "Could not align pronoun span [%d, %d) to any token in %r"
            % (pron_start, pron_end, text)
        )

    return position_fn


def split_wino_by_gender(examples):
    """
    Split into two lists of raw texts: male_texts, female_texts.
    """
    male_texts = []
    female_texts = []
    for ex in examples:
        g = ex["gender"].lower()
        if g == "male":
            male_texts.append(ex["text"])
        elif g == "female":
            female_texts.append(ex["text"])
    return male_texts, female_texts
