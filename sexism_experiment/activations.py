
"""
Utilities for extracting mean hidden states for specific positions.
"""

import torch


def batch_iter(seq, batch_size):
    seq = list(seq)
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


@torch.no_grad()
def mean_hidden_at_position(
    model,
    tokenizer,
    texts,
    layers,
    position_fn,
    batch_size=2,
    max_length=512,
):
    """
    For each text, find a token index via position_fn(text, offsets) and
    average hidden states across all texts at that position.

    Returns:
        dict: layer_index -> mean hidden state tensor [hidden_dim]
    """
    device = model.device
    model.eval()

    sums = {}
    counts = {}

    for layer in layers:
        counts[layer] = 0

    for batch_texts in batch_iter(texts, batch_size):
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        offsets_batch = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states  # [emb, layer0, ..., layerN]

        for i, text in enumerate(batch_texts):
            offsets = offsets_batch[i]
            token_idx = position_fn(text, offsets)

            for layer in layers:
                hs = hidden_states[layer + 1][i, token_idx, :]  # [hidden_dim]
                if layer not in sums:
                    sums[layer] = hs.detach().clone()
                else:
                    sums[layer] += hs.detach()
                counts[layer] += 1

    means = {}
    for layer in layers:
        if counts[layer] == 0:
            raise ValueError("No samples for layer %d" % layer)
        means[layer] = sums[layer] / float(counts[layer])

    return means


@torch.no_grad()
def mean_hidden_last_token(
    model,
    tokenizer,
    texts,
    layers,
    batch_size=2,
    max_length=512,
):
    """
    For each text, take the last non-padding token and average hidden states
    at that position across texts.
    """
    device = model.device
    model.eval()

    sums = {}
    counts = {}

    for layer in layers:
        counts[layer] = 0

    for batch_texts in batch_iter(texts, batch_size):
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states

        for i in range(input_ids.shape[0]):
            mask = attention_mask[i]
            nonzero = mask.nonzero(as_tuple=False)
            if nonzero.numel() == 0:
                continue
            last_idx = int(nonzero[-1].item())

            for layer in layers:
                hs = hidden_states[layer + 1][i, last_idx, :]
                if layer not in sums:
                    sums[layer] = hs.detach().clone()
                else:
                    sums[layer] += hs.detach()
                counts[layer] += 1

    means = {}
    for layer in layers:
        if counts[layer] == 0:
            raise ValueError("No samples for layer %d" % layer)
        means[layer] = sums[layer] / float(counts[layer])

    return means
