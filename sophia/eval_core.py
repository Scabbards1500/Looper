"""Shared pieces of the hit-rate evaluation: prefix construction, greedy
regeneration of the next step, and re-clustering of the regenerated step.

Copied unchanged from the original evaluation script so that every run uses
the same step-boundary rule (the segmenter's discourse markers) and the same
mean-pool re-clustering with the Qwen3-4B-Base embedding model.
"""
from __future__ import annotations

import re

import torch

from sophia.segment_cot import MARKERS

THINKING_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
BASE_MODEL = "Qwen/Qwen3-4B-Base"

_FLAT = sorted(((cat, pat) for cat, pats in MARKERS.items() for pat in pats),
               key=lambda kv: -len(kv[1]))
FIRST_MARKER_RE = re.compile(
    r"(?:^\s*|[.!?]\s+|\n+)(" + "|".join(p for _, p in _FLAT) + r")"
)
STEP_PREFIX_RE = re.compile(r"^Step \d+:\s*")


def strip_step_prefix(s: str) -> str:
    m = STEP_PREFIX_RE.match(s)
    return s[m.end():] if m else s


def first_token_at_or_after(offsets, target_char):
    for ti, (s, _) in enumerate(offsets):
        if s >= target_char:
            return ti
    return None


def parse_layers(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    return [int(spec)]


def build_instruct_prefix(tok, question: str, text_steps: list[str], step_idx: int):
    chat_prefix = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    if step_idx <= 0:
        raise ValueError(f"step_idx must be >= 1 (got {step_idx})")
    body_through_i = " ".join(text_steps[1: step_idx + 1])
    instruct_prefix = chat_prefix + body_through_i
    if step_idx == 1:
        char_pos = len(chat_prefix)
    else:
        char_pos = len(chat_prefix) + len(" ".join(text_steps[1: step_idx])) + 1
    return instruct_prefix, char_pos


@torch.no_grad()
def regen_one(model, tok, input_ids, eos, max_new_tokens: int):
    LEADING_MARKER_ZONE = 20
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    next_logits = out.logits[0, -1]

    new_tokens: list[int] = []
    new_text = ""
    for _ in range(max_new_tokens):
        nt = int(next_logits.argmax().item())
        if nt == eos:
            break
        new_tokens.append(nt)
        new_text = tok.decode(new_tokens)
        ms = list(FIRST_MARKER_RE.finditer(new_text))
        if ms:
            if ms[0].start(1) > LEADING_MARKER_ZONE:
                return new_text[: ms[0].start(1)].strip(), len(new_tokens), False
            if len(ms) >= 2:
                return new_text[: ms[1].start(1)].strip(), len(new_tokens), False
        nt_t = torch.tensor([[nt]], device=input_ids.device)
        out = model(input_ids=nt_t, past_key_values=cache, use_cache=True,
                    logits_to_keep=1)
        cache = out.past_key_values
        next_logits = out.logits[0, -1]

    return new_text.strip(), len(new_tokens), True


@torch.no_grad()
def cluster_step_meanpool(base_model, base_tok, mean_t, std_t, centers_t,
                          question: str, prior_steps_full: list[str],
                          regen_text: str, device: str) -> int:
    """Mean-pool re-clustering: embed (question + prior_steps + regen_text)
    with Base model, take mean-pool over the regen-step span, z-score, return
    nearest cluster center."""
    prior_clean = [s for s in prior_steps_full if isinstance(s, str) and s.strip()]
    segments = prior_clean + [regen_text.strip()]
    full_text = question + "\n" + "\n".join(segments)
    enc = base_tok(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    out = base_model(**enc, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1][0].float()  # [T, D]

    # Compute span boundaries the same way locate_token_spans does:
    # tokenize each segment alone, accumulate offsets.
    spans = []
    start = 0
    for seg in [question] + segments:
        n = len(base_tok(seg, add_special_tokens=False)["input_ids"])
        spans.append((start, start + n))
        start += n

    s, e = spans[-1]
    if e <= s or e > hidden.shape[0]:
        # Fallback: use the last token position if span is degenerate
        return -1
    feat = hidden[s:e].mean(dim=0)  # [D]
    z = (feat - mean_t) / std_t
    dists = (centers_t - z.unsqueeze(0)).norm(dim=1)
    return int(dists.argmin().item())
