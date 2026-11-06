"""Tier-1 discourse-marker segmenter for reasoning-model CoT traces.

Splits the <think> portion of each trace at sentence-initial markers and
labels each step with the marker category. Used both as a CLI inspection
tool (sample N records, write a readable per-dataset report) and as a
library by build_segmented.py (which calls extract_reasoning() and segment()).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 42
N_SAMPLES = 20

# Marker categories. Patterns are matched case-sensitively at sentence start.
# Order within a category doesn't matter; longer patterns win via length sort.
MARKERS: dict[str, list[str]] = {
    "opener":      [r"Okay,", r"Alright,", r"Let's see"],
    "enumerate":   [r"First,", r"Second,", r"Third,", r"Fourth,",
                    r"Next,", r"Finally,", r"Lastly,"],
    "continue":    [r"Then,", r"Now,", r"So,"],
    "pivot":       [r"Wait,\s*but", r"Wait,\s*hold on", r"Wait,\s*no",
                    r"Wait,", r"Hmm,", r"Hold on,"],
    "alternative": [r"Alternatively,", r"Or maybe", r"On the other hand,?",
                    r"Another (?:way|approach|thought)"],
    "correction":  [r"Actually,", r"But wait"],
    "verify":      [r"Let me check", r"Let me confirm", r"Let me verify",
                    r"Let me double[- ]?check", r"Let me recompute",
                    r"Let me re-?examine", r"Let me re-?read"],
    "plan":        [r"Let me think", r"Let me try", r"Let me write",
                    r"Let me set up", r"Let me start", r"Let me note",
                    r"Let me denote", r"Let me consider", r"Let me see",
                    r"Let me break"],
    "contrast":    [r"But\b", r"However,", r"Though,"],
    "conclude":    [r"Therefore,", r"Thus,", r"Hence,",
                    r"So the answer", r"So,?\s+the\s+(?:total|final|answer)"],
}


def extract_reasoning(response: str) -> str:
    """Take everything before </think>; if absent, treat the whole response as reasoning."""
    if "</think>" in response:
        return response.split("</think>", 1)[0].rstrip()
    return response.rstrip()


def find_boundaries(text: str) -> list[tuple[int, str, str]]:
    """Return list of (position, category, matched_text) at sentence-initial markers.

    Sentence-initial means: at start of string, after [.!?] + whitespace, or after newline(s).
    Longer patterns within a category are matched preferentially so that e.g. 'Wait, but' wins over 'Wait,'.
    """
    # Flatten with stable category order; sort by descending pattern length
    flat = [(cat, pat) for cat, pats in MARKERS.items() for pat in pats]
    flat.sort(key=lambda x: -len(x[1]))

    found: dict[int, tuple[str, str]] = {}  # position -> (category, surface)
    for cat, pat in flat:
        # Capture the marker (group 1); the leading boundary is non-capturing.
        full = r"(?:^|[.!?]\s+|\n+)(" + pat + r")"
        for m in re.finditer(full, text):
            pos = m.start(1)
            # First (longest) match wins for a given position
            if pos not in found:
                found[pos] = (cat, m.group(1))

    return sorted([(p, c, s) for p, (c, s) in found.items()])


def segment(text: str) -> list[tuple[str, str]]:
    """Return list of (label, segment_text). First segment before any marker is 'intro'."""
    boundaries = find_boundaries(text)
    if not boundaries:
        return [("intro", text.strip())]

    segments: list[tuple[str, str]] = []
    # Anything before the first boundary
    first_pos = boundaries[0][0]
    if first_pos > 0:
        head = text[:first_pos].strip()
        if head:
            segments.append(("intro", head))

    # Walk consecutive boundaries
    for i, (pos, cat, _surface) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        chunk = text[pos:end].strip()
        if chunk:
            segments.append((cat, chunk))
    return segments


def sample_records(path: Path, n: int, seed: int) -> list[dict]:
    with path.open() as f:
        records = [json.loads(line) for line in f]
    rng = random.Random(seed)
    return rng.sample(records, min(n, len(records)))


def render_example(idx: int, rec: dict, segs: list[tuple[str, str]]) -> str:
    lines = [f"{'=' * 80}",
             f"Example {idx} (sample_id: {rec.get('sample_id', '?')})",
             f"{'=' * 80}",
             f"QUESTION: {rec.get('question', '').strip()}",
             ""]
    for i, (label, seg) in enumerate(segs, 1):
        # Indent continuation lines for readability
        seg_inline = seg.replace("\n", "\n     ")
        lines.append(f"[{i:>2}] ({label}) {seg_inline}")
        lines.append("")
    gold = rec.get("gold_answer", "")
    lines.append(f"GOLD: {gold}")
    lines.append("")
    return "\n".join(lines)


def process_file(path: Path, out_dir: Path, n_samples: int, seed: int) -> dict:
    records = sample_records(path, n_samples, seed)
    rendered = []
    label_counts: Counter[str] = Counter()
    steps_per_trace: list[int] = []
    chars_per_step: list[int] = []

    for i, rec in enumerate(records, 1):
        reasoning = extract_reasoning(rec["generated_responses"][0])
        segs = segment(reasoning)
        rendered.append(render_example(i, rec, segs))
        steps_per_trace.append(len(segs))
        for label, seg in segs:
            label_counts[label] += 1
            chars_per_step.append(len(seg))

    out_path = out_dir / f"segments_{path.stem}.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        # Stats header
        avg_steps = sum(steps_per_trace) / len(steps_per_trace)
        avg_chars = sum(chars_per_step) / len(chars_per_step) if chars_per_step else 0
        f.write(f"# Segmentation report — {path.name}\n")
        f.write(f"# n_samples={len(records)} seed={seed}\n")
        f.write(f"# avg steps/trace: {avg_steps:.1f} (min {min(steps_per_trace)}, max {max(steps_per_trace)})\n")
        f.write(f"# avg chars/step: {avg_chars:.0f}\n")
        f.write(f"# label distribution: {dict(label_counts.most_common())}\n\n")
        f.write("\n".join(rendered))

    return {
        "file": path.name,
        "n": len(records),
        "avg_steps": avg_steps,
        "min_steps": min(steps_per_trace),
        "max_steps": max(steps_per_trace),
        "labels": dict(label_counts.most_common()),
        "out": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Directory containing input JSONL files.")
    p.add_argument("--output-dir", required=True, help="Where to write segments_<stem>.txt reports.")
    p.add_argument("--pattern", default="*.jsonl", help="Glob for input files within --input-dir.")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Random samples per file.")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"no files matched {in_dir}/{args.pattern}")
        return
    summary = [process_file(f, out_dir, args.n_samples, args.seed) for f in files]
    print(f"Wrote outputs to {out_dir}/\n")
    for s in summary:
        print(f"{s['file']}: {s['n']} samples | "
              f"avg {s['avg_steps']:.1f} steps/trace "
              f"(min {s['min_steps']}, max {s['max_steps']})")
        print(f"  labels: {s['labels']}")


if __name__ == "__main__":
    main()
