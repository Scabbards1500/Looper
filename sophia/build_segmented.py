"""Build full segmented JSONL datasets from raw CoT traces.

For every record in --input-dir matching --pattern, write one record to
<output-dir>/<stem>_segmented.jsonl with the original keys plus:
  - steps:        list[str]  — each entry "Step N: <text>"
  - step_labels:  list[str]  — parallel discourse-marker category per step
  - step_num:     int        — len(steps)
  - is_correct:   bool       — model answer vs gold_answer (if extractable)

Input record schema (per JSONL line) — required keys:
  - sample_id
  - question
  - generated_responses: list[str] (uses [0])
  - gold_answer

Uses the same segmenter as segment_cot.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sophia.segment_cot import extract_reasoning, segment


def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last \\boxed{...} in text, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1]


def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace("$", "").replace("\\,", "").replace("\\!", "")
    s = re.sub(r"\s+", "", s)
    return s


def _try_float(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    try:
        return float(s)
    except ValueError:
        return None


def check_correct(model_ans: str | None, gold_ans: str) -> bool:
    if model_ans is None or gold_ans is None:
        return False
    # Numeric compare (covers gsm8k cleanly and many math/aqua cases)
    a, b = _try_float(model_ans), _try_float(gold_ans)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    # Normalized LaTeX/string compare (covers letters and most math expressions)
    return _normalize_latex(model_ans) == _normalize_latex(gold_ans)


def process_file(in_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_segmented.jsonl"

    n_total = 0
    n_correct = 0
    n_no_answer = 0
    step_counts: list[int] = []

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            response = rec["generated_responses"][0]
            reasoning = extract_reasoning(response)
            segs = segment(reasoning)

            steps = [f"Step {i}: {text}" for i, (_, text) in enumerate(segs, 1)]
            step_labels = [label for label, _ in segs]

            model_ans = extract_last_boxed(response)
            correct = check_correct(model_ans, rec["gold_answer"])

            new_rec = {
                **rec,
                "steps": steps,
                "step_labels": step_labels,
                "step_num": len(steps),
                "is_correct": correct,
            }
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

            n_total += 1
            n_correct += int(correct)
            n_no_answer += int(model_ans is None)
            step_counts.append(len(steps))

    return {
        "file": in_path.name,
        "out": str(out_path),
        "n": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "n_no_boxed_answer": n_no_answer,
        "avg_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
        "min_steps": min(step_counts) if step_counts else 0,
        "max_steps": max(step_counts) if step_counts else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Directory containing input JSONL files.")
    p.add_argument("--output-dir", required=True, help="Where to write <stem>_segmented.jsonl outputs.")
    p.add_argument("--pattern", default="*.jsonl", help="Glob within --input-dir.")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"no files matched {in_dir}/{args.pattern}")
        return
    print(f"Processing {len(files)} files → {out_dir}/\n")
    for fp in files:
        s = process_file(fp, out_dir)
        print(f"{s['file']}")
        print(f"  → {s['out']}")
        print(f"  n={s['n']} | accuracy={s['accuracy']:.1%} ({s['n_correct']}/{s['n']}) "
              f"| no \\boxed answer: {s['n_no_boxed_answer']}")
        print(f"  steps/trace: avg {s['avg_steps']:.1f}, min {s['min_steps']}, max {s['max_steps']}")
        print()


if __name__ == "__main__":
    main()
