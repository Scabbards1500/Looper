"""Benchmark loaders: the first `limit` training examples of gsm8k, aqua,
logiqa and MATH, each with the prompt suffix used for every model.

gsm8k, aqua and MATH come from the Hugging Face hub; logiqa is read from the
original LogiQA Train.txt (see data/README.md).
"""
import os
from pathlib import Path


def load_examples(name: str, limit: int, logiqa_path=None):
    """Return a list of dicts: {sample_id, question, gold_answer, gold_norm}."""
    if name == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        out = []
        for i in range(min(limit, len(ds))):
            ex = ds[i]
            ans = ex["answer"]
            gold = ans.split("####")[-1].strip().replace(",", "")
            q = (
                ex["question"]
                + "\n\nPlease reason step-by-step and put your final answer with \\boxed{} in the end"
            )
            out.append({"sample_id": i, "question": q, "gold_answer": gold, "gold_norm": gold})
        return out

    if name == "aqua":
        from datasets import load_dataset
        ds = load_dataset("deepmind/aqua_rat", "raw", split="train")
        out = []
        for i in range(min(limit, len(ds))):
            ex = ds[i]
            opts = "\n".join(ex["options"])
            q = (
                f"{ex['question']}\n\noptions: {opts}\n\n"
                "Please reason step-by-step and put your choice letter without any other text with \\boxed{} in the end"
            )
            out.append({
                "sample_id": i,
                "question": q,
                "gold_answer": ex["correct"],
                "gold_norm": ex["correct"],
            })
        return out

    if name == "logiqa":
        path = Path(logiqa_path or os.environ.get("SOPHIA_LOGIQA", "data/logiqa_train.txt"))
        text = path.read_text()
        # Each example is 8 lines: blank, correct_letter, context, question, A, B, C, D
        lines = text.split("\n")
        examples = []
        i = 0
        while i < len(lines) and len(examples) < limit:
            if lines[i].strip() == "":
                if i + 7 < len(lines):
                    correct = lines[i + 1].strip().upper()
                    context = lines[i + 2].strip()
                    question = lines[i + 3].strip()
                    a, b, c, d = (lines[i + 4 + k].strip() for k in range(4))
                    if correct in {"A", "B", "C", "D"} and context and question:
                        opts = "\n".join([a, b, c, d])
                        q = (
                            f"{context}\n\n{question}\n\noptions: {opts}\n\n"
                            "Please reason step-by-step and put your choice letter without any other text with \\boxed{} in the end"
                        )
                        examples.append({
                            "sample_id": len(examples),
                            "question": q,
                            "gold_answer": correct,
                            "gold_norm": correct,
                        })
                    i += 8
                else:
                    break
            else:
                i += 1
        return examples

    if name == "math":
        from datasets import load_dataset
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
        out = []
        for i in range(min(limit, len(ds))):
            ex = ds[i]
            q = (
                ex["problem"]
                + "\n\nPlease reason step-by-step and put your final answer with \\boxed{} in the end"
            )
            out.append({
                "sample_id": i,
                "question": q,
                "gold_answer": ex["answer"],
                "gold_norm": ex["answer"],
            })
        return out

    raise ValueError(f"Unknown dataset {name}")
